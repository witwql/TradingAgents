"""LangChain callback handler streaming per-agent activity to the dashboard.

Attaches to the graph's runtime config; LangChain's context propagation makes
every nested invocation fire through here, including LLM calls inside agent
nodes and ToolNode executions. Attribution uses two mechanisms:

- start events carry ``metadata["langgraph_node"]`` directly (verified on
  langgraph 1.2.x: node functions run inside the graph's Runnable context, so
  even bare ``chain.invoke(state)`` inherits it),
- end events only carry ``run_id``, so starts register a run_id -> node map.

Every hook is failure-isolated: the callback channel must never crash or slow
the analysis. Payload strings are hard-truncated before they reach SQLite/SSE.
"""

import logging
import time

from langchain_core.callbacks import BaseCallbackHandler

from .runner import TaskAbandoned

logger = logging.getLogger(__name__)

INPUT_TAIL_CHARS = 220
TEXT_TAIL_CHARS = 600
TOOL_TAIL_CHARS = 500

# Minimum spacing between llm_progress heartbeat events. Streaming LLM calls
# emit no boundary events for their whole duration; without a heartbeat the
# task watchdog reads a legit 5-20 min generation as a wedged worker.
HEARTBEAT_INTERVAL = 60.0


def _tail(value, limit):
    s = str(value or "")
    return ("…" if len(s) > limit else "") + s[-limit:]


def _flat(content):
    if isinstance(content, list | tuple):
        return " ".join(_flat(part) for part in content)
    return content


class AgentStreamHandler(BaseCallbackHandler):
    """Emits llm_start / llm_end / tool_start / tool_end events."""

    raise_error = False

    def __init__(self, emit):
        self._emit = emit
        self._runs: dict[str, dict] = {}
        # -inf so the very first token emits even in the first HEARTBEAT_INTERVAL
        # seconds of process uptime, when monotonic() is still tiny.
        self._last_heartbeat = float("-inf")

    # -- attribution helpers --------------------------------------------------
    @staticmethod
    def _node(metadata) -> str | None:
        try:
            return (metadata or {}).get("langgraph_node")
        except AttributeError:
            return None

    def _remember(self, run_id, payload):
        if run_id is not None:
            self._runs[str(run_id)] = payload

    def _recall(self, run_id):
        return self._runs.pop(str(run_id), None) if run_id is not None else None

    @staticmethod
    def _model_name(serialized) -> str:
        try:
            kwargs = (serialized or {}).get("kwargs", {})
            for key in ("model_name", "model", "deployment_name"):
                if kwargs.get(key):
                    return str(kwargs[key])
            name = (serialized or {}).get("name")
            return str(name) if name else "unknown-model"
        except Exception:
            return "unknown-model"

    @staticmethod
    def _tool_name(serialized) -> str:
        try:
            if isinstance(serialized, dict):
                nm = serialized.get("name") or (serialized.get("kwargs", {}) or {}).get("name")
                return str(nm) if nm else "unknown-tool"
        except Exception:
            pass
        return "unknown-tool"

    # -- chat models -----------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id=None,
                            parent_run_id=None, tags=None, metadata=None,
                            **kwargs):
        try:
            node = self._node(metadata)
            model = self._model_name(serialized)
            last = messages[-1] if messages else None
            last_content = getattr(last, "content", "") or (
                last[1] if isinstance(last, tuple) and len(last) > 1 else ""
            )
            self._remember(run_id, {"kind": "llm", "node": node})
            self._emit("llm_start", {
                "node": node,
                "model": model,
                "msgs": len(messages),
                "input": _tail(_flat(last_content), INPUT_TAIL_CHARS),
            })
        except Exception as exc:
            logger.debug("agent stream llm_start failed: %s", exc)

    def on_llm_new_token(self, token, *, run_id=None, parent_run_id=None,
                         tags=None, metadata=None, **kwargs):
        # Throttled heartbeat: resets the task watchdog's idle clock during a
        # long streaming generation, which emits no boundary events otherwise.
        now = time.monotonic()
        if now - self._last_heartbeat < HEARTBEAT_INTERVAL:
            return
        self._last_heartbeat = now
        try:
            info = self._runs.get(str(run_id)) if run_id is not None else None
            self._emit("llm_progress", {"node": (info or {}).get("node")})
        except Exception:
            logger.debug("agent stream llm_progress failed")

    def on_llm_error(self, error, *, run_id=None, parent_run_id=None,
                     metadata=None, **kwargs):
        info = self._recall(run_id)
        try:
            self._emit("llm_error", {
                "node": (info or {}).get("node"),
                "error": _tail(error, TOOL_TAIL_CHARS),
            })
        except Exception:
            logger.debug("agent stream llm_error failed")

    def on_llm_end(self, response, *, run_id=None, parent_run_id=None,
                   tags=None, **kwargs):
        info = self._recall(run_id)
        node = (info or {}).get("node")
        text = ""
        tool_calls: list[str] = []
        reasoning = ""
        usage_tokens = None
        try:
            gens = list(response.generations or [])
            gen = gens[-1][-1] if gens else None
            text = getattr(gen, "text", "") or ""
            message = getattr(gen, "message", None)
            calls = getattr(message, "tool_calls", None) or []
            tool_calls = [str(c.get("name")) for c in calls if isinstance(c, dict)]
            extra = getattr(message, "additional_kwargs", None) or {}
            reasoning = str(extra.get("reasoning_content") or "")
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage") or {}
            usage_tokens = token_usage.get("total_tokens")
        except Exception:
            logger.debug("agent stream llm_end extraction failed", exc_info=True)

        try:
            self._emit("llm_end", {
                "node": node,
                "text": _tail(text, TEXT_TAIL_CHARS),
                "text_len": len(str(text)),
                "reasoning": _tail(reasoning, TOOL_TAIL_CHARS) if reasoning else "",
                "tool_calls": tool_calls,
                "tokens": usage_tokens,
            })
        except Exception:
            logger.debug("agent stream llm_end failed")

    # -- tools ---------------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id=None,
                      parent_run_id=None, tags=None, metadata=None,
                      **kwargs):
        try:
            node = self._node(metadata)
            tool = self._tool_name(serialized)
            self._remember(run_id, {"kind": "tool", "node": node, "tool": tool})
            self._emit("tool_start", {
                "node": node,
                "tool": tool,
                "args": _tail(input_str, INPUT_TAIL_CHARS),
            })
        except Exception as exc:
            logger.debug("agent stream tool_start failed: %s", exc)

    def on_tool_error(self, error, *, run_id=None, parent_run_id=None,
                      metadata=None, **kwargs):
        info = self._recall(run_id)
        try:
            self._emit("tool_error", {
                "node": (info or {}).get("node"),
                "tool": (info or {}).get("tool"),
                "error": _tail(error, TOOL_TAIL_CHARS),
            })
        except Exception:
            logger.debug("agent stream tool_error failed")

    def on_tool_end(self, output, *, run_id=None, parent_run_id=None,
                    tags=None, **kwargs):
        info = self._recall(run_id)
        out_text = ""
        try:
            content = getattr(output, "content", output)
            out_text = _flat(content)
        except Exception:
            out_text = str(output)
        try:
            self._emit("tool_end", {
                "node": (info or {}).get("node"),
                "tool": (info or {}).get("tool"),
                "result": _tail(out_text, TOOL_TAIL_CHARS),
                "size": len(out_text),
            })
        except Exception:
            logger.debug("agent stream tool_end failed")


class AbandonmentGuard(BaseCallbackHandler):
    """Aborts a run whose task the watchdog already failed.

    Unlike the stream handler, ``raise_error`` is True: the TaskAbandoned
    raised in ``on_chat_model_start`` propagates out of the chat model
    invocation — before any HTTP request — and unwinds the graph. This is
    the only reliable choke point: event callbacks and the progress hook
    swallow exceptions by design, so a wedged/abandoned worker could
    otherwise grind through the remaining graph for hours, burning tokens
    while its task row already reads failed.
    """

    raise_error = True

    def __init__(self, is_abandoned):
        self._is_abandoned = is_abandoned

    def on_chat_model_start(self, serialized, messages, *, run_id=None,
                            parent_run_id=None, tags=None, metadata=None,
                            **kwargs):
        if self._is_abandoned():
            raise TaskAbandoned(
                "task already failed by the watchdog; aborting the run"
            )


__all__ = ["AgentStreamHandler", "AbandonmentGuard"]
