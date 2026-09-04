"""Bind-time tool gating and the analyst tool-round budget.

Two determinism mechanisms for the analyst phase:

- ``tool_available`` / ``available_tools`` decide which tools get bound to an
  analyst LLM and registered in the tool nodes. A tool is dropped when the
  config disables it by name (``disabled_tools``) or when its data vendor is
  known-unconfigured (FRED without ``FRED_API_KEY``). Gating at bind time
  removes the failure mode where the analyst burns a tool round-trip plus an
  LLM digestion round on a call that can only fail; the prompt is built from
  the same filtered list so it never advertises an unbound tool.
- ``analyst_tool_budget`` reports the per-analyst tool-call-round cap
  (``analyst_max_tool_rounds``); 0 or invalid means unlimited.
"""

import os

from langchain_core.messages import AIMessage

from tradingagents.dataflows.config import get_config

# Predicates mirroring the vendor layer's own "not configured" checks — same
# source of truth, evaluated locally so binding never triggers a network call.
_VENDOR_KEYED_TOOLS = {
    "get_macro_indicators": lambda: bool(os.getenv("FRED_API_KEY")),
}


def _disabled_names(config: dict | None) -> set[str]:
    raw = (config or get_config()).get("disabled_tools", "")
    return {name.strip() for name in raw.split(",") if name.strip()}


def tool_available(name: str, config: dict | None = None) -> bool:
    if name in _disabled_names(config):
        return False
    check = _VENDOR_KEYED_TOOLS.get(name)
    if check is not None and not check():
        return False
    return True


def available_tools(tools: list, config: dict | None = None) -> list:
    return [tool for tool in tools if tool_available(tool.name, config)]


def analyst_tool_budget(config: dict | None = None) -> int:
    raw = (config or get_config()).get("analyst_max_tool_rounds", 0)
    try:
        cap = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(cap, 0)


def tool_rounds_used(messages: list) -> int:
    """Count LLM turns in ``messages`` that issued tool calls.

    One round == one analyst invocation that produced ``tool_calls`` (the
    following tool-result turn is a ToolMessage, not an AIMessage, so it is
    not double-counted).
    """
    return sum(
        1
        for m in messages
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )
