"""Bridge between the dashboard queue and the TradingAgents framework.

One runner invocation = one full multi-agent analysis for a single ticker.
It builds the config (GLM LLM provider, akshare A-share data preset),
maps LangGraph node-completion events onto user-visible workflow stages,
and persists the report tree the framework writes for each run.
"""

import contextlib
import logging
import traceback
from pathlib import Path

from tradingagents.dataflows.config import config_scope
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

logger = logging.getLogger(__name__)

# LangGraph node name -> workflow stage id shown in the UI. Tool nodes and
# msg-clear nodes fold into their owning stage so the pipeline reads as one
# step per agent team member.
NODE_TO_STAGE: dict[str, str] = {
    "Global Macro Analyst": "macro",
    "tools_macro": "macro",
    "Msg Clear Macro": "macro",
    "Market Analyst": "market",
    "tools_market": "market",
    "Msg Clear Market": "market",
    "Sentiment Analyst": "sentiment",
    "tools_social": "sentiment",
    "Msg Clear Sentiment": "sentiment",
    "News Analyst": "news",
    "tools_news": "news",
    "Msg Clear News": "news",
    "Fundamentals Analyst": "fundamentals",
    "tools_fundamentals": "fundamentals",
    "Msg Clear Fundamentals": "fundamentals",
    "Bull Researcher": "bull_researcher",
    "Bear Researcher": "bear_researcher",
    "Research Manager": "research_manager",
    "Trader": "trader",
    "Aggressive Analyst": "risk_debate",
    "Conservative Analyst": "risk_debate",
    "Neutral Analyst": "risk_debate",
    "Portfolio Manager": "portfolio_manager",
}

STAGE_LABELS: dict[str, str] = {
    "market": "市场分析师",
    "macro": "全球宏观分析师",
    "sentiment": "情绪分析师（social）",
    "news": "新闻分析师",
    "fundamentals": "基本面分析师",
    "bull_researcher": "多方研究员",
    "bear_researcher": "空方研究员",
    "research_manager": "研究主管裁决",
    "trader": "交易员决策",
    "risk_debate": "风险团队辩论",
    "portfolio_manager": "组合经理终审",
}

# Analyst config keys -> workflow stage ids (the sentiment analyst's wire key
# stays "social"; its displayed stage is the same agent).
ANALYST_STAGE_KEY = {"market": "market", "social": "sentiment",
                     "news": "news", "fundamentals": "fundamentals",
                     "macro": "macro"}

ALL_ANALYST_KEYS = ("market", "social", "news", "fundamentals", "macro")

# Stage ids of the analyst team. With the parallel graph topology these run
# concurrently, so their stage rows are tracked individually (started when
# the branch's first node completes, done when its "Msg Clear X" completes)
# instead of through the single-slot serial rule.
ANALYST_STAGES = frozenset({"market", "sentiment", "news", "fundamentals", "macro"})


class TaskAbandoned(RuntimeError):
    """Raised to unwind a run whose task the watchdog already failed.

    RuntimeError (not plain Exception) so langgraph's ``default_retry_on``
    returns False — an abandoned task must not burn retry attempts.
    """


def build_stages(analysts: list[str]) -> list[tuple[str, str]]:
    """Ordered [(stage_id, label)] pipeline rows for the selected analysts."""
    stages: list[tuple[str, str]] = [
        (ANALYST_STAGE_KEY[key], STAGE_LABELS[ANALYST_STAGE_KEY[key]])
        for key in ALL_ANALYST_KEYS if key in analysts
    ]
    stages += [
        ("bull_researcher", STAGE_LABELS["bull_researcher"]),
        ("bear_researcher", STAGE_LABELS["bear_researcher"]),
        ("research_manager", STAGE_LABELS["research_manager"]),
        ("trader", STAGE_LABELS["trader"]),
        ("risk_debate", STAGE_LABELS["risk_debate"]),
        ("portfolio_manager", STAGE_LABELS["portfolio_manager"]),
    ]
    return stages


class AnalysisRunner:
    """Executes one task row against the framework; reports via callbacks."""

    def __init__(self, settings: dict[str, str]):
        self.settings = settings
        self._current_stage = None
        self._started_stages = set()
        self._done_stages = set()

    def _build_config(self, task: dict) -> dict:
        config = DEFAULT_CONFIG.copy()
        # GLM LLM (Zhipu). Region switchable: glm-cn uses ZHIPU_CN_API_KEY at
        # open.bigmodel.cn; glm (international) uses ZHIPU_API_KEY at api.z.ai.
        config["llm_provider"] = self.settings.get("glm_region", "glm-cn")
        default_model = self.settings.get("glm_model", "glm-5.3-flash")
        config["deep_think_llm"] = self.settings.get("deep_model", default_model)
        config["quick_think_llm"] = self.settings.get("quick_model", default_model)
        if temp := self.settings.get("temperature"):
            with contextlib.suppress(ValueError):
                config["temperature"] = float(temp)
        config["output_language"] = task.get("output_language", "Chinese") or "Chinese"
        config["max_debate_rounds"] = max(1, int(task.get("debate_rounds", 1)))
        config["max_risk_discuss_rounds"] = max(1, int(task.get("risk_rounds", 1)))

        # A-share data preset. EastMoney's push2his quote host is throttled
        # under load; the Sina and Yahoo Finance hosts are stable, so they
        # are the only members of the price/fundamentals chains. EastMoney's
        # *news* host (search-api) sits on different infrastructure and has
        # been reliable, so the news chain keeps akshare first. Money flow
        # (EM-only history) degrades to a THS snapshot fallback under throttle.
        if task.get("asset_type", "stock") == "stock" and _looks_like_ashare(
            task["ticker"]
        ):
            config["data_vendors"] = {
                "core_stock_apis": "sina,yfinance",
                "technical_indicators": "sina,yfinance",
                "fundamental_data": "sina,yfinance",
                "news_data": "akshare,yfinance",
                "macro_data": "fred",
                "prediction_markets": "polymarket",
            }
            # A-share reports don't use Polymarket odds, and the host has been
            # unreachable from this deployment — each bound call burns a ~30s
            # timeout plus an LLM digestion round. Unbind it outright; non-A-
            # share runs keep it (the polymarket circuit breaker backstops).
            config["disabled_tools"] = "get_prediction_markets"
        return config

    def run(self, task: dict, emit, db, is_abandoned=None) -> dict:
        """Run the analysis. ``emit(type, payload)`` publishes SSE events.

        ``is_abandoned`` — optional zero-arg predicate, polled before every LLM
        call; when it turns True (the watchdog failed this task), the guard
        raises :class:`TaskAbandoned` so the worker stops instead of grinding
        through the remaining hours of graph on a task that is already failed.

        Returns a small result dict (rating/summary/report_dir) the queue can
        persist on success.
        """
        def progress(event: dict):
            node = event.get("node", "")
            stage = NODE_TO_STAGE.get(node)
            self._advance_stage(db, task["id"], emit, stage, node)
            emit(
                "node",
                {"node": node, "stage": stage},
            )

        # Per-agent live feed: forward LLM/tool activity with resolved stage
        # labels so the UI groups the stream without re-mapping names.
        from .agent_stream import AgentStreamHandler, AbandonmentGuard

        def stream_emit(kind: str, payload: dict):
            payload = {**payload, "stage": NODE_TO_STAGE.get(payload.get("node") or "")}
            emit(kind, payload)

        cfg = self._build_config(task)
        # Scoped, not global: vendor calls inside this run see exactly cfg,
        # while the screener/spot threads keep their own view of the world.
        agent_handler = AgentStreamHandler(stream_emit)
        runtime_callbacks = [agent_handler]
        if is_abandoned is not None:
            runtime_callbacks.append(AbandonmentGuard(is_abandoned))
        with config_scope(cfg):
            return self._run_analysis(task, cfg, progress, runtime_callbacks)

    def _run_analysis(self, task, cfg, progress, runtime_callbacks):
        ticker = task["ticker"]
        graph = TradingAgentsGraph(
            selected_analysts=tuple(
                a for a in task.get("analysts", ALL_ANALYST_KEYS) if a in ALL_ANALYST_KEYS
            ),
            debug=False,
            config=cfg,
            progress_callback=progress,
            runtime_callbacks=runtime_callbacks,
        )
        final_state, rating = graph.propagate(ticker, task["trade_date"])

        # Persist the same markdown report tree the CLI saves on completion.
        # write_report_tree returns the consolidated complete_report.md path;
        # the dashboard browses the tree, so keep the tree root instead.
        report_dir: Path | None = None
        try:
            saved = graph.save_reports(final_state, ticker)
            report_dir = saved.parent if (saved.is_file() and saved.name == "complete_report.md") else saved
        except Exception:
            logger.exception("saving report tree failed for %s", ticker)

        summary_source = str(final_state.get("final_trade_decision", "") or "")
        summary_lines = [ln.strip() for ln in summary_source.splitlines() if ln.strip()]
        summary = "\n".join(summary_lines[:12])
        return {
            "rating": str(rating or ""),
            "summary": summary,
            "report_dir": str(report_dir) if report_dir else "",
        }

    def _advance_stage(self, db, task_id, emit, stage: str | None, node: str = ""):
        if not stage:
            return
        if stage in ANALYST_STAGES:
            self._advance_analyst_stage(db, task_id, emit, stage, node)
        else:
            self._advance_serial_stage(db, task_id, emit, stage)

    def _advance_analyst_stage(self, db, task_id, emit, stage: str, node: str):
        started = self._started_stages
        done = self._done_stages
        if node.startswith("Msg Clear"):
            if stage in done:
                return
            done.add(stage)
            db.update_stage(task_id, stage, "done")
            emit("stage", {"completed": stage})
        elif stage not in started:
            started.add(stage)
            db.update_stage(task_id, stage, "running")
            emit("stage", {"started": stage})
            # The task list shows current_stage as a progress label; with
            # concurrent analyst starts the last writer wins, which is fine
            # for display purposes.
            db.update_task(task_id, current_stage=stage)

    def _advance_serial_stage(self, db, task_id, emit, stage: str):
        prev = getattr(self, "_current_stage", None)
        if stage == prev:
            return
        if prev:
            db.update_stage(task_id, prev, "done")
            emit("stage", {"completed": prev})
        db.update_stage(task_id, stage, "running")
        emit("stage", {"started": stage})
        self._current_stage = stage
        db.update_task(task_id, current_stage=stage)
        # The tail phase starts only after the analyst fan-in barrier, so any
        # analyst stage still tracked as running is finished by definition;
        # this is belt-and-braces against event-order surprises.
        for leftover in self._started_stages - self._done_stages:
            self._done_stages.add(leftover)
            db.update_stage(task_id, leftover, "done")
            emit("stage", {"completed": leftover})

    def reset(self, task_id: str, db, stages: list[tuple[str, str]]):
        db.set_stages(task_id, [name for name, _ in stages])
        self._current_stage = None
        self._started_stages = set()
        self._done_stages = set()

    def finish_stages(self, task_id: str, db):
        prev = getattr(self, "_current_stage", None)
        if prev:
            db.update_stage(task_id, prev, "done")
        self._current_stage = None
        for leftover in self._started_stages - self._done_stages:
            self._done_stages.add(leftover)
            db.update_stage(task_id, leftover, "done")


def _looks_like_ashare(ticker: str) -> bool:
    upper = str(ticker).upper().rstrip("+")
    if "." in upper:
        body, _, suffix = upper.partition(".")
        ok_suffix = suffix in ("SS", "SH", "SZ")
    else:
        body, ok_suffix = upper, True
    return len(body) == 6 and body.isdigit() and ok_suffix


def format_exception(exc: BaseException) -> str:
    """Last lines of the full traceback — root causes usually live deeper."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return text.strip()[-900:]
