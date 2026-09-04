"""Parallel-analyst topology: channels, tool gating, tool-round cap, stages.

Covers the v2 graph topology where the five analysts run as concurrent
LangGraph branches, each on its own ``messages_<key>`` channel, with
bind-time tool gating (``disabled_tools`` / unconfigured vendors) and a
per-analyst tool-call-round cap.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

from tradingagents.agents.analysts.fundamentals_analyst import (
    create_fundamentals_analyst,
)
from tradingagents.agents.analysts.global_macro_analyst import (
    create_global_macro_analyst,
)
from tradingagents.agents.analysts.market_analyst import create_market_analyst
from tradingagents.agents.analysts.news_analyst import create_news_analyst
from tradingagents.agents.utils.tool_availability import (
    analyst_tool_budget,
    available_tools,
    tool_available,
    tool_rounds_used,
)
from tradingagents.dataflows.config import config_scope
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.setup import GraphSetup


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class _BoundFakeChat(GenericFakeChatModel):
    """Queued replies; bind_tools is a no-op returning the same model."""

    def bind_tools(self, tools, **kwargs):
        return self


# Unique marker -> reply text; lets one shared LLM serve every parallel
# branch deterministically regardless of invocation interleaving.
_PROMPT_TAGS = {
    "news researcher": "NEWS 报告正文",
    "trading assistant": "MARKET 报告正文",
    "financial market sentiment analyst": "SENTIMENT 报告正文",
    "analyzing fundamental information": "FUNDAMENTALS 报告正文",
    "the Global Macro Analyst": "MACRO 报告正文",
}


class _TagLLM(BaseChatModel):
    """Replies keyed off prompt markers so per-branch outputs are assertable."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = " ".join(str(getattr(m, "content", m)) for m in messages)
        for marker, reply in _PROMPT_TAGS.items():
            if marker in text:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="通用团队决策文本"))])

    @property
    def _llm_type(self) -> str:
        return "tag-fake"

    def bind_tools(self, tools, **kwargs):
        return self


class _FixedLLM(BaseChatModel):
    """Always the same reply; no structured-output support."""

    reply: str

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    @property
    def _llm_type(self) -> str:
        return "fixed-fake"

    def bind_tools(self, tools, **kwargs):
        return self


# ---------------------------------------------------------------------------
# tool gating
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestToolAvailability:
    def test_disabled_tools_config_blocks_tool(self):
        with config_scope({"disabled_tools": "get_prediction_markets, get_news"}):
            assert not tool_available("get_prediction_markets")
            assert not tool_available("get_news")
            assert tool_available("get_global_news")

    def test_fred_gated_without_api_key(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        assert not tool_available("get_macro_indicators")

    def test_freed_with_api_key(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "demo")
        assert tool_available("get_macro_indicators")

    def test_available_tools_filters_by_name(self, monkeypatch):
        class _T:
            def __init__(self, name):
                self.name = name

        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with config_scope({"disabled_tools": "get_prediction_markets"}):
            kept = available_tools(
                [_T("get_news"), _T("get_macro_indicators"), _T("get_prediction_markets")]
            )
        assert [t.name for t in kept] == ["get_news"]

    def test_budget_parses_config(self):
        with config_scope({"analyst_max_tool_rounds": 5}):
            assert analyst_tool_budget() == 5
        with config_scope({"analyst_max_tool_rounds": "not-a-number"}):
            assert analyst_tool_budget() == 0
        with config_scope({"analyst_max_tool_rounds": -2}):
            assert analyst_tool_budget() == 0

    def test_tool_rounds_used_counts_only_tool_calling_turns(self):
        from langchain_core.messages import HumanMessage, ToolMessage

        msgs = [
            HumanMessage(content="go"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "a"}]),
            ToolMessage(content="r", tool_call_id="a"),
            AIMessage(content="plain answer"),
        ]
        assert tool_rounds_used(msgs) == 1


# ---------------------------------------------------------------------------
# config defaults
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_new_config_defaults():
    assert DEFAULT_CONFIG["analyst_max_tool_rounds"] == 3
    assert DEFAULT_CONFIG["disabled_tools"] == ""


@pytest.mark.unit
def test_new_config_env_coercion(monkeypatch):
    import importlib

    monkeypatch.setenv("TRADINGAGENTS_ANALYST_MAX_TOOL_ROUNDS", "5")
    monkeypatch.setenv("TRADINGAGENTS_DISABLED_TOOLS", "get_news")
    import tradingagents.default_config as dc

    dc = importlib.reload(dc)
    assert dc.DEFAULT_CONFIG["analyst_max_tool_rounds"] == 5
    assert dc.DEFAULT_CONFIG["disabled_tools"] == "get_news"
    importlib.reload(dc)  # restore for other tests


# ---------------------------------------------------------------------------
# msg_delete channel parameter
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestMsgDeleteChannel:
    def test_named_channel_only_touches_that_channel(self):
        from tradingagents.agents.utils.agent_utils import create_msg_delete

        state = {
            "messages_news": [
                HumanMessage(content="old", id="h1"),
                AIMessage(content="reply", id="a1"),
            ],
            "company_of_interest": "EC",
            "trade_date": "2026-05-28",
        }
        result = create_msg_delete("messages_news")(state)
        assert set(result) == {"messages_news"}
        removals = [m for m in result["messages_news"] if isinstance(m, RemoveMessage)]
        assert len(removals) == 2
        assert isinstance(result["messages_news"][-1], HumanMessage)


# ---------------------------------------------------------------------------
# analyst channel isolation (node level)
# ---------------------------------------------------------------------------
_MIN_STATE = {
    "trade_date": "2026-01-15",
    "company_of_interest": "600519.SS",
    "asset_type": "stock",
    "instrument_context": "The instrument to analyze is `600519.SS`.",
}


@pytest.mark.unit
class TestAnalystChannelIsolation:
    @pytest.mark.parametrize(
        "factory,channel,report",
        [
            (create_market_analyst, "messages_market", "market_report"),
            (create_news_analyst, "messages_news", "news_report"),
            (create_fundamentals_analyst, "messages_fundamentals", "fundamentals_report"),
            (create_global_macro_analyst, "messages_macro", "macro_report"),
        ],
    )
    def test_analyst_writes_only_its_own_channel(self, factory, channel, report):
        llm = _BoundFakeChat(messages=iter([AIMessage(content="报告正文")]))
        state = {
            **_MIN_STATE,
            channel: [("human", "600519.SS 分析")],
        }
        result = factory(llm)(state)
        assert set(result) == {channel, report}
        assert result[report] == "报告正文"
        assert isinstance(result[channel][0], AIMessage)

    def test_sentiment_writes_only_its_own_channel(self, monkeypatch):
        from tradingagents.agents.analysts import sentiment_analyst as sa

        monkeypatch.setattr(sa, "fetch_stocktwits_messages", lambda *a, **k: "st")
        monkeypatch.setattr(sa, "fetch_reddit_posts", lambda *a, **k: "rd")
        monkeypatch.setattr(sa.get_news, "func", lambda *a, **k: "news", raising=False)

        llm = _FixedLLM(reply="情绪报告正文")
        state = {**_MIN_STATE, "messages_social": []}
        result = sa.create_sentiment_analyst(llm)(state)
        assert set(result) == {"messages_social", "sentiment_report"}
        assert result["sentiment_report"] == "情绪报告正文"


# ---------------------------------------------------------------------------
# tool-round cap (node level, driving the router loop like langgraph does)
# ---------------------------------------------------------------------------
_TOOL_CALL = {"name": "get_stock_data", "args": {"ticker": "600519.SS"}, "id": "c1"}

_CAP_CASES = [
    (create_market_analyst, "messages_market", "market_report"),
    (create_news_analyst, "messages_news", "news_report"),
    (create_fundamentals_analyst, "messages_fundamentals", "fundamentals_report"),
    (create_global_macro_analyst, "messages_macro", "macro_report"),
]


def _drive_tool_loop(node, channel, tool_call_reply):
    """Run one analyst branch by hand: node → (fake tools) → node → ..."""
    state = {**_MIN_STATE, channel: [("human", "600519.SS 分析")]}
    results = []
    for _ in range(3):
        result = node(state)
        results.append(result)
        last = result[channel][0]
        if not getattr(last, "tool_calls", None):
            break
        tool_msgs = [ToolMessage(content="工具输出", tool_call_id=tc["id"])
                     for tc in last.tool_calls]
        state = {**state, channel: state[channel] + [last, *tool_msgs]}
    return results


@pytest.mark.unit
class TestAnalystToolRoundCap:
    @pytest.mark.parametrize("factory,channel,report", _CAP_CASES)
    def test_report_written_within_cap_plus_one_calls(self, factory, channel, report):
        tool_call = AIMessage(content="", tool_calls=[_TOOL_CALL])
        llm = _BoundFakeChat(messages=iter([tool_call, tool_call, AIMessage(content="最终报告")]))
        node = factory(llm)
        with config_scope({"analyst_max_tool_rounds": 2}):
            results = _drive_tool_loop(node, channel, tool_call)
        assert len(results) == 3  # cap 2 + 1 forced final call
        assert results[-1][report] == "最终报告"
        assert not results[0][report] and not results[1][report]

    @pytest.mark.parametrize("factory,channel,report", _CAP_CASES)
    def test_zero_cap_disables_limit(self, factory, channel, report):
        tool_call = AIMessage(content="", tool_calls=[_TOOL_CALL])
        llm = _BoundFakeChat(messages=iter([tool_call]))
        node = factory(llm)
        with config_scope({"analyst_max_tool_rounds": 0}):
            result = node({**_MIN_STATE, channel: [("human", "600519.SS 分析")]})
        assert result[report] == ""  # still in tool mode → no report yet
        assert result[channel][0].tool_calls


# ---------------------------------------------------------------------------
# full-graph parallel topology
# ---------------------------------------------------------------------------
def _fake_tool_node(state):
    """Stands in for a ToolNode; only reached if an analyst emits tool_calls."""
    raise AssertionError("fake tool node must not execute in the parallel-topology test")


@pytest.mark.unit
class TestParallelTopology:
    def test_all_analyst_branches_reach_reports_and_pipeline_completes(self, monkeypatch):
        from tradingagents.agents.analysts import sentiment_analyst as sa

        monkeypatch.setattr(sa, "fetch_stocktwits_messages", lambda *a, **k: "st")
        monkeypatch.setattr(sa, "fetch_reddit_posts", lambda *a, **k: "rd")
        monkeypatch.setattr(sa.get_news, "func", lambda *a, **k: "news", raising=False)

        tool_nodes = {key: _fake_tool_node for key in ANALYST_NODE_SPECS}
        setup = GraphSetup(_TagLLM(), _TagLLM(), tool_nodes, ConditionalLogic(1, 1))
        graph = setup.setup_graph(
            ["market", "social", "news", "fundamentals", "macro"]
        ).compile()

        state = Propagator().create_initial_state(
            "600519.SS", "2026-01-15", instrument_context="Test identity context"
        )
        final = graph.invoke(state, config={"recursion_limit": 60})

        # every branch produced its own tagged report (proves both isolation
        # and that all five branches completed)
        assert final["market_report"] == "MARKET 报告正文"
        assert final["sentiment_report"] == "SENTIMENT 报告正文"
        assert final["news_report"] == "NEWS 报告正文"
        assert final["fundamentals_report"] == "FUNDAMENTALS 报告正文"
        assert final["macro_report"] == "MACRO 报告正文"

        # the pipeline reached the end through the serial tail
        assert final["final_trade_decision"].strip()
        assert final["investment_plan"].strip()
        assert final["trader_investment_plan"].strip()

        # each private channel was cleared down to its placeholder
        for key in ("messages_market", "messages_social", "messages_news",
                    "messages_fundamentals", "messages_macro"):
            assert len(final[key]) == 1 and isinstance(final[key][0], HumanMessage)

    def test_single_analyst_still_wires(self):
        tool_nodes = {key: _fake_tool_node for key in ANALYST_NODE_SPECS}
        setup = GraphSetup(_TagLLM(), _TagLLM(), tool_nodes, ConditionalLogic(1, 1))
        workflow = setup.setup_graph(["market"])
        compiled = workflow.compile()
        state = Propagator().create_initial_state(
            "600519.SS", "2026-01-15", instrument_context="Test identity context"
        )
        final = compiled.invoke(state, config={"recursion_limit": 60})
        assert final["market_report"] == "MARKET 报告正文"
        assert final["final_trade_decision"].strip()


# ---------------------------------------------------------------------------
# runner stage tracking with parallel analysts
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestRunnerStageTracking:
    def _db(self, tmp_path):
        from server.db import Database

        db = Database(tmp_path / "dash.db")
        return db, db.create_task({"ticker": "600519.SS", "trade_date": "2026-09-01"})

    def test_parallel_analyst_rows_track_independently(self, tmp_path):
        from server.runner import AnalysisRunner, build_stages

        db, task_id = self._db(tmp_path)
        runner = AnalysisRunner({})
        events = []
        runner.reset(task_id, db, build_stages(["market", "news"]))

        runner._advance_stage(db, task_id, lambda k, p: events.append((k, p)),
                              "market", "Market Analyst")
        runner._advance_stage(db, task_id, lambda k, p: events.append((k, p)),
                              "news", "News Analyst")
        runner._advance_stage(db, task_id, lambda k, p: events.append((k, p)),
                              "news", "Msg Clear News")

        status = {s["name"]: s["status"] for s in db.get_stages(task_id)}
        assert status["market"] == "running"
        assert status["news"] == "done"

        runner._advance_stage(db, task_id, lambda k, p: events.append((k, p)),
                              "market", "Msg Clear Market")
        status = {s["name"]: s["status"] for s in db.get_stages(task_id)}
        assert status["market"] == "done"

        kinds = [e[0] for e in events]
        assert kinds == ["stage", "stage", "stage", "stage"]
        payloads = [p for _, p in events]
        assert {"started": "market"} in payloads
        assert {"started": "news"} in payloads
        assert {"completed": "news"} in payloads

    def test_tail_phase_is_serial_and_closes_leftover_analyst_rows(self, tmp_path):
        from server.runner import AnalysisRunner, build_stages

        db, task_id = self._db(tmp_path)
        runner = AnalysisRunner({})
        runner.reset(task_id, db, build_stages(["market", "news"]))
        runner._advance_stage(db, task_id, lambda k, p: None, "market", "Market Analyst")
        runner._advance_stage(db, task_id, lambda k, p: None, "news", "News Analyst")

        # bull only starts after the fan-in barrier, but if a stage row was
        # left running it must be closed when the tail begins
        runner._advance_stage(db, task_id, lambda k, p: None,
                              "bull_researcher", "Bull Researcher")
        status = {s["name"]: s["status"] for s in db.get_stages(task_id)}
        assert status["market"] == "done" and status["news"] == "done"
        assert status["bull_researcher"] == "running"

        runner._advance_stage(db, task_id, lambda k, p: None,
                              "bear_researcher", "Bear Researcher")
        status = {s["name"]: s["status"] for s in db.get_stages(task_id)}
        assert status["bull_researcher"] == "done"
        assert status["bear_researcher"] == "running"

        runner.finish_stages(task_id, db)
        status = {s["name"]: s["status"] for s in db.get_stages(task_id)}
        assert status["bear_researcher"] == "done"
        # every row that actually started is closed; rows never reached stay
        # pending (the run never got there)
        for name in ("market", "news", "bull_researcher", "bear_researcher"):
            assert status[name] == "done"

    def test_finish_stages_closes_still_running_rows(self, tmp_path):
        from server.runner import AnalysisRunner, build_stages

        db, task_id = self._db(tmp_path)
        runner = AnalysisRunner({})
        runner.reset(task_id, db, build_stages(["market"]))
        runner._advance_stage(db, task_id, lambda k, p: None, "market", "Market Analyst")
        runner.finish_stages(task_id, db)
        status = {s["name"]: s["status"] for s in db.get_stages(task_id)}
        assert status["market"] == "done"
