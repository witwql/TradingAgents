"""Dashboard server tests: SQLite queue lifecycle + API surface, no network.

Runs are exercised through a fake runner so the full submit→claim→execute→
complete pipeline (and its failure path) is verified without any LLM calls.
"""
import contextlib
import time
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.db import Database
from server.queue import TaskQueue
from server.runner import build_stages


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "dash.db")


class FakeRunner:
    """Records runs; can be made to fail specific tickers."""

    calls: list = []
    fail_tickers: set[str] = set()

    def __init__(self, settings):
        self.settings = settings
        self._stage = None

    def reset(self, task_id, db, stages):
        self.calls.append(("reset", task_id))
        db.set_stages(task_id, [name for name, _ in stages])
        self._stage = None

    def run(self, task, emit, db, is_abandoned=None):
        ticker = task["ticker"]
        if ticker in self.fail_tickers:
            raise RuntimeError(f"boom for {ticker}")
        self.calls.append(("run", task["id"]))
        emit("llm_start", {"node": "Market Analyst", "model": "glm-5.2", "msgs": 3, "input": "分析"})
        emit("tool_start", {"node": "Market Analyst", "tool": "get_stock_data", "args": "{x}"})
        emit("tool_end", {"node": "Market Analyst", "tool": "get_stock_data",
                          "result": "csv...", "size": 1234})
        emit("llm_end", {"node": "Market Analyst", "text": "结论：看多", "text_len": 6,
                         "reasoning": "", "tool_calls": ["get_indicators"], "tokens": 512})
        emit("node", {"node": "Trader", "stage": "trader"})
        db.update_task(task["id"], current_stage="portfolio_manager")
        emit("node", {"node": "Portfolio Manager", "stage": "portfolio_manager"})
        self._stage = "portfolio_manager"
        return {"rating": "Final Decision: BUY", "summary": "- 建议：买入\n- 预计区间",
                "report_dir": ""}

    def finish_stages(self, task_id, db):
        self._stage and db.update_stage(task_id, self._stage, "done")
        self._stage = None


def _make(db, workers=1, runner_cls=FakeRunner):
    q = TaskQueue(db, workers=workers, runner_cls=runner_cls)
    app = create_app(db=db, queue=q, start_spot=False)
    return q, TestClient(app)


@pytest.mark.unit
class TestQueueLifecycle:
    def test_submit_creates_one_task_per_ticker(self, db):
        q, client = _make(db)
        r = client.post("/api/tasks", json={"tickers": "600519, 000001.SZ, 600519"})
        assert r.status_code == 201
        ids = r.json()["task_ids"]
        assert len(ids) == 2  # duplicate suppressed
        rows = {t["ticker"] for t in db.list_tasks()}
        assert rows == {"600519.SS", "000001.SZ"}

    def test_full_pipeline_completes(self, db):
        FakeRunner.calls.clear()
        q, client = _make(db)
        ids = client.post("/api/tasks", json={"tickers": ["600519"], "debate_rounds": 2}).json()["task_ids"]
        q._execute(db.get_task(ids[0]))
        task = db.get_task(ids[0])
        assert task["status"] == "completed"
        assert "BUY" in task["rating"]
        stages = {s["name"]: s["status"] for s in db.get_stages(ids[0])}
        assert stages["portfolio_manager"] == "done"
        types = [e["type"] for e in db.events_since(ids[0], 0)]
        for t in ("llm_start", "tool_start", "tool_end", "llm_end"):
            assert t in types, t
        agent_payload = next(e["payload"] for e in db.events_since(ids[0], 0)
                             if e["type"] == "llm_end")
        assert "看多" in agent_payload["text"]
        assert types[-1] == "status"

    def test_failure_records_error_and_status(self, db):
        FakeRunner.fail_tickers = {"000001.SZ"}
        q, _client = _make(db)
        ids = client_submit(q, ["000001.SZ"])
        q._execute(db.get_task(ids[0]))
        task = db.get_task(ids[0])
        assert task["status"] == "failed"
        assert "boom for" in task["error"]

    def test_worker_drains_fifo(self, db):
        class ImmediateQueue(TaskQueue):
            pass
        FakeRunner.calls.clear()
        q = TaskQueue(db, workers=1, runner_cls=FakeRunner)
        q.start()
        try:
            q.submit({"tickers": ["600519", "510300"], "trade_date": "2026-08-27"})
            deadline = time.time() + 10
            while len(FakeRunner.calls) < 4 and time.time() < deadline:
                time.sleep(0.1)
            statuses = [t["status"] for t in db.list_tasks()]
            assert statuses == ["completed", "completed"]
        finally:
            q.stop()

    def test_cancel_pending_only(self, db):
        q, _client = _make(db)
        (tid,) = client_submit(q, ["600519"])
        assert q.cancel(tid) is True
        assert db.get_task(tid)["status"] == "cancelled"

    def test_stages_follow_selected_analysts(self, db):
        names = [n for n, _ in build_stages(["market", "fundamentals"])]
        assert names == ["market", "fundamentals", "bull_researcher", "bear_researcher",
                         "research_manager", "trader", "risk_debate", "portfolio_manager"]


def client_submit(q, tickers):
    return q.submit({"tickers": tickers, "trade_date": "2026-08-27"})


@pytest.mark.unit
class TestApiSurface:
    def test_health_reports_key_absence(self, db, monkeypatch, tmp_path):
        # Detach from any real developer .env: detection walks CWD for the
        # project's dotfile, so run inside a temp dir with env vars cleared.
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ZHIPU_CN_API_KEY", raising=False)
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        _q, client = _make(db)
        body = client.get("/api/health").json()
        assert body["ok"] is True
        assert body["has_zhipu_cn_key"] is False

    def test_task_validation_rejects_bad_analysts(self, db):
        _q, client = _make(db)
        r = client.post("/api/tasks", json={"tickers": ["600519"], "analysts": ["astrology"]})
        assert r.status_code == 422

    def test_detail_404_and_reports_without_dir(self, db):
        q, client = _make(db)
        assert client.get("/api/tasks/missing").status_code == 404
        ids = client_submit(q, ["600519"])  # pending, never executed
        r = client.get(f"/api/tasks/{ids[0]}/reports").json()
        assert r == {"report_dir": "", "files": []}

    def test_settings_roundtrip_and_unknown_key(self, db):
        _q, client = _make(db)
        r = client.put("/api/settings", json={"glm_region": "glm"})
        assert r.json()["glm_region"] == "glm"
        bad = client.put("/api/settings", json={"glm_region": "openai"})
        assert bad.status_code == 422

    def test_favorites_crud(self, db):
        _q, client = _make(db)
        assert client.post("/api/favorites", json={"code": "510300"}).status_code == 201
        body = client.get("/api/favorites").json()
        codes = [f["code"] for f in body["favorites"]]
        assert codes == ["510300"]
        assert "quotes_ready" in body and "quote_ts" in body  # instant-response contract
        client.delete("/api/favorites/510300")
        assert client.get("/api/favorites").json()["favorites"] == []

    def test_index_page_served(self, db):
        _q, client = _make(db)
        html = client.get("/").text
        assert "TradingAgents" in html and "static/app.js" in html


class TestAgentStreamHandler:
    """Real LangGraph run with a fake chat model + tool; assert attribution."""

    def _graph(self, model):
        from typing import TypedDict

        from langchain_core.tools import tool
        from langgraph.graph import END, START, StateGraph

        @tool
        def echo_probe(x: str) -> str:
            """echo back"""
            return f"echo:{x}"

        class S(TypedDict, total=False):
            out: str

        def analyst(state):
            text = model.invoke([("human", "分析")])
            echo_probe.invoke({"x": "RSI"})
            return {"out": str(text)}

        g = StateGraph(S)
        g.add_node("Market Analyst", analyst)
        g.add_edge(START, "Market Analyst")
        g.add_edge("Market Analyst", END)
        return g.compile()

    def test_llm_and_tool_events_attributed_to_node(self):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        from server.agent_stream import AgentStreamHandler

        events = []
        h = AgentStreamHandler(lambda kind, payload: events.append((kind, payload)))
        model = GenericFakeChatModel(messages=iter(["趋势向好，结论：看多。"]))
        graph = self._graph(model)
        list(graph.stream({"out": ""}, config={"callbacks": [h], "recursion_limit": 10}))

        kinds = [k for k, _ in events]
        for expected in ("llm_start", "llm_end", "tool_start", "tool_end"):
            assert expected in kinds, (expected, kinds)

        start = dict(events)["llm_start"]
        end = dict(events)["llm_end"]
        tool_start = dict(events)["tool_start"]
        tool_end = dict(events)["tool_end"]
        assert start["node"] == "Market Analyst"
        assert start["model"]  # serialized fake-model name present
        assert "看多" in end["text"]
        assert end["tool_calls"] == []  # plain-text reply
        assert tool_start["tool"] == "echo_probe"
        assert tool_start["args"].endswith("RSI}") or "RSI" in tool_start["args"]
        assert tool_end["size"] > 0 and "echo:" in tool_end["result"]

    def test_callback_failures_never_break_execution(self):
        # emit raising must not leak into the graph run.
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        from server.agent_stream import AgentStreamHandler

        def boom(kind, payload):
            raise RuntimeError("noisy UI")

        h = AgentStreamHandler(boom)
        graph = self._graph(GenericFakeChatModel(messages=iter(["ok"])))
        final = {}
        for mode, payload in graph.stream(
            {"out": ""}, stream_mode=["updates", "values"],
            config={"callbacks": [h], "recursion_limit": 10},
        ):
            if mode == "values":
                final = payload
        assert final and "out" in final

    def test_trading_graph_merges_runtime_callbacks_into_invoke_config(self):
        import functools as _ft
        from unittest.mock import MagicMock

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        marker = object()
        captured = {}

        tg = MagicMock()
        tg.debug = False
        tg.config = {"results_dir": "/tmp/x"}
        tg.__dict__.update({"_runtime_callbacks": [marker], "_progress_hook": None})
        state = {"final_trade_decision": "", "company_of_interest": "X",
                 "trade_date": "2026-01-01"}
        tg.propagator.create_initial_state.return_value = state
        tg.propagator.get_graph_args.return_value = {"config": {"recursion_limit": 1}}
        tg.graph.invoke.return_value = state
        tg.memory_log.store_decision = lambda **k: captured.update(k)
        real_run = _ft.partial(TradingAgentsGraph._run_graph, tg)

        class Dummy:
            pass

        stand_in = type("StandIn", (), {})()
        stand_in.ticker = None
        stand_in.log_states_dict = {}
        stand_in.memory_log = MagicMock(memory_log_path="/tmp/x.md")
        stand_in.process_signal = MagicMock(return_value="Buy")
        real_run(stand_in, "X", "2026-01-01")
        kwargs = tg.graph.invoke.call_args.kwargs
        assert marker in (kwargs.get("config") or {}).get("callbacks", [])


class TestSpotQuoteCache:
    """Background spot cache: instant reads, stale-while-revalidate semantics."""

    def _cache(self, fetch, seconds=30):
        from server.app import SpotQuoteCache

        return SpotQuoteCache(fetch, codes_provider=lambda: [], refresh_seconds=seconds)

    def test_first_pass_marks_ready_and_stores_rows(self):
        calls = []
        cache = self._cache(
            lambda codes: (calls.append(codes), {"510300": {"name": "沪深300ETF", "price": 4.65, "pct": 0.4}})[1]
        )
        rows, ts, ready = cache.get()
        assert ready is False and rows == {}
        # simulate one manual pass instead of sleeping the daemon loop
        rows = cache._fetch([])
        cache._rows = rows
        cache._ts = 123.0
        cache._ready = True
        rows, ts, ready = cache.get()
        assert ready is True and ts == 123.0
        assert rows["510300"]["name"] == "沪深300ETF"

    def test_failure_keeps_previous_snapshot(self):
        state = {"n": 0}
        def flaky(codes):
            state["n"] += 1
            if state["n"] == 1:
                return {"159994": {"name": "芯片ETF", "price": 1.2, "pct": -0.5}}
            raise RuntimeError("throttled")
        cache = self._cache(flaky)
        cache._rows = cache._fetch([])
        cache._ready = True
        with contextlib.suppress(RuntimeError):
            cache._fetch([])
        rows, _ts, ready = cache.get()
        assert ready is True and rows == {"159994": {"name": "芯片ETF", "price": 1.2, "pct": -0.5}}

    def test_snapshot_is_isolated_from_later_mutation(self):
        cache = self._cache(lambda codes: {})
        cache._rows = {"a": 1}
        cache._ready = True
        snap, _ts, _ready = cache.get()
        snap["a"] = 999
        assert cache.get()[0]["a"] == 1


class TestSinaSpotParsing:
    def test_parse_real_payload_shape(self):
        import requests as _requests

        import server.app as app_mod

        payload = (
            'var hq_str_sh510300="沪深300ETF华泰柏瑞,4.684,4.691,4.688,4.689,4.679,'
            '4.688,4.689,22019066,103143760.000,135800,4.688,1507500,4.687,382500,'
            '4.686,1059800,4.685,709900,4.684,1066000,2026-08-27,15:00:00,00";\n'
            'var hq_str_sz159994="5GETF,1.280,1.287,1.293,1.297,1.278,1.293,1.295,'
            '9533100,12261857.200,3634000,1.293,364000,1.292,247400,1.291,514100,'
            '1.290,266500,1.289,107600,1.288,2026-08-27,15:00:00,00";\n'
        )

        class FakeResp:
            content = payload.encode("gb18030")

            def raise_for_status(self):
                return None

        with mock.patch.object(_requests, "get", return_value=FakeResp()):
            rows = app_mod._fetch_sina_spot(["510300", "159994"])

        assert rows["510300"]["name"] == "沪深300ETF华泰柏瑞"
        assert rows["510300"]["price"] == 4.688
        assert rows["510300"]["pct"] == pytest.approx((4.688 - 4.691) / 4.691 * 100, rel=1e-6)
        assert rows["159994"]["price"] == 1.293

    def test_sina_code_prefix_mapping(self):
        import server.app as app_mod

        assert app_mod._sina_code("510300") == "sh510300"
        assert app_mod._sina_code("159994") == "sz159994"
        assert app_mod._sina_code("600519") == "sh600519"
        assert app_mod._sina_code("000001") == "sz000001"

    def test_runner_preset_leads_with_sina(self):
        from server.runner import AnalysisRunner

        task = {"ticker": "600519.SS", "trade_date": "2026-08-27", "asset_type": "stock"}
        cfg = AnalysisRunner({})._build_config(task)
        assert cfg["data_vendors"]["core_stock_apis"] == "sina,yfinance"
        assert cfg["data_vendors"]["technical_indicators"] == "sina,yfinance"
        assert "akshare" not in cfg["data_vendors"]["fundamental_data"]
        assert cfg["data_vendors"]["news_data"] == "akshare,yfinance"


class TestRetrospectiveFixes:
    """复盘修复回归：取消竞态 + 事件保留 + health 路径。"""

    def test_cancel_race_is_atomic(self, db):
        # 先把任务置为 running，再取消 → 必须失败（不能覆盖 running）。
        q, _client = _make(db)
        (tid,) = client_submit(q, ["600519"])
        db.update_task(tid, status="running", started_at=1.0)
        assert q.cancel(tid) is False
        assert db.get_task(tid)["status"] == "running"

    def test_cancel_pending_still_works(self, db):
        q, _client = _make(db)
        (tid,) = client_submit(q, ["600519"])
        assert q.cancel(tid) is True
        assert db.get_task(tid)["status"] == "cancelled"

    def test_prune_events_keeps_recent_tasks_only(self, db):
        # 3 个任务各 1 条事件，保留最近 2 个 → 最旧的事件被清理。
        import time as _t
        ids = []
        for i in range(3):
            tid = db.create_task({"ticker": f"60051{i}", "trade_date": "2026-08-28"})
            db.update_task(tid, created_at=_t.time() + i)
            db.append_event(tid, "node", {"n": i})
            ids.append(tid)
        removed = db.prune_events(keep_tasks=2)
        assert removed == 1
        assert db.events_since(ids[0], 0) == []           # oldest pruned
        assert len(db.events_since(ids[1], 0)) == 1
        assert len(db.events_since(ids[2], 0)) == 1

    def test_health_reports_actual_db_path(self, db):
        q, client = _make(db)
        h = client.get("/api/health").json()
        assert h["db_path"] == db.path()  # 以真实连接为准，而非猜测的默认路径


class TestEventBus:
    def test_cross_thread_publish_reaches_async_subscriber(self):
        import asyncio
        import threading

        from server.events import EventBus

        bus = EventBus()
        received = []

        async def main():
            q = await bus.subscribe("t1")
            def produce():
                bus.publish("t1", {"id": 1, "type": "node"})
                bus.publish("t1", {"id": 2, "type": "status"})
            threading.Thread(target=produce).start()
            for _ in range(2):
                ev = await asyncio.wait_for(q.get(), timeout=2)
                received.append(ev)

        asyncio.run(main())
        assert [e["id"] for e in received] == [1, 2]

    def test_unsubscribe_stops_delivery(self):
        import asyncio

        from server.events import EventBus

        bus = EventBus()

        async def main():
            q = await bus.subscribe("t2")
            bus.unsubscribe("t2", q)
            bus.publish("t2", {"id": 1})
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q.get(), timeout=0.2)

        asyncio.run(main())

    def test_sse_replays_full_history_and_closes_on_terminal(self, db):
        """终态后订阅：流应重放全部事件并自动关闭（TestClient 无法测实时
        推送——实时投递由 TestEventBus 的跨线程用例覆盖）。"""
        FakeRunner.calls.clear()
        q = TaskQueue(db, workers=0, runner_cls=FakeRunner)
        client = TestClient(create_app(db=db, queue=q, start_spot=False))
        (tid,) = client_submit(q, ["600519"])
        q._execute(db.get_task(tid))          # 先完成，产生全部事件

        got = b""
        with client.stream("GET", f"/api/tasks/{tid}/events") as resp:
            for chunk in resp.iter_raw():
                got += chunk
        body = got.decode()
        assert '"type": "llm_start"' in body
        assert '"status": "completed"' in body
        # 终态后流必须自然关闭（不含心跳挂起）
        assert body.count("data:") == len(db.events_since(tid, 0))


@pytest.mark.unit
class TestParallelWorkers:
    def test_two_workers_run_concurrently(self, db):
        """The whole point of scoped configs: two runners in flight at once.

        Each fake run waits on a 2-party barrier; with a single worker the
        barrier would time out and the task would fail.
        """
        import threading

        barrier = threading.Barrier(2, timeout=10)

        class BarrierRunner(FakeRunner):
            calls = []
            fail_tickers = set()

            def __init__(self, settings):
                super().__init__(settings)

            def run(self, task, emit, db, is_abandoned=None):
                emit("node", {"node": "Trader", "stage": "trader"})
                barrier.wait()
                self.calls.append(("run", task["id"]))
                return {"rating": "Hold", "summary": "", "report_dir": ""}

        BarrierRunner.calls = []
        q = TaskQueue(db, workers=2, runner_cls=BarrierRunner)
        q.start()
        try:
            q.submit({"tickers": ["600519", "000001"], "trade_date": "2026-08-27"})
            deadline = time.time() + 15
            while len(BarrierRunner.calls) < 2 and time.time() < deadline:
                time.sleep(0.1)
            statuses = [t["status"] for t in db.list_tasks()]
            assert statuses == ["completed", "completed"]
        finally:
            q.stop()

    def test_worker_env_override(self, db, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_QUEUE_WORKERS", "3")
        q = TaskQueue(db)
        assert q.workers == 3
        monkeypatch.setenv("TRADINGAGENTS_QUEUE_WORKERS", "bogus")
        assert TaskQueue(db).workers == 2  # falls back to the default


@pytest.mark.unit
class TestSSELivePath:
    """The user-hit SSE regression: queue.emit used to publish bus events
    flattened ({'id','type',**payload}) while the SSE handler consumed
    them expecting the db.events_since row shape ({'payload': {...}}) —
    a live event winning the race against the DB re-poll killed the
    stream with KeyError('payload'). TestClient buffers streaming
    responses (the handler finishes before the client reads), so the
    live race is covered by a real uvicorn thread at the bottom of this
    class; unit tests pin both ends of the contract."""

    def test_emit_publishes_db_row_shape(self, db):
        """Producer contract: bus events mirror events_since rows."""
        captured = []

        class BusStub:
            def publish(self, topic, event):
                captured.append(event)

        q = TaskQueue(db, workers=1, runner_cls=FakeRunner, bus=BusStub())
        (tid,) = client_submit(q, ["600519"])
        q._execute(db.get_task(tid))
        assert captured, "emit never reached the bus"
        for ev in captured:
            assert set(ev) == {"id", "type", "payload"}, f"stray keys: {ev!r}"
        statuses = [e for e in captured if e["type"] == "status"]
        assert statuses[-1]["payload"]["status"] == "completed"

    def test_normalize_accepts_both_shapes(self):
        from server.app import _normalize_bus_event

        row = {"id": 3, "ts": 1.0, "type": "status", "payload": {"status": "completed"}}
        assert _normalize_bus_event(row)["payload"] == {"status": "completed"}

        legacy_flat = {"id": 3, "type": "status", "status": "completed"}
        normalized = _normalize_bus_event(legacy_flat)
        assert normalized["payload"] == {"status": "completed"}

    def test_live_bus_event_over_real_http(self, db):
        """End-to-end over real HTTP: subscriber attaches mid-run, a live
        bus event (legacy flattened shape, direct publish — exactly what
        the old emit did) must be delivered, not KeyError the stream."""
        import httpx
        import threading
        import uvicorn

        release = threading.Event()
        started = threading.Event()

        class ParkedRunner(FakeRunner):
            def run(self, task, emit, db, is_abandoned=None):
                emit("status", {"status": "running", "ticker": task["ticker"]})
                started.set()
                release.wait(timeout=30)
                return {"rating": "Buy", "summary": "", "report_dir": ""}

        q = TaskQueue(db, workers=1, runner_cls=ParkedRunner)
        app = create_app(db=db, queue=q, start_spot=False)
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]

        (tid,) = client_submit(q, ["600519"])
        assert started.wait(timeout=10)

        body = b""
        published = got_live = False
        with httpx.stream("GET", f"http://127.0.0.1:{port}/api/tasks/{tid}/events",
                          timeout=30) as resp:
            for chunk in resp.iter_raw():
                body += chunk
                if not published and b'"status": "running"' in body:
                    published = True
                    # legacy flattened shape, exactly what emit() did
                    app.state.bus.publish(tid, {
                        "id": 10 ** 6, "type": "llm_end",
                        "node": "Trader", "text": "结论：看多",
                    })
                elif published and b"llm_end" in body:
                    got_live = True
                    break
        release.set()
        thread.join(timeout=5)
        assert got_live, f"live bus event never delivered; got: {body[:400]!r}"
        assert '"text": "结论：看多"' in body.decode()
