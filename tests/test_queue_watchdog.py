"""Queue watchdog: wedged workers get failed instead of haunting the UI.

A worker blocked in a call that never returns (no timeout anywhere in a
vendor/SDK chain) cannot be interrupted. These tests verify the watchdog
fails such tasks from the outside, that the WHERE-guarded state writes
keep a late-finishing worker from resurrecting its task, that the worker
actually unwinds at the next LLM boundary (AbandonmentGuard), and that
streaming LLM calls keep the idle clock fed via heartbeat events.
"""
import threading
import time

import pytest

from server.agent_stream import AbandonmentGuard, AgentStreamHandler
from server.db import Database
from server.queue import TaskQueue
from server.runner import TaskAbandoned


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "dash.db")


def _running_task(db, *, started_age: float = 0.0, last_event_age: float | None = None):
    """Claim one task, then backdate its start / last event for the test."""
    task_id = db.create_task({"ticker": "515880.SS", "trade_date": "2026-09-01"})
    claimed = db.claim_next_pending()
    assert claimed and claimed["id"] == task_id
    now = time.time()
    if started_age:
        db.update_task(task_id, started_at=now - started_age)
    if last_event_age is not None:
        db.append_event(task_id, "llm_start", {"node": "Market Analyst"})
        with db._lock:
            db._conn.execute(
                "UPDATE task_events SET ts=? WHERE task_id=?", (now - last_event_age, task_id)
            )
            db._conn.commit()
    return task_id


class _LateRunner:
    """Mimics a worker that unblocks after the watchdog gave up on it."""

    calls: list = []

    def __init__(self, settings):
        self.settings = settings

    def reset(self, task_id, db, stages):
        db.set_stages(task_id, [name for name, _ in stages])

    def run(self, task, emit, db, is_abandoned=None):
        self.calls.append(task["id"])
        emit("llm_start", {"node": "Market Analyst"})
        return {"rating": "BUY", "summary": "late outcome", "report_dir": ""}

    def finish_stages(self, task_id, db):
        pass


@pytest.mark.unit
class TestWatchdogSweep:
    def test_recent_activity_is_not_flagged(self, db):
        task_id = _running_task(db, last_event_age=60)
        q = TaskQueue(db, workers=0, max_idle=2400)
        assert q._watchdog_sweep() == []
        assert db.get_task(task_id)["status"] == "running"

    def test_stalled_task_is_failed_and_event_emitted(self, db):
        task_id = _running_task(db, last_event_age=3000)
        q = TaskQueue(db, workers=0, max_idle=2400)
        assert q._watchdog_sweep() == [task_id]
        row = db.get_task(task_id)
        assert row["status"] == "failed"
        assert "看门狗" in row["error"]
        assert row["finished_at"] is not None
        events = db.events_since(task_id, 0)
        assert events[-1]["type"] == "status"
        assert events[-1]["payload"]["status"] == "failed"
        # a second sweep is a no-op: the task is no longer running
        assert q._watchdog_sweep() == []

    def test_task_without_events_falls_back_to_started_at(self, db):
        task_id = _running_task(db, started_age=3000)
        q = TaskQueue(db, workers=0, max_idle=2400)
        assert q._watchdog_sweep() == [task_id]

    def test_recent_event_overrides_old_start(self, db):
        task_id = _running_task(db, started_age=3000, last_event_age=60)
        q = TaskQueue(db, workers=0, max_idle=2400)
        assert q._watchdog_sweep() == []

    def test_env_override_sets_max_idle(self, db, monkeypatch):
        monkeypatch.setenv("TRADINGAGENTS_TASK_MAX_IDLE", "600")
        assert TaskQueue(db, workers=0).max_idle == 600.0


@pytest.mark.unit
class TestZombieWorkerGuards:
    def test_late_worker_stays_silent_and_cannot_complete(self, db):
        _LateRunner.calls.clear()
        task_id = _running_task(db, last_event_age=3000)
        q = TaskQueue(db, workers=0, max_idle=2400, runner_cls=_LateRunner)
        assert q._watchdog_sweep() == [task_id]
        # the wedged-then-unblocked worker drains, but goes quiet and its
        # outcome is dropped in favor of the watchdog verdict
        q._execute(db.get_task(task_id))
        row = db.get_task(task_id)
        assert row["status"] == "failed"
        assert "看门狗" in row["error"]
        events = db.events_since(task_id, 0)
        assert [e["type"] for e in events] == ["llm_start", "status"]
        assert events[-1]["payload"]["status"] == "failed"

    def test_zombie_worker_cannot_resurrect_failed_task(self, db):
        task_id = _running_task(db, last_event_age=3000)
        assert db.fail_task_unless_terminal(task_id, "watchdog")
        assert not db.complete_task_unless_terminal(
            task_id, rating="BUY", summary="late", report_dir=""
        )
        assert not db.fail_task_unless_terminal(task_id, "late crash")
        row = db.get_task(task_id)
        assert row["status"] == "failed"
        assert row["error"] == "watchdog"

    def test_pending_task_still_completes(self, db):
        # _execute may be driven directly on a pending row (the SSE replay
        # test does exactly this); the terminal guard must not drop that
        # outcome and leave the event stream waiting forever.
        task_id = db.create_task({"ticker": "515880.SS", "trade_date": "2026-09-01"})
        assert db.complete_task_unless_terminal(task_id, rating="BUY", summary="s", report_dir="")
        assert db.get_task(task_id)["status"] == "completed"

    def test_cancelled_task_is_frozen(self, db):
        task_id = db.create_task({"ticker": "515880.SS", "trade_date": "2026-09-01"})
        assert db.cancel_if_pending(task_id)
        assert not db.complete_task_unless_terminal(task_id, rating="BUY", summary="s", report_dir="")
        assert not db.fail_task_unless_terminal(task_id, "late")
        assert db.get_task(task_id)["status"] == "cancelled"

    def test_completion_blocked_once_terminal(self, db):
        task_id = _running_task(db)
        assert db.complete_task_unless_terminal(task_id, rating="BUY", summary="s", report_dir="")
        assert db.get_task(task_id)["status"] == "completed"
        assert not db.complete_task_unless_terminal(task_id, rating="SELL", summary="x", report_dir="")
        assert db.get_task(task_id)["rating"] == "BUY"


@pytest.mark.unit
class TestAbandonmentGuard:
    def test_raises_when_abandoned(self):
        guard = AbandonmentGuard(lambda: True)
        with pytest.raises(TaskAbandoned):
            guard.on_chat_model_start(None, [], run_id="r1")

    def test_silent_while_task_live(self):
        guard = AbandonmentGuard(lambda: False)
        guard.on_chat_model_start(None, [], run_id="r1")  # no raise

    def test_raise_propagates_through_model_invoke(self):
        # The whole point of raise_error=True: the guard's exception must
        # escape the chat-model invocation, not be logged-and-swallowed.
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        model = GenericFakeChatModel(messages=iter(["never reached"]))
        with pytest.raises(TaskAbandoned):
            model.invoke([("human", "hi")], config={"callbacks": [AbandonmentGuard(lambda: True)]})

    def test_live_task_invoke_untouched(self):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        model = GenericFakeChatModel(messages=iter(["ok"]))
        response = model.invoke(
            [("human", "hi")], config={"callbacks": [AbandonmentGuard(lambda: False)]}
        )
        assert response.content == "ok"


@pytest.mark.unit
class TestAbandonedWorkerUnwinds:
    def _backdate_events(self, db, task_id, age):
        with db._lock:
            db._conn.execute(
                "UPDATE task_events SET ts=? WHERE task_id=?",
                (time.time() - age, task_id),
            )
            db._conn.commit()

    def test_runner_receives_abandonment_predicate(self, db):
        release = threading.Event()
        captured = {}

        class ParkedRunner(_LateRunner):
            def run(self, task, emit, db, is_abandoned=None):
                emit("llm_start", {"node": "Market Analyst"})
                captured["before"] = bool(is_abandoned and is_abandoned())
                release.wait(timeout=10)
                captured["after"] = bool(is_abandoned and is_abandoned())
                return {"rating": "", "summary": "", "report_dir": ""}

        task_id = _running_task(db)
        q = TaskQueue(db, workers=0, max_idle=2400, runner_cls=ParkedRunner)
        t = threading.Thread(target=q._execute, args=(db.get_task(task_id),), daemon=True)
        t.start()
        deadline = time.time() + 5
        while len(db.events_since(task_id, 0)) < 2 and time.time() < deadline:
            time.sleep(0.02)
        self._backdate_events(db, task_id, 3000)
        assert q._watchdog_sweep() == [task_id]
        release.set()
        t.join(timeout=5)
        assert captured == {"before": False, "after": True}

    def test_worker_unwinds_after_watchdog_failure(self, db):
        release = threading.Event()
        state = {"aborted": False, "completed": False}

        class ParkedRunner(_LateRunner):
            def run(self, task, emit, db, is_abandoned=None):
                emit("llm_start", {"node": "Market Analyst"})
                release.wait(timeout=10)
                if is_abandoned is not None and is_abandoned():
                    state["aborted"] = True
                    raise TaskAbandoned("worker aborted")
                state["completed"] = True
                return {"rating": "BUY", "summary": "late", "report_dir": ""}

        task_id = _running_task(db)
        q = TaskQueue(db, workers=0, max_idle=2400, runner_cls=ParkedRunner)
        t = threading.Thread(target=q._execute, args=(db.get_task(task_id),), daemon=True)
        t.start()
        deadline = time.time() + 5
        while len(db.events_since(task_id, 0)) < 2 and time.time() < deadline:
            time.sleep(0.02)  # status running + llm_start emitted
        self._backdate_events(db, task_id, 3000)
        assert q._watchdog_sweep() == [task_id]
        release.set()
        t.join(timeout=5)
        assert not t.is_alive()
        assert state["aborted"] and not state["completed"]
        row = db.get_task(task_id)
        assert row["status"] == "failed"
        assert "看门狗" in row["error"]
        types = [e["type"] for e in db.events_since(task_id, 0)]
        assert types == ["status", "llm_start", "status"]
        assert db.events_since(task_id, 0)[-1]["payload"]["status"] == "failed"


@pytest.mark.unit
class TestStreamHeartbeat:
    def _handler(self, events):
        return AgentStreamHandler(lambda kind, payload: events.append((kind, payload)))

    def test_heartbeat_emitted_throttled_and_attributed(self, monkeypatch):
        events = []
        h = self._handler(events)
        clock = {"t": 1000.0}
        monkeypatch.setattr("server.agent_stream.time.monotonic", lambda: clock["t"])
        h._remember("run-1", {"kind": "llm", "node": "News Analyst"})

        h.on_llm_new_token("a", run_id="run-1")
        assert events == [("llm_progress", {"node": "News Analyst"})]

        h.on_llm_new_token("b", run_id="run-1")  # inside the quiet window
        assert len(events) == 1

        clock["t"] += 61.0
        h.on_llm_new_token("c", run_id="run-1")
        assert len(events) == 2

    def test_heartbeat_without_attribution_has_no_node(self, monkeypatch):
        events = []
        h = self._handler(events)
        monkeypatch.setattr("server.agent_stream.time.monotonic", lambda: 5.0)
        h.on_llm_new_token("x", run_id="unknown-run")
        assert events == [("llm_progress", {"node": None})]
