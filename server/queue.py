"""Background task queue driving TradingAgents analyses.

A fixed pool of worker threads (default 2, env-overridable via
``TRADINGAGENTS_QUEUE_WORKERS``) claims FIFO tasks and runs them through
:class:`server.runner.AnalysisRunner`, streaming progress into the events
table (SSE consumers poll it) and recording outcomes.

Parallelism is safe because every runner builds its config and executes
inside ``config_scope`` — ``set_config`` inside a scope only touches the
scoped view, so concurrent runs never see each other's vendors. The truly
process-global resources (akshare/py-mini_racer calls, the memory log,
cache-file writes) carry their own locks; the LLM API is the real
bottleneck, so more than ~4 workers rarely helps.

A watchdog thread fails any running task that stops emitting events for
``max_idle`` seconds — a worker wedged in a call that never returns cannot
be interrupted, but its task row and SSE stream must not hang forever.
Streaming LLM calls emit throttled ``llm_progress`` heartbeats, so a healthy
slow generation is not mistaken for a wedge.

The verdict is also enforced worker-side: the runner polls an abandonment
predicate before every LLM call and raises ``TaskAbandoned`` once the
watchdog has failed the task, so the thread unwinds instead of grinding
through the remaining graph (burning provider tokens) on a failed task.
"""

import logging
import os
import threading

from .db import Database
from .runner import ALL_ANALYST_KEYS, TaskAbandoned, build_stages, format_exception

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 2

# Watchdog: a running task that emits no events for this long is wedged in a
# call that will never return (no timeout anywhere in a vendor/SDK chain).
# Streaming LLM calls emit llm_progress heartbeats, so the quiet window is
# bounded by non-streaming LLM calls (llm_timeout x attempts) and slow tool
# chains, not by the whole generation; 40 min keeps generous slack over both.
# Raise TRADINGAGENTS_TASK_MAX_IDLE alongside llm_timeout.
DEFAULT_MAX_IDLE = 2400
WATCHDOG_INTERVAL = 30.0


class TaskQueue:
    def __init__(
        self,
        db: Database,
        settings: dict[str, str] | None = None,
        workers: int | None = None,
        runner_cls=None,
        bus=None,
        max_idle: float | None = None,
    ):
        from .runner import AnalysisRunner

        self.db = db
        self.settings = settings or {}
        if workers is None:
            try:
                workers = int(os.environ.get("TRADINGAGENTS_QUEUE_WORKERS", DEFAULT_WORKERS))
            except ValueError:
                workers = DEFAULT_WORKERS
            workers = max(1, int(workers))
        self.workers = max(0, int(workers))
        if max_idle is None:
            try:
                max_idle = float(os.environ.get("TRADINGAGENTS_TASK_MAX_IDLE", DEFAULT_MAX_IDLE))
            except ValueError:
                max_idle = DEFAULT_MAX_IDLE
        self.max_idle = float(max_idle)
        self.runner_cls = runner_cls or AnalysisRunner
        self.bus = bus
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        # Tasks the watchdog declared dead while their worker thread was still
        # blocked; the worker must go quiet and never resurrect them.
        self._abandoned: set[str] = set()
        self._abandoned_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        for i in range(self.workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"ta-worker-{i}", daemon=True
            )
            t.start()
            self._threads.append(t)
        if self.workers > 0 and self.max_idle > 0:
            t = threading.Thread(target=self._watchdog_loop, name="ta-watchdog", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()

    # -- submission ------------------------------------------------------------
    def submit(self, spec: dict) -> list[str]:
        """Enqueue one task per ticker; returns created task ids."""
        tickers = [t.strip() for t in spec.get("tickers", []) if t.strip()]
        if not tickers:
            raise ValueError("tickers must not be empty")
        analysts = [a for a in spec.get("analysts", ALL_ANALYST_KEYS) if a in ALL_ANALYST_KEYS]
        ids: list[str] = []
        for ticker in tickers:
            task_id = self.db.create_task(
                {
                    **spec,
                    "ticker": ticker,
                    "analysts": analysts or list(ALL_ANALYST_KEYS),
                }
            )
            ids.append(task_id)
        return ids

    def cancel(self, task_id: str) -> bool:
        """Cancel while still pending. Running tasks cannot be interrupted."""
        row = self.db.get_task(task_id)
        if not row:
            raise KeyError(task_id)
        cancelled = self.db.cancel_if_pending(task_id)
        if cancelled:
            self.db.append_event(task_id, "status", {"status": "cancelled"})
        return cancelled

    # -- events ---------------------------------------------------------------
    def _publish(self, task_id: str, type_: str, payload: dict):
        """Append an event row and fan it out to SSE bus subscribers."""
        try:
            event_id = self.db.append_event(task_id, type_, payload)
        except Exception:
            logger.exception("append_event failed")
            return
        if self.bus is not None:
            try:
                # Same shape as db.events_since rows (payload nested, not
                # flattened) — the SSE handler consumes both paths.
                self.bus.publish(task_id, {
                    "id": event_id, "type": type_, "payload": payload,
                })
            except Exception:
                logger.exception("event bus publish failed")

    # -- watchdog ---------------------------------------------------------------
    def _watchdog_loop(self):
        while not self._stop.wait(WATCHDOG_INTERVAL):
            try:
                self._watchdog_sweep()
            except Exception:
                logger.exception("task watchdog sweep failed")

    def _watchdog_sweep(self, now: float | None = None) -> list[str]:
        """Fail running tasks that stopped emitting events; returns their ids.

        A wedged worker cannot be interrupted (threads have no kill), so the
        row is failed out from under it: SSE consumers get a terminal status
        instead of waiting on a ghost. The worker's own state writes are
        WHERE-guarded so it cannot resurrect the task, and its events stay
        suppressed in case it ever unblocks and drains.
        """
        failed: list[str] = []
        for task in self.db.stalled_running_tasks(self.max_idle, now=now):
            minutes = self.max_idle / 60
            error = f"任务看门狗超时：超过 {minutes:.0f} 分钟无任何进展，已标记失败"
            if not self.db.fail_task_unless_terminal(task["id"], error):
                continue
            with self._abandoned_lock:
                self._abandoned.add(task["id"])
            failed.append(task["id"])
            logger.warning(
                "task %s (%s) idle for %.0f min; watchdog marked it failed",
                task["id"], task.get("ticker", "?"), minutes,
            )
            self._publish(task["id"], "status", {"status": "failed", "error": error})
        return failed

    # -- worker ------------------------------------------------------------------
    def _is_abandoned(self, task_id: str) -> bool:
        with self._abandoned_lock:
            return task_id in self._abandoned

    def _worker_loop(self):
        while not self._stop.is_set():
            task = None
            try:
                task = self.db.claim_next_pending()
            except Exception:
                logger.exception("claiming next pending task failed")
                self._stop.wait(1.0)
                continue
            if task is None:
                self._stop.wait(0.5)
                continue
            try:
                self._execute(task)
            except Exception as exc:
                logger.exception("task %s crashed unexpectedly", task["id"])
                if self.db.fail_task_unless_terminal(
                    task["id"], format_exception(exc)
                ):
                    self.db.append_event(
                        task["id"], "status", {"status": "failed", "error": format_exception(exc)}
                    )

    def _execute(self, task: dict):
        db = self.db
        task_id = task["id"]
        stages = build_stages(task["analysts"])
        # Settings are re-read per run so UI changes apply to the next task.
        runner = self.runner_cls({**self.settings, **db.get_settings()})
        runner.reset(task_id, db, stages)

        def emit(type_: str, payload: dict):
            with self._abandoned_lock:
                if task_id in self._abandoned:
                    return
            self._publish(task_id, type_, payload)

        emit("status", {"status": "running", "ticker": task["ticker"]})
        result: dict | None = None
        try:
            result = runner.run(
                task, emit=emit, db=db, is_abandoned=lambda: self._is_abandoned(task_id)
            )
        except TaskAbandoned:
            logger.warning("task %s abandoned; worker stopped executing it", task_id)
            return
        except Exception as exc:
            message = format_exception(exc)
            logger.error("task %s failed: %s", task_id, message)
            # Guarded: the watchdog may have failed this task already; its
            # verdict (and event) must win over the late crash report.
            if db.fail_task_unless_terminal(task_id, message):
                emit("status", {"status": "failed", "error": message})
            return
        finally:
            runner.finish_stages(task_id, db)

        if not db.complete_task_unless_terminal(
            task_id,
            rating=result.get("rating", ""),
            summary=result.get("summary", ""),
            report_dir=result.get("report_dir", ""),
        ):
            # The watchdog failed this task while the worker was still
            # finishing; keep that verdict, drop the stale outcome.
            logger.warning(
                "task %s finished after being marked failed; outcome dropped", task_id
            )
            return
        db.prune_events()
        emit(
            "status",
            {
                "status": "completed",
                "rating": result.get("rating", ""),
                "report_dir": result.get("report_dir", ""),
            },
        )
