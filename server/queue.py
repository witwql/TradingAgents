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
"""

import logging
import os
import threading
import time

from .db import Database
from .runner import ALL_ANALYST_KEYS, build_stages, format_exception

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 2


class TaskQueue:
    def __init__(
        self,
        db: Database,
        settings: dict[str, str] | None = None,
        workers: int | None = None,
        runner_cls=None,
        bus=None,
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
        self.runner_cls = runner_cls or AnalysisRunner
        self.bus = bus
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        for i in range(self.workers):
            t = threading.Thread(
                target=self._worker_loop, name=f"ta-worker-{i}", daemon=True
            )
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

    # -- worker ------------------------------------------------------------------
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
                self.db.update_task(
                    task["id"],
                    status="failed",
                    error=format_exception(exc),
                    finished_at=time.time(),
                )
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
            try:
                event_id = db.append_event(task_id, type_, payload)
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

        emit("status", {"status": "running", "ticker": task["ticker"]})
        result: dict | None = None
        try:
            result = runner.run(task, emit=emit, db=db)
        except Exception as exc:
            message = format_exception(exc)
            logger.error("task %s failed: %s", task_id, message)
            db.update_task(
                task_id,
                status="failed",
                error=message,
                finished_at=time.time(),
            )
            emit("status", {"status": "failed", "error": message})
            return
        finally:
            runner.finish_stages(task_id, db)

        db.update_task(
            task_id,
            status="completed",
            rating=result.get("rating", ""),
            summary=result.get("summary", ""),
            report_dir=result.get("report_dir", ""),
            current_stage="done",
            finished_at=time.time(),
        )
        db.prune_events()
        emit(
            "status",
            {
                "status": "completed",
                "rating": result.get("rating", ""),
                "report_dir": result.get("report_dir", ""),
            },
        )
