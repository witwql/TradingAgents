"""SQLite persistence for the dashboard: tasks, workflow stages, events,
favorites and settings. Single writer-thread friendly (WAL mode) with a coarse
lock so FastAPI request handlers and queue workers share it safely.
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

_DEFAULT_HOME = Path(os.environ.get("TRADINGAGENTS_DASHBOARD_HOME", "")) if os.environ.get(
    "TRADINGAGENTS_DASHBOARD_HOME"
) else Path.home() / ".tradingagents" / "dashboard"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    ticker TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    asset_type TEXT NOT NULL DEFAULT 'stock',
    analysts TEXT NOT NULL,
    debate_rounds INTEGER NOT NULL DEFAULT 1,
    risk_rounds INTEGER NOT NULL DEFAULT 1,
    output_language TEXT NOT NULL DEFAULT 'Chinese',
    status TEXT NOT NULL DEFAULT 'pending',
    current_stage TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    rating TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    report_dir TEXT NOT NULL DEFAULT '',
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE TABLE IF NOT EXISTS task_stages (
    task_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at REAL,
    finished_at REAL,
    PRIMARY KEY (task_id, name)
);
CREATE TABLE IF NOT EXISTS task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_task ON task_events(task_id);
CREATE TABLE IF NOT EXISTS favorites (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    added_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS value_runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL DEFAULT 'running',
    trade_date TEXT NOT NULL DEFAULT '',
    universe INTEGER NOT NULL DEFAULT 0,
    stage TEXT NOT NULL DEFAULT '',
    processed INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    qualifying INTEGER NOT NULL DEFAULT 0,
    results TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    cancel_requested INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS screen_runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL DEFAULT 'running',
    trade_date TEXT NOT NULL DEFAULT '',
    universe INTEGER NOT NULL DEFAULT 0,
    results TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS pick_returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,             -- 'screen' (动量) | 'value' (价值)
    run_id TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    code TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    pick_price REAL,                    -- selection-time spot price
    baseline_price REAL,                -- OHLCV close on trade_date
    score REAL,                         -- momentum probability / value score
    rank INTEGER,
    ret_1d REAL,
    ret_5d REAL,
    settled_at REAL,                    -- ts of last settlement attempt
    UNIQUE(run_type, run_id, code)
);
CREATE INDEX IF NOT EXISTS idx_pick_returns_run ON pick_returns(run_type, run_id);
"""


class Database:
    """Thin thread-safe wrapper over a single SQLite connection."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            _DEFAULT_HOME.mkdir(parents=True, exist_ok=True)
            path = _DEFAULT_HOME / "dashboard.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self):
        """Additive column migrations for tables created by older builds."""
        for table, column, decl in (
            ("screen_runs", "stage", "TEXT NOT NULL DEFAULT ''"),
            ("screen_runs", "processed", "INTEGER NOT NULL DEFAULT 0"),
            ("screen_runs", "total", "INTEGER NOT NULL DEFAULT 0"),
            ("screen_runs", "qualifying", "INTEGER NOT NULL DEFAULT 0"),
            ("screen_runs", "cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
            ("value_runs", "cancel_requested", "INTEGER NOT NULL DEFAULT 0"),
            ("value_runs", "stage", "TEXT NOT NULL DEFAULT ''"),
            ("value_runs", "processed", "INTEGER NOT NULL DEFAULT 0"),
            ("value_runs", "total", "INTEGER NOT NULL DEFAULT 0"),
            ("value_runs", "qualifying", "INTEGER NOT NULL DEFAULT 0"),
        ):
            cols = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # -- generic helpers ---------------------------------------------------
    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    # -- tasks ---------------------------------------------------------------
    def create_task(self, spec: dict) -> str:
        task_id = uuid.uuid4().hex[:12]
        now = time.time()
        self.execute(
            "INSERT INTO tasks (id, created_at, ticker, trade_date, asset_type, analysts,"
            " debate_rounds, risk_rounds, output_language)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                task_id,
                now,
                spec["ticker"],
                spec["trade_date"],
                spec.get("asset_type", "stock"),
                json.dumps(spec.get("analysts", ["market", "social", "news", "fundamentals", "macro"])),
                int(spec.get("debate_rounds", 1)),
                int(spec.get("risk_rounds", 1)),
                spec.get("output_language", "Chinese"),
            ),
        )
        return task_id

    def get_task(self, task_id: str) -> dict | None:
        row = self.fetchone("SELECT * FROM tasks WHERE id=?", (task_id,))
        if row:
            row["analysts"] = json.loads(row["analysts"])
        return row

    def list_tasks(self, limit: int = 100, status: str | None = None) -> list[dict]:
        if status:
            rows = self.fetchall(
                "SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            rows = self.fetchall(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        for row in rows:
            row["analysts"] = json.loads(row["analysts"])
        return rows

    def update_task(self, task_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE tasks SET {cols} WHERE id=?", (*fields.values(), task_id))

    def cancel_if_pending(self, task_id: str) -> bool:
        """Atomically cancel ONLY while still pending.

        The claim-then-check pattern in TaskQueue.cancel raced with a worker
        claiming between the check and the write, letting a cancelled status
        overwrite a running/completed task. The WHERE guard makes the state
        transition atomic.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE tasks SET status='cancelled', finished_at=?"
                " WHERE id=? AND status='pending'",
                (time.time(), task_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def claim_next_pending(self) -> dict | None:
        """Atomically claim the oldest pending task (FIFO)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM tasks WHERE status='pending' ORDER BY created_at LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                return None
            task = dict(row)
            task["analysts"] = json.loads(task["analysts"])
            self._conn.execute(
                "UPDATE tasks SET status='running', started_at=? WHERE id=? AND status='pending'",
                (time.time(), task["id"]),
            )
            self._conn.commit()
        refreshed = self.get_task(task["id"])
        return refreshed if refreshed and refreshed["status"] == "running" else None

    def stalled_running_tasks(self, max_idle: float, now: float | None = None) -> list[dict]:
        """Running tasks whose last activity is older than ``max_idle`` seconds.

        Last activity is the newest task_events row, falling back to
        ``started_at`` for tasks that have not emitted anything yet. A worker
        wedged in a call that never returns stops emitting events, so this is
        the signal the queue watchdog acts on. ``now`` is injectable for tests.
        """
        now = time.time() if now is None else now
        return self.fetchall(
            "SELECT t.* FROM tasks t WHERE t.status='running' AND"
            " COALESCE((SELECT MAX(e.ts) FROM task_events e WHERE e.task_id=t.id),"
            "          t.started_at) < ?"
            " ORDER BY t.started_at",
            (now - max_idle,),
        )

    _TERMINAL_TASK_STATES = ("completed", "failed", "cancelled")

    def fail_task_unless_terminal(self, task_id: str, error: str) -> bool:
        """Atomically fail a task unless it already reached a terminal state.

        Same WHERE-guard pattern as cancel_if_pending: the watchdog may fail a
        task whose worker is still alive in a blocked call, and a later state
        write from that worker must not resurrect it. Non-terminal (pending/
        running) tasks are writable — a worker crashing before its row was
        claimed still needs its failure recorded.
        """
        placeholders = ",".join("?" * len(self._TERMINAL_TASK_STATES))
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE tasks SET status='failed', error=?, finished_at=?"
                f" WHERE id=? AND status NOT IN ({placeholders})",
                (error, time.time(), task_id, *self._TERMINAL_TASK_STATES),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def complete_task_unless_terminal(self, task_id: str, *, rating: str, summary: str,
                                      report_dir: str) -> bool:
        """Atomically complete a task unless it already reached a terminal state.

        Guards the success path the same way fail_task_unless_terminal guards
        the failure path: a worker that finishes long after the watchdog
        declared the task dead must not overwrite the verdict with 'completed'.
        """
        placeholders = ",".join("?" * len(self._TERMINAL_TASK_STATES))
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE tasks SET status='completed', rating=?, summary=?, report_dir=?,"
                f" current_stage='done', finished_at=?"
                f" WHERE id=? AND status NOT IN ({placeholders})",
                (rating, summary, report_dir, time.time(), task_id,
                 *self._TERMINAL_TASK_STATES),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def request_screen_cancel(self, run_id: str) -> bool:
        """Cooperative cancellation: flag the run; the worker checks per item."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE screen_runs SET cancel_requested=1 WHERE id=? AND status='running'",
                (run_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def sweep_interrupted(self, max_running_age: float = 1800) -> dict[str, int]:
        """Fail stale 'running' rows left by a server restart/crash.

        A running screening older than ``max_running_age`` and any running
        analysis task cannot still be alive after a fresh boot; marking them
        here prevents UIs from waiting on ghosts forever.
        """
        now = time.time()
        counts = {"screen_runs": 0, "tasks": 0, "value_runs": 0}
        with self._lock:
            cur = self._conn.execute(
                "UPDATE screen_runs SET status='failed', error='服务重启，运行中断',"
                " finished_at=? WHERE status='running' AND created_at < ?",
                (now, now - max_running_age),
            )
            counts["screen_runs"] = cur.rowcount
            cur = self._conn.execute(
                "UPDATE value_runs SET status='failed', error='服务重启，运行中断',"
                " finished_at=? WHERE status='running' AND created_at < ?",
                (now, now - max_running_age),
            )
            counts["value_runs"] = cur.rowcount
            cur = self._conn.execute(
                "UPDATE tasks SET status='failed', error='服务重启，运行中断',"
                " finished_at=? WHERE status='running' AND started_at < ?",
                (now, now - max_running_age),
            )
            counts["tasks"] = cur.rowcount
            self._conn.commit()
        return counts

    def request_value_cancel(self, run_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE value_runs SET cancel_requested=1 WHERE id=? AND status='running'",
                (run_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def begin_screen_run(self, run_id: str, trade_date: str, reuse_window: float = 1800) -> tuple[str, bool]:
        """Atomically create a screening run or reuse an in-flight one.

        The existence check and INSERT happen under the connection lock, so
        two simultaneous POSTs cannot both spawn workers.
        Returns (run_id, already_running).
        """
        with self._lock:
            running = self._conn.execute(
                "SELECT id FROM screen_runs WHERE status='running' AND created_at > ?"
                " ORDER BY created_at DESC LIMIT 1",
                (time.time() - reuse_window,),
            ).fetchone()
            if running:
                return running["id"], True
            self._conn.execute(
                "INSERT INTO screen_runs (id, created_at, status, trade_date, stage)"
                " VALUES (?,?,?,?,?)",
                (run_id, time.time(), "running", trade_date, "universe"),
            )
            self._conn.commit()
            return run_id, False

    def prune_events(self, keep_tasks: int = 50) -> int:
        """Delete task_events of tasks beyond the most recent ``keep_tasks``.

        Every analysis emits 100-200 stream events; without retention the
        table grows unboundedly across months of use.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM task_events WHERE task_id NOT IN ("
                " SELECT id FROM tasks ORDER BY created_at DESC LIMIT ?)",
                (keep_tasks,),
            )
            self._conn.commit()
            return cur.rowcount

    def delete_task(self, task_id: str) -> None:
        self.execute("DELETE FROM task_stages WHERE task_id=?", (task_id,))
        self.execute("DELETE FROM task_events WHERE task_id=?", (task_id,))
        self.execute("DELETE FROM tasks WHERE id=?", (task_id,))

    # -- stages / events -----------------------------------------------------
    def set_stages(self, task_id: str, names: list[str]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM task_stages WHERE task_id=?", (task_id,))
            for seq, name in enumerate(names):
                self._conn.execute(
                    "INSERT INTO task_stages (task_id, seq, name) VALUES (?,?,?)",
                    (task_id, seq, name),
                )
            self._conn.commit()

    def update_stage(self, task_id: str, name: str, status: str) -> None:
        ts = time.time()
        col = "started_at" if status == "running" else "finished_at"
        self.execute(
            f"UPDATE task_stages SET status=?, {col}=? WHERE task_id=? AND name=?",
            (status, ts, task_id, name),
        )

    def get_stages(self, task_id: str) -> list[dict]:
        return self.fetchall(
            "SELECT name, seq, status, started_at, finished_at FROM task_stages"
            " WHERE task_id=? ORDER BY seq",
            (task_id,),
        )

    def append_event(self, task_id: str, type_: str, payload: dict) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO task_events (task_id, ts, type, payload) VALUES (?,?,?,?)",
                (task_id, time.time(), type_, json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def events_since(self, task_id: str, last_id: int) -> list[dict]:
        rows = self.fetchall(
            "SELECT id, ts, type, payload FROM task_events WHERE task_id=? AND id>?"
            " ORDER BY id",
            (task_id, last_id),
        )
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    # -- favorites -----------------------------------------------------------
    def list_favorites(self) -> list[dict]:
        return self.fetchall("SELECT * FROM favorites ORDER BY added_at")

    def add_favorite(self, code: str, name: str = "") -> None:
        self.execute(
            "INSERT OR REPLACE INTO favorites (code, name, added_at) VALUES (?,?,?)",
            (code, name, time.time()),
        )

    def remove_favorite(self, code: str) -> None:
        self.execute("DELETE FROM favorites WHERE code=?", (code,))

    # -- settings --------------------------------------------------------------
    def path(self) -> str:
        """Actual SQLite file backing this connection (diagnostics)."""
        row = self._conn.execute("PRAGMA database_list").fetchone()
        return row[2] if row else ""

    def get_settings(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.fetchall("SELECT key, value FROM settings")}

    def set_setting(self, key: str, value: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, value)
        )

    # -- pick returns (选股复盘) ----------------------------------------------
    def upsert_pick_return(self, row: dict) -> None:
        """Insert or refresh one pick's settlement row (keyed by run+code)."""
        self.execute(
            "INSERT INTO pick_returns (run_type, run_id, trade_date, code, name,"
            " pick_price, baseline_price, score, rank, ret_1d, ret_5d, settled_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(run_type, run_id, code) DO UPDATE SET"
            " name=excluded.name, pick_price=excluded.pick_price,"
            " baseline_price=COALESCE(excluded.baseline_price, baseline_price),"
            " score=excluded.score,"
            " rank=excluded.rank,"
            " ret_1d=COALESCE(excluded.ret_1d, ret_1d),"
            " ret_5d=COALESCE(excluded.ret_5d, ret_5d),"
            " settled_at=excluded.settled_at",
            (
                row["run_type"], row["run_id"], row["trade_date"], row["code"],
                row.get("name", ""), row.get("pick_price"), row.get("baseline_price"),
                row.get("score"), row.get("rank"), row.get("ret_1d"),
                row.get("ret_5d"), row.get("settled_at"),
            ),
        )

    def get_pick_returns(self, run_type: str, run_id: str) -> list[dict]:
        return self.fetchall(
            "SELECT * FROM pick_returns WHERE run_type=? AND run_id=?"
            " ORDER BY rank, code",
            (run_type, run_id),
        )

    def pick_return_run_stats(self) -> dict[tuple[str, str], dict]:
        """(run_type, run_id) → settlement aggregates for the review list."""
        rows = self.fetchall(
            "SELECT run_type, run_id, COUNT(*) AS n,"
            " SUM(ret_1d IS NOT NULL) AS settled_1d, AVG(ret_1d) AS avg_1d,"
            " SUM(ret_5d IS NOT NULL) AS settled_5d, AVG(ret_5d) AS avg_5d,"
            " SUM(CASE WHEN ret_5d > 0 THEN 1 ELSE 0 END) AS wins_5d"
            " FROM pick_returns GROUP BY run_type, run_id"
        )
        return {(r["run_type"], r["run_id"]): r for r in rows}


def default_db_path() -> str:
    """Canonical dashboard DB location (also used for UI diagnostics)."""
    _DEFAULT_HOME.mkdir(parents=True, exist_ok=True)
    return str(_DEFAULT_HOME / "dashboard.db")


def resolve_ticker(code: str) -> str:
    """Normalize user-entered codes to the framework's canonical symbol."""
    from tradingagents.dataflows.symbol_utils import normalize_symbol

    return normalize_symbol(code.strip())
