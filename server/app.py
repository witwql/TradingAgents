"""FastAPI application exposing the dashboard API and static frontend.

Endpoints
---------
GET  /                     single-page frontend (server/static)
GET  /api/health           liveness + provider key presence probe
POST /api/tasks            submit analyses (tickers list -> one task per ticker)
GET  /api/tasks[?status]   task list / filter
GET  /api/tasks/{id}       task detail incl. workflow stages
DELETE /api/tasks/{id}     cancel pending or delete finished rows
GET  /api/tasks/{id}/events  Server-Sent Events stream (replays history first)
GET  /api/tasks/{id}/reports    report-tree manifest (markdown files)
GET  /api/tasks/{id}/report?path=  raw markdown content of one report file
GET/POST/DELETE /api/favorites
GET/PUT /api/settings      non-secret runtime knobs (GLM region/model etc.)
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .db import Database, default_db_path, resolve_ticker
from .queue import TaskQueue
from .runner import ALL_ANALYST_KEYS, STAGE_LABELS

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# 全市场 ETF 现货行情刷新周期（秒）。东财单次返回 1000+ 行，曾内联在请求里
# 拖慢自选股页面数秒乃至更久；改为后台定时刷新，请求永远走缓存秒回。
_SPOT_REFRESH_SECONDS = int(
    os.environ.get("TRADINGAGENTS_DASHBOARD_SPOT_REFRESH", "300")
)

# SSE 空闲心跳间隔（秒）。LLM 推理间隙无事件流出，心跳防止连接被掐断。
SSE_HEARTBEAT_SECONDS = int(os.environ.get("TRADINGAGENTS_DASHBOARD_SSE_HEARTBEAT", "15"))


class SpotQuoteCache:
    """Background-refreshed spot quotes for the watchlist.

    ``fetch(favorite_codes)`` runs on a daemon timer — Sina per-code quotes by
    default (unthrottled). Reads are lock-protected snapshots, always instant.
    ``ready`` turns True once any pass returned rows; failures keep serving the
    previous snapshot (stale-while-revalidate).
    """

    def __init__(self, fetch, codes_provider=lambda: [],
                 refresh_seconds: int = _SPOT_REFRESH_SECONDS):
        import threading

        self._codes_provider = codes_provider
        self._fetch = fetch
        self._seconds = max(30, int(refresh_seconds))
        self._rows: dict[str, dict] = {}
        self._ts = 0.0
        self._ready = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self):
        import threading

        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="spot-quotes", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                rows = self._fetch(self._codes_provider())
                if rows:
                    with self._lock:
                        self._rows = rows
                        self._ts = __import__("time").time()
                        self._ready = True
            except Exception as exc:
                logger.warning("spot quotes refresh failed: %s", exc)
            # Before the first success, retry quickly (watchlist is often added
            # right after startup); afterwards settle into the normal cadence.
            with self._lock:
                ready = self._ready
            self._stop.wait(15 if not ready else self._seconds)

    def get(self) -> tuple[dict, float, bool]:
        with self._lock:
            return dict(self._rows), self._ts, self._ready


def _fetch_sina_spot(codes: list[str]) -> dict[str, dict]:
    """Per-code realtime quotes from Sina hq (GB18030, Referer-gated).

    Sina hosts stayed unthrottled through heavy usage while EastMoney dropped
    connections, so this is the favorites page's PRIMARY quote source; it only
    costs one tiny HTTP call for the handful of watched codes.
    """
    import requests

    if not codes:
        return {}
    symbols = ",".join(_sina_code(c) for c in codes)
    resp = requests.get(
        f"https://hq.sinajs.cn/list={symbols}",
        headers={"Referer": "https://finance.sina.com.cn"},
        timeout=8,
    )
    resp.raise_for_status()
    text = resp.content.decode("gb18030", errors="replace")

    rows: dict[str, dict] = {}
    for line in text.strip().splitlines():
        try:
            key, payload = line.split("=", 1)
            code = key.removeprefix("var hq_str_").strip()
            fields = payload.strip().strip('"').split(",")
            if len(fields) < 4 or not fields[0]:
                continue
            prev_close, last = float(fields[2]), float(fields[3])
            pct = (last - prev_close) / prev_close * 100 if prev_close else None
            rows[code[2:]] = {  # strip sh/sz prefix back to bare code
                "name": fields[0],
                "price": last,
                "pct": pct,
            }
        except Exception:
            logger.debug("sina spot line unparsed: %r", line, exc_info=True)
    return rows


def _sina_code(bare: str) -> str:
    """510300 -> sh510300 / 159994 -> sz159994 / 600519 -> sh600519."""
    from tradingagents.dataflows.symbol_utils import ashare_exchange

    exchange = ashare_exchange(bare) or "SZ"
    return exchange.lower() + bare

# Secret-free settings surface: values the dashboard may read/write.
_SETTING_KEYS = {
    "glm_region": ("glm-cn", {"glm-cn", "glm"}),
    "glm_model": ("glm-5.2", None),
    "quick_model": (None, None),
    "deep_model": (None, None),
    "temperature": (None, None),
}


def create_app(db: Database | None = None, queue: TaskQueue | None = None,
               start_spot: bool = True, args_db_path: str | None = None) -> FastAPI:
    db = db or Database()
    db_path = args_db_path if args_db_path else default_db_path()
    queue = queue or TaskQueue(db)
    queue.start()
    spot_cache = SpotQuoteCache(
        _fetch_sina_spot,
        codes_provider=lambda: [r["code"].split(".")[0] for r in db.list_favorites()],
    )
    if start_spot:
        spot_cache.start()
    app = FastAPI(title="TradingAgents Dashboard", version="0.1.0")
    app.state.db = db
    app.state.queue = queue
    app.state.spot = spot_cache

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "db_path": db.path() or db_path,
            "env_home_override": bool(os.environ.get("TRADINGAGENTS_DASHBOARD_HOME")),
            "llm_region": db.get_settings().get("glm_region", "glm-cn"),
            "has_zhipu_cn_key": bool(
                os.environ.get("ZHIPU_CN_API_KEY")
                or _dotenv_has("ZHIPU_CN_API_KEY")
            ),
            "has_zhipu_intl_key": bool(
                os.environ.get("ZHIPU_API_KEY") or _dotenv_has("ZHIPU_API_KEY")
            ),
        }

    @app.get("/")
    @app.get("/dashboard")
    def index():
        return HTMLResponse((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # ----------------------------------------------------------------- tasks
    @app.post("/api/tasks", status_code=201)
    async def submit_tasks(request: Request):
        body = await request.json()
        tickers_raw = body.get("tickers") or []
        if isinstance(tickers_raw, str):
            tickers_raw = [t for chunk in tickers_raw.split(",") for t in chunk.split()]
        tickers = sorted({resolve_ticker(t) for t in tickers_raw if str(t).strip()})
        if not tickers:
            raise HTTPException(422, "at least one ticker is required")
        trade_date = str(body.get("trade_date") or "").strip() or _today_str()
        analysts = body.get("analysts") or list(ALL_ANALYST_KEYS)
        bad = set(analysts) - set(ALL_ANALYST_KEYS)
        if bad:
            raise HTTPException(422, f"unknown analysts: {sorted(bad)}")
        spec = {
            "tickers": tickers,
            "trade_date": trade_date,
            "analysts": list(analysts),
            "debate_rounds": int(body.get("debate_rounds", 1)),
            "risk_rounds": int(body.get("risk_rounds", 1)),
            "output_language": body.get("output_language", "Chinese") or "Chinese",
        }
        try:
            ids = queue.submit(spec)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"task_ids": ids}

    @app.get("/api/tasks")
    def list_tasks(status: str | None = None, limit: int = 100):
        return {"tasks": db.list_tasks(limit=min(limit, 500), status=status)}

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        task = db.get_task(task_id)
        if not task:
            raise HTTPException(404, task_id)
        return {**task, "stages": db.get_stages(task_id)}

    @app.delete("/api/tasks/{task_id}")
    def delete_task(task_id: str):
        task = db.get_task(task_id)
        if not task:
            raise HTTPException(404, task_id)
        if task["status"] == "running":
            raise HTTPException(409, "cannot cancel a running analysis; wait for it to finish")
        if task["status"] == "pending":
            queue.cancel(task_id)
        else:
            db.delete_task(task_id)
        return {"deleted": task_id}

    @app.get("/api/tasks/{task_id}/events")
    async def task_events(task_id: str):
        if not db.get_task(task_id):
            raise HTTPException(404, task_id)

        async def stream():
            last_id = 0
            terminal = {"completed", "failed", "cancelled"}
            # LLM calls think for minutes with zero events; without a heartbeat
            # proxies/browsers silently kill the idle SSE connection and the
            # UI freezes. Comments ("src/app.js reads nothing") keep it alive.
            # No explicit disconnect polling: it blocks on idle streams under
            # the test transport, and real servers cancel the generator on
            # client hangup anyway.
            last_activity = asyncio.get_event_loop().time()
            while True:
                events = await asyncio.to_thread(db.events_since, task_id, last_id)
                for ev in events:
                    last_id = ev["id"]
                    data = json.dumps(
                        {"id": ev["id"], "type": ev["type"], **ev["payload"]},
                        ensure_ascii=False,
                    )
                    yield f"id: {ev['id']}\ndata: {data}\n\n"
                    last_activity = asyncio.get_event_loop().time()
                    if ev["type"] == "status" and ev["payload"].get("status") in terminal:
                        return
                now = asyncio.get_event_loop().time()
                if now - last_activity >= SSE_HEARTBEAT_SECONDS:
                    yield ": keep-alive\n\n"
                    last_activity = now
                await asyncio.sleep(0.6)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --------------------------------------------------------------- reports
    @app.get("/api/tasks/{task_id}/reports")
    def report_manifest(task_id: str):
        task = db.get_task(task_id)
        if not task:
            raise HTTPException(404, task_id)
        base = Path(task["report_dir"]) if task["report_dir"] else None
        files: list[dict] = []
        if base and base.exists():
            for path in sorted(base.rglob("*.md")):
                rel = str(path.relative_to(base))
                files.append({"path": rel, "size": path.stat().st_size})
        return {"report_dir": str(base) if base else "", "files": files}

    @app.get("/api/tasks/{task_id}/report")
    def report_content(task_id: str, path: str):
        task = db.get_task(task_id)
        if not task or not task["report_dir"]:
            raise HTTPException(404, "no report saved for this task")
        base = Path(task["report_dir"]).resolve()
        target = (base / path).resolve()
        if not str(target).startswith(str(base)) or target.suffix != ".md":
            raise HTTPException(400, "invalid report path")
        if not target.exists():
            raise HTTPException(404, path)
        return {"path": path, "content": target.read_text(encoding="utf-8")}

    # -------------------------------------------------------------- favorites
    @app.get("/api/favorites")
    def favorites_list():
        from tradingagents.dataflows.symbol_utils import is_fund_symbol

        spot, quote_ts, ready = app.state.spot.get()
        items = []
        for row in db.list_favorites():
            bare = row["code"].split(".")[0]
            info = spot.get(bare, {})
            items.append({
                **row,
                "is_fund": is_fund_symbol(bare),
                # name persists at add-time; spot name only fills the gap.
                "name": row.get("name") or info.get("name", ""),
                **{k: info.get(k) for k in ("price", "pct")},
            })
        return {
            "favorites": items,
            "quotes_ready": ready,
            "quote_ts": quote_ts,
            "refresh_seconds": _SPOT_REFRESH_SECONDS,
        }

    @app.post("/api/favorites", status_code=201)
    async def favorites_add(request: Request):
        body = await request.json()
        raw = str(body.get("code", "")).strip().upper().rstrip("+")
        bare = raw.split(".")[0]
        if not (len(bare) == 6 and bare.isdigit()):
            raise HTTPException(422, "请输入 6 位数字代码（A股/ETF）")
        name = str(body.get("name", "")).strip()
        if not name:
            # Best-effort: fill the display name from whatever snapshot exists.
            spot, _ts, _ready = app.state.spot.get()
            name = (spot.get(bare) or {}).get("name", "")
        db.add_favorite(bare, name)
        return {"code": bare, "name": name}

    @app.delete("/api/favorites/{code}")
    def favorites_remove(code: str):
        bare = code.strip().upper().split(".")[0]
        db.remove_favorite(bare)
        return {"removed": bare}

    # --------------------------------------------------------------- screener
    @app.post("/api/screen", status_code=202)
    def screen_start():
        """Start a screening run in the background; poll GET for results."""
        from .screener import run_screening

        run_id, already_running = run_screening(db)
        return {"run_id": run_id, "already_running": already_running}

    @app.get("/api/screen/history")
    def screen_history(limit: int = 10):
        from .screener import history

        return {"runs": history(db, limit=min(limit, 50))}

    @app.get("/api/screen/latest")
    def screen_latest():
        from .screener import latest_run

        run = latest_run(db)
        if not run:
            return {"run": None}
        return {
            "run": {
                "id": run["id"],
                "status": run["status"],
                "trade_date": run["trade_date"],
                "created_at": run["created_at"],
                "finished_at": run["finished_at"],
                "universe": run["universe"],
                "stage": run.get("stage", ""),
                "processed": run.get("processed", 0),
                "total": run.get("total", 0),
                "error": run["error"],
                "evaluated": (run["results"] or {}).get("evaluated"),
                "qualifying": (run["results"] or {}).get("qualifying"),
                "picks": (run["results"] or {}).get("picks", []),
                "watchlist": (run["results"] or {}).get("watchlist", []),
            }
        }

    # ---------------------------------------------------------------- settings
    @app.get("/api/settings")
    def settings_get():
        stored = db.get_settings()
        out = {}
        for key, (default, _) in _SETTING_KEYS.items():
            out[key] = stored.get(key, default)
        out["stage_labels"] = STAGE_LABELS
        return out

    @app.put("/api/settings")
    async def settings_put(request: Request):
        body = await request.json()
        for key, value in body.items():
            if key not in _SETTING_KEYS:
                raise HTTPException(422, f"unknown setting '{key}'")
            allowed = _SETTING_KEYS[key][1]
            if allowed is not None and value is not None and value not in allowed:
                raise HTTPException(422, f"invalid value for {key}: {value}")
            if value is not None and value != "":
                db.set_setting(key, str(value))
        return settings_get()

    return app


def _today_str() -> str:
    import datetime as dt

    return dt.date.today().strftime("%Y-%m-%d")


def _dotenv_has(var: str) -> bool:
    """Check a repo/.env file without leaking any value."""
    candidates = [Path.cwd() / ".env"]
    if not any(p.exists() for p in candidates):
        return False
    for line in candidates[0].read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith(f"{var}=") and len(line.split("=", 1)[-1].strip()) > 0:
            return True
        if line.startswith(f"{var}") and "=" not in line:
            continue
    return False
