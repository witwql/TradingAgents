"""选股复盘（pick review): settle realized T+1/T+5 returns for screener picks.

Screening runs record their picks at selection time; this module closes the
loop by measuring what those picks actually did afterwards, so the screeners'
thresholds ("≥80% probability", "score ≥ 8/16") become falsifiable instead of
narrative.

Settlement is lazy and per-run: the review page lists settled aggregates
without touching the network, and opening one run fetches prices only for
its picks (through the same cached vendor chain the screeners use). A pick
is terminal once its 5-day return is known — later views skip the fetch.
Returns are computed inside one 前复权 OHLCV series, so they are
split/dividend-adjusted by construction.
"""

import json
import logging
import time

from tradingagents.dataflows.symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

RUN_TYPES = ("screen", "value")
RUN_LABELS = {"screen": "动量精选", "value": "价值精选"}

# Give up widening ret_5d once this many future bars exist without reaching
# the 5th (long suspension) — the row stays partial and stops refetching.
_MAX_FUTURE_BARS = 15


def _run_table(run_type: str) -> str:
    if run_type == "screen":
        return "screen_runs"
    if run_type == "value":
        return "value_runs"
    raise ValueError(f"unknown run_type: {run_type}")


def _load_run(db, run_type: str, run_id: str) -> dict | None:
    row = db.fetchone(
        f"SELECT id, trade_date, status, results FROM {_run_table(run_type)} WHERE id=?",
        (run_id,),
    )
    if not row or not row["results"]:
        return None
    try:
        row["payload"] = json.loads(row["results"])
    except json.JSONDecodeError:
        logger.warning("review: corrupt payload for %s run %s", run_type, run_id)
        return None
    return row


def _pick_score(run_type: str, pick: dict) -> float | None:
    value = pick.get("probability") if run_type == "screen" else pick.get("score")
    return float(value) if value is not None else None


def _pick_price(run_type: str, pick: dict) -> float | None:
    value = pick.get("close") if run_type == "screen" else pick.get("price")
    return float(value) if value is not None else None


def realized_returns(frame, trade_date: str,
                     baseline_price: float | None = None) -> tuple[float | None, float | None, float | None]:
    """(baseline, ret_1d, ret_5d) from a daily OHLCV frame.

    Baseline is the last close on or before ``trade_date``; returns compare
    the 1st / 5th close strictly after it. Suspensions simply shift the
    future bars — the 5th *trading* row is the 5-day horizon.
    """
    dates = frame["Date"]
    at_or_before = frame[dates <= trade_date]
    if at_or_before.empty:
        return None, None, None
    baseline = float(at_or_before["Close"].iloc[-1])
    if baseline <= 0:
        return None, None, None
    future = frame[dates > trade_date].head(_MAX_FUTURE_BARS)
    closes = [float(c) for c in future["Close"].tolist()]

    def ret_at(n: int) -> float | None:
        return (closes[n - 1] / baseline - 1) if len(closes) >= n else None

    return baseline, ret_at(1), ret_at(5)


def _load_prices(code: str):
    """Daily OHLCV for one pick via the cached vendor chain; None on failure."""
    from tradingagents.dataflows.sina_stock import fetch_daily_ohlcv_sina
    from datetime import datetime

    canonical = normalize_symbol(code)
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        return fetch_daily_ohlcv_sina(code, canonical, today)
    except Exception as exc:
        logger.warning("review: no OHLCV for %s: %s", code, exc)
        return None


def settle_run(db, run_type: str, run_id: str, price_loader=None) -> dict:
    """Settle one run's picks and return the persisted rows.

    Fetches prices only for picks whose 5-day return is still unknown, so a
    settled run costs zero network on every later view. ``price_loader``
    overrides the OHLCV source for tests.
    """
    if run_type not in RUN_TYPES:
        raise ValueError(f"unknown run_type: {run_type}")
    run = _load_run(db, run_type, run_id)
    if run is None:
        return {"run_type": run_type, "run_id": run_id, "exists": False, "picks": []}

    picks = run["payload"].get("picks", []) or []
    trade_date = run["trade_date"]
    loader = price_loader or _load_prices
    now = time.time()

    if run["status"] != "done":
        return {"run_type": run_type, "run_id": run_id, "exists": True,
                "trade_date": trade_date, "picks": []}

    # Terminal rows (5-day return known) never refetch, so a settled run
    # costs zero network on later views.
    done_codes = {r["code"] for r in db.get_pick_returns(run_type, run_id)
                  if r["ret_5d"] is not None}

    for rank, pick in enumerate(picks, start=1):
        code = str(pick.get("code", "")).strip()
        if not code:
            continue
        if code in done_codes:
            continue
        row = {
            "run_type": run_type,
            "run_id": run_id,
            "trade_date": trade_date,
            "code": code,
            "name": str(pick.get("name", "")),
            "pick_price": _pick_price(run_type, pick),
            "score": _pick_score(run_type, pick),
            "rank": rank,
            "baseline_price": None,
            "ret_1d": None,
            "ret_5d": None,
            "settled_at": now,
        }
        frame = loader(code)
        if frame is not None and not frame.empty:
            baseline, ret_1d, ret_5d = realized_returns(frame, trade_date)
            row["baseline_price"] = baseline
            row["ret_1d"] = ret_1d
            row["ret_5d"] = ret_5d
        db.upsert_pick_return(row)

    settled = db.get_pick_returns(run_type, run_id)
    return {"run_type": run_type, "run_id": run_id, "exists": True,
            "trade_date": trade_date, "picks": settled}


def review_summary(db, limit: int = 12) -> list[dict]:
    """Recent runs of both screeners with whatever settlement is on record.

    Read-only: no network, no settlement — detail views settle lazily.
    """
    stats = db.pick_return_run_stats()
    out = []
    for run_type in RUN_TYPES:
        rows = db.fetchall(
            f"SELECT id, created_at, status, trade_date, results FROM {_run_table(run_type)}"
            " WHERE status IN ('done','cancelled') ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        for row in rows:
            n_picks = 0
            if row["results"]:
                try:
                    n_picks = len(json.loads(row["results"]).get("picks", []) or [])
                except json.JSONDecodeError:
                    pass
            agg = stats.get((run_type, row["id"]), {})
            settled_5d = agg.get("settled_5d") or 0
            out.append({
                "run_type": run_type,
                "label": RUN_LABELS[run_type],
                "run_id": row["id"],
                "trade_date": row["trade_date"],
                "created_at": row["created_at"],
                "status": row["status"],
                "n_picks": n_picks,
                "settled_1d": agg.get("settled_1d") or 0,
                "settled_5d": settled_5d,
                "avg_1d": agg.get("avg_1d"),
                "avg_5d": agg.get("avg_5d"),
                "hit_rate_5d": (agg.get("wins_5d") or 0) / settled_5d if settled_5d else None,
            })
    out.sort(key=lambda r: r["created_at"], reverse=True)
    return out
