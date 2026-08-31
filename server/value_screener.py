"""Value+quality screener: undervalued, profitable, growing A-share main-board stocks.

Complements the technical momentum screener with a fundamental lens. Every
metric comes from unthrottled hosts (Sina financial ratios + Baidu daily
valuation); EastMoney's push2his is never touched.

Two-phase evaluation keeps it fast: a quick Sina ratio pass filters ~250
candidates to ~80 with decent ROE/growth/debt, then Baidu PB/市值 is fetched
only for survivors to compute the final 0-16 score.

Score dimensions (0-16):
  低估: PB (0-3) + PE estimate (0-3)
  盈利: ROE (0-3) + 净利率 (0-2)
  成长: 营收增长率 (0-2) + 净利增长率 (0-2)
  安全: 资产负债率 (0-2/-1) + 流动比率 (0-1)
"""

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from tradingagents.dataflows.akshare_lock import AKSHARE_LOCK
from tradingagents.dataflows.symbol_utils import is_main_board_ashare

logger = logging.getLogger(__name__)

MAX_VALUE_CANDIDATES = 250
MIN_VALUE_TURNOVER = 5e7        # 成交额 5000 万（价值股流动性偏低）
SCORE_THRESHOLD = 8             # 8/16 以上入榜
MAX_VALUE_PICKS = 15
_RETRIES = 2

try:
    import akshare as ak
except ImportError:
    ak = None


def _quiet(fn, *args, **kwargs):
    import contextlib
    import io

    last = None
    for attempt in range(_RETRIES + 1):
        try:
            with AKSHARE_LOCK, contextlib.redirect_stderr(io.StringIO()):
                return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            if attempt < _RETRIES:
                import time as _t

                _t.sleep(1.5 * (2**attempt))
    raise last


# ---------------------------------------------------------------------------
# Universe (broader than momentum: lower turnover floor, more candidates)
# ---------------------------------------------------------------------------

def fetch_value_universe() -> list[dict]:
    import akshare as _ak

    with AKSHARE_LOCK:
        spot = _ak.stock_zh_a_spot()

    rows = []
    for _, r in spot.iterrows():
        raw_code = str(r["代码"])
        name = str(r["名称"])
        exchange = raw_code[:2]
        code = raw_code[2:]
        if exchange not in ("sh", "sz") or not is_main_board_ashare(code):
            continue
        if "ST" in name.upper() or "退" in name:
            continue
        price = pd.to_numeric(r.get("最新价"), errors="coerce")
        turnover = pd.to_numeric(r.get("成交额"), errors="coerce")
        if pd.isna(price) or price < 2.0:
            continue
        if pd.isna(turnover) or turnover < MIN_VALUE_TURNOVER:
            continue
        rows.append({"code": code, "name": name, "price": float(price),
                     "turnover": float(turnover)})
    rows.sort(key=lambda x: x["turnover"], reverse=True)
    return rows[:MAX_VALUE_CANDIDATES]


# ---------------------------------------------------------------------------
# Per-stock evaluation
# ---------------------------------------------------------------------------

# Sina financial indicator → EN label + scoring function
_RATIO_FIELDS = [
    # (cn_column, en_label, score_fn(value) -> 0..3)
    ("净资产收益率(%)", "ROE", lambda v: 3 if v > 20 else (2 if v > 12 else (1 if v > 8 else 0))),
    ("销售净利率(%)", "净利率", lambda v: 2 if v > 30 else (1 if v > 15 else 0)),
    ("主营业务收入增长率(%)", "营收增长", lambda v: 2 if v > 15 else (1 if v > 5 else (0.5 if v > 0 else 0))),
    ("净利润增长率(%)", "净利增长", lambda v: 2 if v > 20 else (1 if v > 10 else (0.5 if v > 0 else 0))),
    ("资产负债率(%)", "负债率", lambda v: 2 if v < 30 else (1 if v < 50 else (0 if v < 70 else -1))),
    ("流动比率", "流动比率", lambda v: 1 if v > 2 else (0.5 if v > 1 else 0)),
]


def _score_pb(pb):
    if pb is None or pd.isna(pb):
        return 0
    return 3 if pb < 1.5 else (2 if pb < 3 else (1 if pb < 5 else 0))


def _score_pe(pe):
    if pe is None or pd.isna(pe) or pe <= 0:
        return 0
    return 3 if pe < 15 else (2 if pe < 25 else (1 if pe < 40 else 0))


def evaluate_value_stock(code: str, name: str, price: float, curr_date: str) -> dict | None:
    """Two-phase: Sina ratios first (quick gate), Baidu PB only for survivors."""

    # Phase 1: Sina financial analysis indicator (quick, ~0.6s)
    try:
        yr = str((pd.Timestamp(curr_date) if curr_date else pd.Timestamp.today()).year - 1)
        ratios = _quiet(ak.stock_financial_analysis_indicator, symbol=code, start_year=yr)
    except Exception as exc:
        logger.debug("value screener: %s ratios unavailable: %s", code, exc)
        return None

    if ratios is None or ratios.empty:
        return None

    r = ratios.iloc[-1]
    metrics = {}
    quick_score = 0
    for cn, en, score_fn in _RATIO_FIELDS:
        v = r.get(cn)
        if v is not None and pd.notna(v):
            s = score_fn(float(v))
            metrics[en] = {"value": round(float(v), 2), "score": s}
            quick_score += s

    # Quick gate: ROE > 5% AND no catastrophic metrics → proceed to Baidu
    roe_metric = metrics.get("ROE", {})
    if roe_metric.get("score", 0) == 0 and (roe_metric.get("value", 0) or 0) < 5:
        return None

    eps = r.get("摊薄每股收益(元)")
    pe_est = float(price) / float(eps) if eps and pd.notna(eps) and float(eps) > 0 else None

    # Phase 2: Baidu PB + market cap (only for survivors, ~0.5s)
    pb = None
    mktcap = None
    try:
        pb_df = _quiet(ak.stock_zh_valuation_baidu, symbol=code,
                       indicator="市净率", period="近一年")
        if pb_df is not None and not pb_df.empty:
            pb = float(pd.to_numeric(pb_df["value"], errors="coerce").iloc[-1])

        mc_df = _quiet(ak.stock_zh_valuation_baidu, symbol=code,
                       indicator="总市值", period="近一年")
        if mc_df is not None and not mc_df.empty:
            mktcap = float(pd.to_numeric(mc_df["value"], errors="coerce").iloc[-1])
    except Exception as exc:
        logger.debug("value screener: %s baidu unavailable: %s", code, exc)

    pb_score = _score_pb(pb)
    pe_score = _score_pe(pe_est)
    total_score = quick_score + pb_score + pe_score

    metrics["PB"] = {"value": round(pb, 2) if pb else None, "score": pb_score}
    metrics["PE"] = {"value": round(pe_est, 1) if pe_est else None, "score": pe_score}
    if mktcap:
        metrics["总市值"] = {"value": round(mktcap, 0), "score": 0}

    return {
        "code": code,
        "name": name,
        "price": round(price, 2),
        "score": round(total_score, 1),
        "max_score": 16,
        "metrics": metrics,
        "period": str(r.get("日期", "")),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_value_screening(db, curr_date: str | None = None) -> tuple[str, bool]:
    run_id = uuid.uuid4().hex[:10]
    trade_date = curr_date or datetime.now().strftime("%Y-%m-%d")

    running = db.fetchone(
        "SELECT id FROM value_runs WHERE status='running' AND created_at > ? LIMIT 1",
        (time.time() - 1800,),
    )
    if running:
        return running["id"], True

    db.execute(
        "INSERT INTO value_runs (id, created_at, status, trade_date, stage)"
        " VALUES (?,?,?,?,?)",
        (run_id, time.time(), "running", trade_date, "universe"),
    )
    threading.Thread(target=_run_value_blocking, args=(db, run_id, trade_date), daemon=True).start()
    return run_id, False


def _run_value_blocking(db, run_id: str, curr_date: str):
    try:
        universe = fetch_value_universe()
        logger.info("value screener: universe=%d", len(universe))
        db.execute(
            "UPDATE value_runs SET stage='analyzing', total=? WHERE id=?",
            (len(universe), run_id),
        )

        results = []
        done_count = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(evaluate_value_stock, u["code"], u["name"], u["price"], curr_date): u
                for u in universe
            }
            for fut in as_completed(futures):
                done_count += 1
                r = fut.result()
                if r:
                    results.append(r)
                qualifying = sum(1 for x in results if x["score"] >= SCORE_THRESHOLD)
                flag = db.fetchone(
                    "SELECT cancel_requested FROM value_runs WHERE id=?", (run_id,)
                )
                if flag and flag["cancel_requested"]:
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
                db.execute(
                    "UPDATE value_runs SET processed=?, qualifying=? WHERE id=?",
                    (done_count, qualifying, run_id),
                )

        qualifying = sorted(
            [r for r in results if r["score"] >= SCORE_THRESHOLD],
            key=lambda x: x["score"], reverse=True,
        )
        picks = qualifying[:MAX_VALUE_PICKS]
        watchlist = sorted(
            [r for r in results if r["score"] >= SCORE_THRESHOLD - 2 and r not in picks],
            key=lambda x: x["score"], reverse=True,
        )[:MAX_VALUE_PICKS]

        payload = json.dumps({
            "evaluated": len(results),
            "qualifying": len(qualifying),
            "picks": picks,
            "watchlist": watchlist,
            "finished_at_ts": time.time(),
        }, ensure_ascii=False)
        db.execute(
            "UPDATE value_runs SET status='done', finished_at=?, universe=?,"
            " results=? WHERE id=?",
            (time.time(), len(universe), payload, run_id),
        )
    except Exception as exc:
        logger.exception("value screening %s failed", run_id)
        db.execute(
            "UPDATE value_runs SET status='failed', finished_at=?, error=? WHERE id=?",
            (time.time(), str(exc)[:500], run_id),
        )


def latest_value_run(db) -> dict | None:
    row = db.fetchone("SELECT * FROM value_runs ORDER BY created_at DESC LIMIT 1")
    if not row:
        return None
    if row["results"]:
        try:
            row["results"] = json.loads(row["results"])
        except json.JSONDecodeError:
            row["results"] = None
    return row


def value_history(db, limit: int = 10) -> list[dict]:
    rows = db.fetchall(
        "SELECT id, created_at, finished_at, status, trade_date, universe, results"
        " FROM value_runs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    out = []
    for row in rows:
        results = None
        if row["results"]:
            with __import__("contextlib").suppress(json.JSONDecodeError):
                results = json.loads(row["results"])
        out.append({
            "id": row["id"], "created_at": row["created_at"],
            "finished_at": row["finished_at"], "status": row["status"],
            "trade_date": row["trade_date"], "universe": row["universe"],
            "evaluated": (results or {}).get("evaluated"),
            "qualifying": (results or {}).get("qualifying"),
            "top_score": max(
                (p["score"] for p in (results or {}).get("picks", [])),
                default=None,
            ),
        })
    return out
