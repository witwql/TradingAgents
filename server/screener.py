"""明日精选 screener: rank main-board stocks by next-day-up probability.

Model contract (honest by design):
- A multi-factor resonance engine scores each stock on five binary setups
  (pullback-in-uptrend, volume breakout, MACD zero-line cross, RSI strength,
  shrinking-volume stabilization).
- Every factor's probability contribution comes from the STOCK'S OWN 500-day
  history: P(next-day up | factor fired), with sample size shown. No factor
  with fewer than 30 historical occurrences is trusted.
- Composite probability is a weighted-linear lift over the stock's base
  up-rate, hard-clamped to [0.05, 0.95]. Requiring >= 0.80 with >= 3 factors
  firing means most days output few or zero picks — that is the honest
  behavior, not a bug.
- Universe is restricted to ordinary-account main-board codes
  (600/601/603/605, 000/001/002/003), ST/delisting excluded.

This is a research/learning tool. Historical conditional frequencies are not
a guarantee of future results; nothing here is investment advice.
"""

import contextlib
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from tradingagents.dataflows.akshare_lock import AKSHARE_LOCK

logger = logging.getLogger(__name__)

MAIN_BOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
MIN_PRICE = 2.0
MIN_TURNOVER = 2e8          # 成交额 2 亿
MAX_CANDIDATES = 120        # 流动性排名后的入池深度
MIN_FACTORS_FIRED = 3
PROBABILITY_THRESHOLD = 0.80
MAX_PICKS = 5
MIN_FACTOR_SAMPLES = 30     # 单因子历史样本下限
HISTORY_ROWS = 500          # 因子统计窗口（交易日）

_UNIVERSE_TTL = 900         # 股票池缓存 15 分钟


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def fetch_universe() -> list[dict]:
    """Sina full-market spot -> filtered main-board liquid candidates."""
    import akshare as ak

    with AKSHARE_LOCK:
        spot = ak.stock_zh_a_spot()

    rows = []
    for _, r in spot.iterrows():
        raw_code = str(r["代码"])            # e.g. sh600519 / sz000001 / bj920000
        name = str(r["名称"])
        exchange = raw_code[:2]
        code = raw_code[2:]
        if exchange not in ("sh", "sz") or not code.startswith(MAIN_BOARD_PREFIXES):
            continue
        if "ST" in name.upper() or "退" in name:
            continue
        price = pd.to_numeric(r.get("最新价"), errors="coerce")
        turnover = pd.to_numeric(r.get("成交额"), errors="coerce")
        volume = pd.to_numeric(r.get("成交量"), errors="coerce")
        if pd.isna(price) or price < MIN_PRICE:
            continue
        if pd.isna(turnover) or turnover < MIN_TURNOVER or pd.isna(volume) or volume <= 0:
            continue
        rows.append({
            "code": code,
            "name": name,
            "price": float(price),
            "turnover": float(turnover),
        })

    rows.sort(key=lambda x: x["turnover"], reverse=True)
    return rows[:MAX_CANDIDATES]


# ---------------------------------------------------------------------------
# Factor engine (vectorized over one stock's history)
# ---------------------------------------------------------------------------

def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean factor columns aligned to each historical date."""
    d = df.reset_index(drop=True).copy()
    close, high, low, vol = d["Close"], d["High"], d["Low"], d["Volume"]

    d["ma5"] = close.rolling(5).mean()
    d["ma10"] = close.rolling(10).mean()
    d["ma20"] = close.rolling(20).mean()
    d["vol5"] = vol.rolling(5).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    d["dif"], d["dea"] = dif, dea
    d["hist"] = dif - dea
    d["hist_prev"] = d["hist"].shift(1)
    d["macd_cross_up"] = (d["dif"] > d["dea"]) & (d["dif"].shift(1) <= d["dea"].shift(1))
    d["hist_grow_red"] = (d["hist"] > 0) & (d["hist"] > d["hist_prev"])

    d["rsi"] = _wilder_rsi(close)
    d["rsi_prev"] = d["rsi"].shift(1)
    d["high20_prev"] = high.shift(1).rolling(20).max()

    # F1 多头排列回踩 MA10（今日最低触及 MA10±2% 且收于其上）
    d["f1_pullback"] = (
        (d["ma5"] > d["ma10"]) & (d["ma10"] > d["ma20"])
        & (low <= d["ma10"] * 1.02) & (close > d["ma10"])
    )
    # F2 放量突破 20 日高
    d["f2_breakout"] = (close > d["high20_prev"]) & (vol >= 1.5 * d["vol5"])
    # F3 MACD 零上金叉或红柱放大
    d["f3_macd"] = (d["dif"] > 0) & (d["macd_cross_up"] | d["hist_grow_red"])
    # F4 RSI 强势区且上行
    d["f4_rsi"] = (d["rsi"] >= 55) & (d["rsi"] <= 72) & (d["rsi"] > d["rsi_prev"])
    # F5 缩量回踩收阳企稳（前两日收跌、今日缩量收阳、守 20 日线）
    prev_down = (close.shift(1) < close.shift(2)) & (close.shift(2) < close.shift(3))
    d["f5_shrink_stabilize"] = (
        prev_down & (vol <= 0.8 * d["vol5"])
        & (close > close.shift(1)) & (close >= d["ma20"])
    )
    return d


FACTOR_LABELS = {
    "f1_pullback": "多头排列回踩MA10",
    "f2_breakout": "放量突破20日高",
    "f3_macd": "MACD零上金叉/红柱放大",
    "f4_rsi": "RSI强势区上行",
    "f5_shrink_stabilize": "缩量回踩收阳企稳",
}

FACTOR_COLUMNS = list(FACTOR_LABELS.keys())


def _factor_stats(d: pd.DataFrame) -> dict[str, tuple[float, int]]:
    """Per-factor historical P(next-day up | fired) + sample count."""
    next_up = (d["Close"].shift(-1) > d["Close"]).astype(float)
    stats = {}
    for col in FACTOR_COLUMNS:
        fired = d[col] == True  # noqa: E712 — pandas boolean mask
        n = int(fired.sum())
        if n >= MIN_FACTOR_SAMPLES:
            p = float(next_up[fired].mean())
        else:
            p, n = 0.0, int(n)  # below sample floor: unusable
        stats[col] = (p, n)
    return stats


def _usable(stats: dict) -> dict[str, tuple[float, int]]:
    return {k: v for k, v in stats.items() if v[1] >= MIN_FACTOR_SAMPLES and v[0] > 0}


def _composite_probability(d: pd.DataFrame, stats: dict) -> tuple[float | None, list[dict], int]:
    """Today's composite P(next-day up) using only today-fired, usable factors."""
    p0 = float((d["Close"].shift(-1) > d["Close"]).mean())
    p0 = min(max(p0, 0.35), 0.65)  # degenerate-history guard

    fired, contributions, weight_sum, lift_sum = 0, [], 0.0, 0.0
    for col in FACTOR_COLUMNS:
        if not bool(d[col].iloc[-1]):
            continue
        p, n = stats[col]
        fired += 1
        if n < MIN_FACTOR_SAMPLES:
            contributions.append({
                "factor": FACTOR_LABELS[col], "fired": True,
                "p": None, "n": n, "used": False,
                "note": "样本不足，未计入",
            })
            continue
        w = min(1.0, n / 150)
        lift = (p - p0) * w
        lift_sum += lift
        weight_sum += w
        contributions.append({
            "factor": FACTOR_LABELS[col], "fired": True,
            "p": round(p, 4), "n": n, "used": True,
            "lift": round(lift, 4),
        })

    if fired < MIN_FACTORS_FIRED or weight_sum == 0:
        return None, contributions, fired

    prob = p0 + lift_sum
    return min(max(prob, 0.05), 0.95), contributions, fired


def _resonance_calibration(d: pd.DataFrame) -> tuple[float | None, int]:
    """Historical hit-rate on days when >=3 factors co-fired (calibration)."""
    fired_count = d[FACTOR_COLUMNS].sum(axis=1)
    resonance = fired_count >= MIN_FACTORS_FIRED
    n = int(resonance.sum())
    if n < 5:
        return None, n
    next_up = (d["Close"].shift(-1) > d["Close"])
    return float(next_up[resonance].mean()), n


# ---------------------------------------------------------------------------
# Per-stock pipeline
# ---------------------------------------------------------------------------

def evaluate_stock(code: str, name: str, price: float, curr_date: str) -> dict | None:
    """Full evaluation for one candidate; None when data unusable."""
    from tradingagents.dataflows.sina_stock import fetch_daily_ohlcv_sina

    canonical = f"{code}.SS" if code.startswith(("6",)) else f"{code}.SZ"
    try:
        df = fetch_daily_ohlcv_sina(canonical, canonical, curr_date)
    except Exception as exc:
        logger.warning("screener: %s history unavailable: %s", code, exc)
        return None
    if df is None or len(df) < 60:
        return None

    d = df.tail(HISTORY_ROWS).reset_index(drop=True)
    d = compute_factors(d)

    # Look-ahead guard: drop rows after curr_date (cache can hold fresher rows).
    d = d[pd.to_datetime(d["Date"]) <= pd.Timestamp(curr_date)]
    if d.empty or len(d) < 60:
        return None

    stats = _factor_stats(d)
    prob, contributions, fired = _composite_probability(d, stats)
    if prob is None:
        return None

    hit, hit_n = _resonance_calibration(d)
    last = d.iloc[-1]
    return {
        "code": code,
        "name": name,
        "close": float(last["Close"]),
        "probability": round(prob, 4),
        "factors_fired": fired,
        "contributions": contributions,
        "resonance_hit_rate": round(hit, 4) if hit is not None else None,
        "resonance_samples": hit_n,
        "history_days": int(len(d)),
    }


# ---------------------------------------------------------------------------
# Runner (background thread + sqlite persistence via db param)
# ---------------------------------------------------------------------------

def run_screening(db, curr_date: str | None = None) -> tuple[str, bool]:
    """Start a screening run; returns (run_id, already_running).

    A running run (started <30min ago) is reused instead of spawning a
    duplicate, so double-clicks / re-entry are harmless.
    """
    running = db.fetchone(
        "SELECT id, created_at FROM screen_runs WHERE status='running'"
        " ORDER BY created_at DESC LIMIT 1"
    )
    if running and time.time() - running["created_at"] < 1800:
        return running["id"], True

    run_id = uuid.uuid4().hex[:10]
    db.execute(
        "INSERT INTO screen_runs (id, created_at, status, trade_date, stage)"
        " VALUES (?,?,?,?,?)",
        (run_id, time.time(), "running",
         curr_date or datetime.now().strftime("%Y-%m-%d"), "universe"),
    )
    threading.Thread(target=_run_blocking, args=(db, run_id, curr_date), daemon=True).start()
    return run_id, False


def _run_blocking(db, run_id: str, curr_date: str | None):
    import datetime as dt

    try:
        curr = curr_date or dt.date.today().strftime("%Y-%m-%d")
        universe = fetch_universe()
        logger.info("screener: universe=%d candidates", len(universe))
        db.execute(
            "UPDATE screen_runs SET stage='analyzing', total=?, processed=0 WHERE id=?",
            (len(universe), run_id),
        )

        results = []
        done_count = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(evaluate_stock, u["code"], u["name"], u["price"], curr): u
                for u in universe
            }
            for fut in as_completed(futures):
                done_count += 1
                r = fut.result()
                if r:
                    results.append(r)
                # Live progress: every finished future counts toward the bar,
                # even when the stock produced no composite (fewer than
                # MIN_FACTORS_FIRED) — otherwise the bar would jumpp.
                qualifying_now = sum(
                    1 for x in results if x["probability"] >= PROBABILITY_THRESHOLD
                )
                db.execute(
                    "UPDATE screen_runs SET processed=?, qualifying=? WHERE id=?",
                    (done_count, qualifying_now, run_id),
                )

        picks, watchlist = split_results(results)

        payload = json.dumps({
            "evaluated": len(results),
            "qualifying": len(picks),
            "picks": picks,
            "watchlist": watchlist,
            "finished_at_ts": time.time(),
        }, ensure_ascii=False)
        db.execute(
            "UPDATE screen_runs SET status='done', finished_at=?, universe=?, results=? WHERE id=?",
            (time.time(), len(universe), payload, run_id),
        )
    except Exception as exc:
        logger.exception("screening run %s failed", run_id)
        db.execute(
            "UPDATE screen_runs SET status='failed', finished_at=?, error=? WHERE id=?",
            (time.time(), str(exc)[:500], run_id),
        )


def split_results(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """(达标picks, 观察名单): threshold gate then probability ranking."""
    by_prob = sorted(results, key=lambda x: x["probability"], reverse=True)
    qualifying = [r for r in by_prob if r["probability"] >= PROBABILITY_THRESHOLD]
    return qualifying[:MAX_PICKS], by_prob[:MAX_PICKS]


def history(db, limit: int = 10) -> list[dict]:
    rows = db.fetchall(
        "SELECT id, created_at, finished_at, status, trade_date, universe, results"
        " FROM screen_runs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    out = []
    for row in rows:
        results = None
        if row["results"]:
            with contextlib.suppress(json.JSONDecodeError):
                results = json.loads(row["results"])
        out.append({
            "id": row["id"],
            "created_at": row["created_at"],
            "finished_at": row["finished_at"],
            "status": row["status"],
            "trade_date": row["trade_date"],
            "universe": row["universe"],
            "evaluated": (results or {}).get("evaluated"),
            "qualifying": (results or {}).get("qualifying"),
            "top_probability": max(
                (p["probability"] for p in (results or {}).get("picks", [])),
                default=None,
            ),
        })
    return out


def latest_run(db) -> dict | None:
    row = db.fetchone("SELECT * FROM screen_runs ORDER BY created_at DESC LIMIT 1")
    if not row:
        return None
    if row["results"]:
        try:
            row["results"] = json.loads(row["results"])
        except json.JSONDecodeError:
            logger.warning("screen run %s has corrupt results; dropping payload", row["id"])
            row["results"] = None
    return row
