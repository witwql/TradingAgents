"""Main-capital money flow (主力资金) fetcher, shared by the macro analyst
and the next-day picks screener.

Data: EastMoney per-stock daily flow history (主力=超大单+大单). Frames are
cached per (code, trade_date) under data_cache_dir so repeated screener runs
on the same day cost zero EastMoney calls. Every call serializes through
AKSHARE_LOCK (py-mini-racer safety) with bounded retry/backoff.
"""

import logging
import os
import time

import pandas as pd

from .akshare_lock import AKSHARE_LOCK
from .config import get_config
from .errors import NoMarketDataError, VendorNotConfiguredError
from .symbol_utils import ashare_exchange
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

try:
    import akshare as ak
except ImportError:
    ak = None


_RETRIES = 3
_BASE_DELAY = 2.0

# EM 连续失败后的冷却窗口：期间 fetch 直接走 THS 兜底，避免 120 次候选拖死筛选。
_EM_COOLDOWN_SECONDS = 600
_em_fail_until = 0.0


def _quiet(fn, *args, retries=_RETRIES, **kwargs):
    import time

    if ak is None:
        raise VendorNotConfiguredError(
            'akshare package is not installed: pip install "tradingagents[akshare]"'
        )
    fn_name = getattr(fn, "__name__", str(fn))
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with AKSHARE_LOCK:
                return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            if attempt < retries:
                delay = _BASE_DELAY * (2**attempt)
                logger.warning(
                    "money-flow %s transient failure (%d/%d): %s; retrying",
                    fn_name, attempt + 1, retries, exc,
                )
                time.sleep(delay)
    assert last is not None
    raise last


def reset_em_cooldown() -> None:
    """Clear the EM cooldown (tests and manual retries)."""
    global _em_fail_until
    _em_fail_until = 0.0


def _ths_snapshot_frame(bare: str) -> pd.DataFrame | None:
    """THS 全市场即时资金流 → 单行框架（仅当日，无历史）。"""
    try:
        spot = _quiet(ak.stock_fund_flow_individual, symbol="即时", retries=1)
    except Exception as exc:
        logger.warning("THS flow snapshot unavailable: %s", exc)
        return None
    if spot is None or spot.empty:
        return None
    row = spot[spot["股票代码"].astype(str) == bare]
    if row.empty:
        return None
    row = row.iloc[0]

    def parse_yi(v):
        s = str(v).replace("亿", "").strip()
        try:
            return float(s) * 1e8
        except ValueError:
            return None

    net = parse_yi(row.get("净额"))
    turnover = parse_yi(row.get("成交额"))
    pct = round(net / turnover * 100, 2) if (net is not None and turnover) else None
    today = pd.Timestamp.today().normalize()
    return pd.DataFrame([{
        "日期": today.date(),
        "主力净流入-净额": net,
        "主力净流入-净占比": pct,
        "超大单净流入-净额": None,
        "收盘价": pd.to_numeric(row.get("最新价"), errors="coerce"),
        "_source": "ths_snapshot",
    }])


def fetch_money_flow(symbol: str, curr_date: str, lookback_days: int = 40,
                     retries: int = _RETRIES) -> pd.DataFrame:
    """Daily main-capital flow rows on or before ``curr_date``.

    Columns kept: 日期 (datetime index col ``_d``), 主力净流入-净额,
    主力净流入-净占比, 超大单净流入-净额, 收盘价. Raises NoMarketDataError
    when the vendor returns nothing usable; callers treating flow as optional
    enrichment should catch it.
    """
    if ak is None:
        raise VendorNotConfiguredError(
            'akshare package is not installed: pip install "tradingagents[akshare]"'
        )
    bare = str(symbol).strip().upper().rstrip("+").split(".")[0]
    market = (ashare_exchange(bare) or "SZ").lower()

    config = get_config()
    os.makedirs(config["data_cache_dir"], exist_ok=True)
    cache_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_ticker_component(bare)}-Flow-{curr_date}.csv",
    )

    raw = None
    if os.path.exists(cache_file):
        cached = pd.read_csv(cache_file)
        if not cached.empty:
            raw = cached

    global _em_fail_until

    if raw is None:
        em_blocked = time.time() < _em_fail_until
        if not em_blocked:
            try:
                raw = _quiet(ak.stock_individual_fund_flow, stock=bare, market=market,
                             retries=retries)
            except Exception as exc:
                # 探测性失败：进入冷却，本次与后续调用立即走兜底。
                _em_fail_until = time.time() + _EM_COOLDOWN_SECONDS
                logger.warning("EM money flow unavailable (%s); cooling down "
                               "%ss, falling back to THS snapshot", exc,
                               _EM_COOLDOWN_SECONDS)
        if raw is None or getattr(raw, "empty", True):
            raw = _ths_snapshot_frame(bare)
            if raw is None or raw.empty:
                raise NoMarketDataError(symbol, bare, "no money-flow rows returned")
            # 兜底数据只有当日快照：仍写缓存（键含日期，天然隔离）
            raw.to_csv(cache_file, index=False, encoding="utf-8")

    out = raw.copy()
    out["_d"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out[out["_d"].notna() & (out["_d"] <= pd.Timestamp(curr_date))]
    out = out[out["_d"] >= pd.Timestamp(curr_date) - pd.Timedelta(days=max(lookback_days, 25))]
    if out.empty:
        raise NoMarketDataError(symbol, bare, f"no flow rows on or before {curr_date}")
    return out.sort_values("_d").reset_index(drop=True)


__all__ = ["fetch_money_flow", "reset_em_cooldown"]
