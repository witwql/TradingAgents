"""Main-capital money flow (主力资金) fetcher, shared by the macro analyst
and the next-day picks screener.

Data: EastMoney per-stock daily flow history (主力=超大单+大单). Frames are
cached per (code, trade_date) under data_cache_dir so repeated screener runs
on the same day cost zero EastMoney calls. Every call serializes through
AKSHARE_LOCK (py-mini-racer safety) with bounded retry/backoff.
"""

import logging
import os

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

    if raw is None:
        raw = _quiet(ak.stock_individual_fund_flow, stock=bare, market=market,
                     retries=retries)
        if raw is None or raw.empty:
            raise NoMarketDataError(symbol, bare, "no money-flow rows returned")
        raw.to_csv(cache_file, index=False, encoding="utf-8")

    out = raw.copy()
    out["_d"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out[out["_d"].notna() & (out["_d"] <= pd.Timestamp(curr_date))]
    out = out[out["_d"] >= pd.Timestamp(curr_date) - pd.Timedelta(days=max(lookback_days, 25))]
    if out.empty:
        raise NoMarketDataError(symbol, bare, f"no flow rows on or before {curr_date}")
    return out.sort_values("_d").reset_index(drop=True)


__all__ = ["fetch_money_flow"]
