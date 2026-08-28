import logging
import os
import time
from typing import Annotated

import pandas as pd
import yfinance as yf
from stockstats import wrap
from yfinance.exceptions import YFRateLimitError

from .config import get_config
from .errors import (
    NoMarketDataError as _NoMarketDataErrorBase,  # noqa: F401
    VendorRateLimitError,
)
from .symbol_utils import NoMarketDataError, normalize_symbol
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

# A vendor's latest OHLCV row this many calendar days before the requested date
# is treated as stale. Generous enough to span long holiday weekends, tight
# enough to catch the year-old frames yfinance occasionally returns (#1021).
MAX_OHLCV_STALE_DAYS = 10

# How long a same-day cache that does not yet reach the requested day may be
# reused before it is refetched (#1150). Short enough that an intraday run picks
# up today's close soon after it publishes, long enough that a day with no bar
# at all (weekend, holiday) cannot trigger a download on every call.
OHLCV_CACHE_TTL_SECONDS = 900


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately. Exhausted
    limits surface as the routing layer's typed ``VendorRateLimitError``
    so multi-vendor chains skip to the next source instead of aborting.
    """
    from .errors import VendorRateLimitError

    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise VendorRateLimitError(
                    "Yahoo Finance 429 rate limit exhausted after "
                    f"{max_retries} retries"
                ) from None


def _ensure_date_column(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the date column to ``Date``.

    Some yfinance builds leave the index unnamed (so ``reset_index()`` yields
    ``index``) or use ``Datetime`` for intraday data. Rename the first
    date-like column so indicators don't silently drop when it isn't ``Date``.
    """
    if "Date" in data.columns:
        return data
    for candidate in ("index", "Datetime", "date"):
        if candidate in data.columns:
            return data.rename(columns={candidate: "Date"})
    return data


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data = _ensure_date_column(data)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def _coerce_ohlcv_dates(data: pd.DataFrame) -> pd.Series:
    """Return parsed dates from an OHLCV frame, whether Date is a column or the index."""
    if "Date" in data.columns:
        return pd.to_datetime(data["Date"], errors="coerce").dropna()
    # yfinance keeps the dates in the index (a DatetimeIndex, sometimes unnamed).
    if isinstance(data.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(data.index, errors="coerce")).dropna()
    # Fallback: expose the index and look for any date-like column.
    df = data.reset_index()
    for col in ("Date", "Datetime", "date", "index"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if not parsed.empty:
                return parsed
    return pd.Series(dtype="datetime64[ns]")


def _assert_ohlcv_not_stale(
    data: pd.DataFrame,
    curr_date: str,
    symbol: str,
    canonical: str | None = None,
    *,
    max_stale_days: int = MAX_OHLCV_STALE_DAYS,
) -> None:
    """Reject OHLCV whose latest row is far older than curr_date.

    Raises NoMarketDataError (with a stale-specific detail) so the router treats
    it like any other "no usable data from this vendor" — try the next vendor,
    then emit one clear unavailable signal. Empty frames are left to the
    caller's existing no-data handling; this guards only the dangerous case of
    present-but-stale rows (a vendor returning a year-old frame that would
    otherwise feed wrong prices to the agent, #1021).
    """
    if data is None or data.empty:
        return
    requested = pd.to_datetime(curr_date, errors="coerce")
    if pd.isna(requested):
        return
    requested = requested.normalize()
    dates = _coerce_ohlcv_dates(data)
    if dates.empty:
        return
    latest = dates.max().normalize()
    stale_days = (requested - latest).days
    if stale_days > max_stale_days:
        raise NoMarketDataError(
            symbol,
            canonical,
            f"latest row is {latest.date()}, {stale_days} days before the "
            f"requested {requested.date()} (stale) — refusing to use it",
        )


def _needs_same_day_refresh(data_file, curr_date_dt, today_date) -> bool:
    """Whether a cached frame must be refetched to reflect the requested day.

    The cache file is keyed per day, so without this a run started before the
    day's bar was final keeps serving that snapshot to every later run (#1150).
    Two distinct staleness cases exist for a current-day request: the bar may be
    missing entirely, or present but still in progress — Yahoo publishes a
    partial daily candle during market hours, whose ``Close`` is not the closing
    price. Row inspection cannot tell a partial bar from a final one, so the TTL
    governs every current-day cache. Historical requests always reuse the cache,
    since those rows are immutable.
    """
    if curr_date_dt.date() < today_date.date():
        return False
    return time.time() - os.path.getmtime(data_file) > OHLCV_CACHE_TTL_SECONDS


def _resolved_ohlcv_chain() -> list[str]:
    """Configured OHLCV source chain for indicator calc & verified snapshots.

    Mirrors router precedence: tool-level override, then the technical
    indicators category, then core stock APIs. Multi-vendor chains are
    honored in declared order (e.g. ``akshare,yfinance``), keeping the
    dashboard preset resilient when the primary Chinese endpoint throttles.
    """
    config = get_config()
    tool_vendors = config.get("tool_vendors") or {}
    data_vendors = config.get("data_vendors") or {}
    chain = (
        tool_vendors.get("get_stock_data")
        or data_vendors.get("technical_indicators")
        or data_vendors.get("core_stock_apis")
        or "yfinance"
    )
    return [v.strip().lower() for v in str(chain).split(",") if v.strip()]


def _fetch_ohlcv_source(source: str, symbol: str, canonical: str,
                        start_str: str, end_str: str) -> pd.DataFrame | None:
    """One OHLCV source probe. Returns rows or None when unusable."""
    if source == "akshare":
        from tradingagents.dataflows.akshare_stock import (
            fetch_daily_ohlcv_akshare as _fetch_aks,
        )

        # The adapter owns its own 5y cache + staleness + look-ahead filtering.
        return _fetch_aks(symbol, canonical, end_str)

    if source == "sina":
        from tradingagents.dataflows.sina_stock import (
            fetch_daily_ohlcv_sina as _fetch_sin,
        )

        return _fetch_sin(symbol, canonical, end_str)

    if source == "yfinance":
        canonical_safe = safe_ticker_component(canonical)
        config = get_config()
        os.makedirs(config["data_cache_dir"], exist_ok=True)
        data_file = os.path.join(
            config["data_cache_dir"],
            f"{canonical_safe}-YFin-data-{start_str}-{end_str}.csv",
        )
        data = None
        if os.path.exists(data_file):
            cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
            today_date = pd.Timestamp.today()
            curr_dt = pd.to_datetime(end_str)
            if (
                not cached.empty
                and "Close" in cached.columns
                and not _needs_same_day_refresh(data_file, curr_dt, today_date)
            ):
                data = cached

        if data is None:
            downloaded = yf_retry(lambda: yf.download(
                canonical,
                start=start_str,
                end=end_str,
                multi_level_index=False,
                progress=False,
                auto_adjust=True,
            ))
            downloaded = _ensure_date_column(downloaded.reset_index())
            # Only persist real data — an empty/columnless frame is never
            # written to disk (poisoned-cache guard) and reports as no rows.
            if (
                not isinstance(downloaded, pd.DataFrame)
                or downloaded.empty
                or "Close" not in downloaded.columns
            ):
                raise NoMarketDataError(symbol, canonical, "yfinance returned no rows")
            downloaded.to_csv(data_file, index=False, encoding="utf-8")
            data = downloaded

        if not isinstance(data, pd.DataFrame) or data.empty or "Close" not in data.columns:
            return None
        data = _clean_dataframe(data)
        return data.sort_values("Date").reset_index(drop=True)

    logger.warning("Unknown OHLCV source %r skipped", source)
    return None


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Sources follow the configured vendor chain (default ``yfinance``): each
    source gets one attempt, a clean miss/unavailable result moves to the
    next, and the last error surfaces if every source is exhausted. Rows
    after curr_date are always filtered so backtests never see future prices.
    """
    canonical = normalize_symbol(symbol)

    chain = _resolved_ohlcv_chain()
    today_date = pd.Timestamp.today()
    start_str = (today_date - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end_inclusive = (today_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = None
    last_error: Exception | None = None
    for source in chain:
        try:
            candidate = _fetch_ohlcv_source(
                source, symbol, canonical, start_str, end_inclusive
            )
        except (NoMarketDataError, VendorRateLimitError) as exc:
            last_error = exc
            logger.warning("OHLCV source %s unusable for %s: %s", source, symbol, exc)
            continue
        except Exception as exc:
            last_error = exc
            logger.warning("OHLCV source %s failed for %s: %s", source, symbol, exc)
            continue
        if candidate is not None and not candidate.empty:
            data = candidate
            break
        last_error = last_error or NoMarketDataError(
            symbol, canonical, f"{source} returned no rows"
        )

    if data is None:
        raise last_error or NoMarketDataError(symbol, canonical, "no OHLCV source")

    curr_date_dt = pd.to_datetime(curr_date)

    # Look-ahead protection shared by every source: drop rows past the
    # analysis date, then reject year-old frames before they feed indicators.
    data = data[data["Date"] <= curr_date_dt]
    _assert_ohlcv_not_stale(data, curr_date, symbol, canonical)
    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
