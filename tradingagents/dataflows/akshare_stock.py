"""AKShare vendor implementation for China A-share market data.

AKShare (https://github.com/akfamily/akshare) aggregates free public data
from Chinese market providers (EastMoney, THS, Sina, ...). This module maps
the framework's abstract tool methods onto A-share-appropriate endpoints:

    framework method            AKShare endpoint
    ------------------------    ------------------------------------------
    get_stock_data              stock_zh_a_hist        (EastMoney daily)
    get_indicators              stock_zh_a_hist + stockstats (local calc)
    get_fundamentals            stock_value_em (+ stock_individual_info_em)
    get_balance_sheet           stock_balance_sheet_by_report_em
    get_cashflow                stock_cash_flow_sheet_by_report_em
    get_income_statement        stock_profit_sheet_by_report_em
    get_insider_transactions    stock_shareholder_change_ths

Symbols: the vendor serves SSE/SZSE cash equities only. Accepted inputs are
bare six-digit codes ("600519") or suffixed Yahoo forms ("600519.SS",
"600519.SH", "000001.SZ"). Anything else raises ``NoMarketDataError`` so the
router can fall back to another configured vendor instead of serving wrong
data (#988 contract).

All prices are 前复权 (forward-adjusted) so indicator windows match current
price levels; volume is converted from 手 (lots of 100) to shares.

News endpoints live in :mod:`tradingagents.dataflows.akshare_news`.
"""

import contextlib
import io
import logging
import os
from datetime import datetime, timedelta
from typing import Annotated

import pandas as pd

from .akshare_lock import AKSHARE_LOCK
from .config import get_config
from .errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorRateLimitError,
)
from .stockstats_utils import (
    MAX_OHLCV_STALE_DAYS,
    _assert_ohlcv_not_stale,
    _clean_dataframe,
    _needs_same_day_refresh,
)
from .symbol_utils import ashare_exchange, is_fund_symbol, normalize_symbol
from .utils import prune_superseded_cache_files, safe_ticker_component
from .y_finance import INDICATOR_DESCRIPTIONS

logger = logging.getLogger(__name__)

try:
    import akshare as ak
except ImportError:  # optional dependency: configured but not installed
    ak = None


def _require_ak():
    if ak is None:
        raise VendorNotConfiguredError(
            "akshare package is not installed. Install the optional extra: "
            'pip install "tradingagents[akshare]"'
        )


def _no_market(symbol: str, detail: str) -> NoMarketDataError:
    """Typed error carrying both requested and attempted canonical symbols."""
    return NoMarketDataError(symbol, normalize_symbol(symbol), detail)


def to_acode(symbol: str) -> str:
    """Return the bare six-digit A-share code for a framework symbol.

    Accepts ``600519``, ``600519.SS``/``.SH``/``.SS``, ``000001.SZ`` and
    in-market fund codes (``510300.SS``, ``159994.SZ``, bare too).
    Raises NoMarketDataError for anything that cannot be an SSE/SZSE
    instrument, which lets the router fall through to the next vendor.
    """
    if not isinstance(symbol, str):
        raise _no_market(str(symbol), "symbol must be a string")
    s = symbol.strip().upper().rstrip("+")

    if len(s) == 6 and s.isdigit():
        code = s
    elif (
        len(s) == 9 and s[:6].isdigit() and s[6] == "." and s[7:] in ("SS", "SH", "SZ")
    ):
        code = s[:6]
    else:
        raise _no_market(
            symbol,
            f"'{symbol}' is not an A-share code (expected e.g. 600519.SS "
            "or 000001.SZ); the akshare vendor only covers SSE/SZSE equities",
        )

    if ashare_exchange(code) is None:
        raise _no_market(
            symbol,
            f"A-share code '{code}' is outside the auto-supported SSE/SZSE "
            "ranges (use an explicit exchange suffix or another vendor)",
        )
    return code


def to_em_symbol(symbol: str) -> str:
    """Exchange-prefixed form used by EastMoney report-sheet endpoints."""
    code = to_acode(symbol)
    prefix = ashare_exchange(code) or "SZ"
    return f"{prefix}{code}"


_QUIET_RETRIES = 4
_QUIET_BASE_DELAY = 2.0

# Transient transport failures worth one bounded backoff. Public Chinese-market
# endpoints intermittently abort python-requests connections outright.
try:
    import requests as _requests
    _TRANSIENT_ERRORS = (
        ConnectionError,
        TimeoutError,
        _requests.exceptions.ConnectionError,
        _requests.exceptions.Timeout,
        _requests.exceptions.ChunkedEncodingError,
    )
except ImportError:
    _TRANSIENT_ERRORS = (ConnectionError, TimeoutError)


def _quiet(fn, *args, **kwargs):
    """Call an AKShare endpoint with backoff, swallowing its tqdm bars."""

    _require_ak()
    fn_name = getattr(fn, "__name__", str(fn))
    # py-mini_racer (V8) inside akshare is process-global and thread-hostile.
    with AKSHARE_LOCK:
        return _quiet_locked(fn, fn_name, *args, **kwargs)


def _quiet_locked(fn, fn_name, *args, **kwargs):
    import time

    last: Exception | None = None
    for attempt in range(_QUIET_RETRIES + 1):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                return fn(*args, **kwargs)
        except _TRANSIENT_ERRORS as exc:
            last = exc
            if attempt < _QUIET_RETRIES:
                delay = _QUIET_BASE_DELAY * (2**attempt)
                logger.warning(
                    "AKShare %s transient failure (attempt %d/%d): %s; retrying",
                    fn_name, attempt + 1, _QUIET_RETRIES, exc,
                )
                time.sleep(delay)
    assert last is not None  # retries exhausted
    # N consecutive dropped connections on a public endpoint behave exactly
    # like throttling — classify them that way so declared fallback chains
    # take over instead of the run dying (routed callers only).
    raise VendorRateLimitError(
        f"AKShare {fn_name}: transport failures persisted through "
        f"{_QUIET_RETRIES} retries ({type(last).__name__})"
    ) from last


def fetch_daily_ohlcv_akshare(
    symbol: str,
    canonical: str,
    curr_date: str,
    max_stale_days: int = MAX_OHLCV_STALE_DAYS,
) -> pd.DataFrame:
    """Five-year qfq daily OHLCV for an A-share, cached and look-ahead safe.

    Mirrors ``stockstats_utils.load_ohlcv`` semantics (same-day TTL refresh,
    staleness guard, rows-after-curr_date filtering) but sources rows from
    AKShare/EastMoney under its own ``-AkShare-`` cache namespace so switching
    vendors never serves one vendor's cached frame to the other.
    """
    _require_ak()
    acode = to_acode(canonical)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)

    today_date = pd.Timestamp.today()
    start_str = (today_date - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end_str = today_date.strftime("%Y-%m-%d")  # AKShare end date is inclusive

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_ticker_component(canonical)}-AkShare-data-{start_str}-{end_str}.csv",
    )

    data = None
    if os.path.exists(data_file):
        cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
        if (
            not cached.empty
            and "Close" in cached.columns
            and not _needs_same_day_refresh(data_file, curr_date_dt, today_date)
        ):
            data = cached

    if data is None:
        if is_fund_symbol(acode):
            raw = _quiet(
                ak.fund_etf_hist_em,
                symbol=acode,
                period="daily",
                start_date=start_str.replace("-", ""),
                end_date=end_str.replace("-", ""),
                adjust="qfq",
            )
        else:
            raw = _quiet(
                ak.stock_zh_a_hist,
                symbol=acode,
                period="daily",
                start_date=start_str.replace("-", ""),
                end_date=end_str.replace("-", ""),
                adjust="qfq",
            )
        data = _normalize_eastmoney_frame(raw, symbol, canonical)
        if data.empty or "Close" not in data.columns:
            raise NoMarketDataError(
                symbol, canonical, f"AKShare returned no rows for {acode}"
            )
        data.to_csv(data_file, index=False, encoding="utf-8")
        # Filename embeds today's date — superseded windows are dead weight.
        prune_superseded_cache_files(
            data_file,
            os.path.join(config["data_cache_dir"],
                         f"{safe_ticker_component(canonical)}-AkShare-data-*.csv"),
        )

    data = _clean_dataframe(data)

    # Look-ahead protection: never expose rows past the analysis date.
    data = data[data["Date"] <= curr_date_dt]

    _assert_ohlcv_not_stale(
        data, curr_date, symbol, canonical, max_stale_days=max_stale_days
    )
    return data


def _normalize_eastmoney_frame(raw: pd.DataFrame, symbol: str, canonical: str) -> pd.DataFrame:
    """Map the EastMoney Chinese daily frame onto the framework OHLCV schema."""
    if raw is None or raw.empty:
        raise NoMarketDataError(
            symbol, canonical, f"AKShare returned no rows for {to_acode(canonical)}"
        )

    renames = {
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
    }
    missing = [c for c in renames if c not in raw.columns]
    if missing:
        raise NoMarketDataError(
            symbol, canonical, f"daily frame missing expected columns: {missing}"
        )

    data = raw.rename(columns=renames)[
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ].copy()
    # 成交量 is in lots (1 lot = 100 shares); convert to shares so volume-based
    # indicators (MFI/VWMA) and the verified snapshot read the same unit as
    # the yfinance vendor does.
    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce") * 100
    return data


def get_stock_data_akshare(
    symbol: Annotated[str, "A-share ticker, e.g. 600519.SS or 000001.SZ"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """OHLCV window for an A-share from AKShare (EastMoney daily, qfq)."""
    datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    canonical = normalize_symbol(symbol)
    data = fetch_daily_ohlcv_akshare(symbol, canonical, end_date)

    data = data[(data["Date"] >= pd.Timestamp(start_date)) & (data["Date"] <= end_dt)]
    if data.empty:
        raise NoMarketDataError(
            symbol, canonical, f"no rows between {start_date} and {end_date}"
        )
    data = data.reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close"]:
        data[col] = data[col].round(2)

    csv_string = data.to_csv(index=False)
    header = f"# Stock data for {canonical} [A-share {to_acode(symbol)}] from {start_date} to {end_date}\n"
    header += "# Source: AKShare (EastMoney daily, qfq-adjusted), volume in shares\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


def _indicator_lookup_bulk(data: pd.DataFrame, indicator: str) -> dict[str, str]:
    """Compute an indicator over every cached row via stockstats."""
    from stockstats import wrap

    df = wrap(data.copy())
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df[indicator]  # triggers calculation

    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        value = row[indicator]
        lookup[row["Date"]] = "N/A" if pd.isna(value) else str(value)
    return lookup


def get_indicators_akshare(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Indicator history for an A-share, same shape as the yfinance vendor."""
    if indicator not in INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. "
            f"Please choose from: {list(INDICATOR_DESCRIPTIONS.keys())}"
        )

    canonical = normalize_symbol(symbol)
    data = fetch_daily_ohlcv_akshare(symbol, canonical, curr_date)

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_dt - timedelta(days=look_back_days)
    lookup = _indicator_lookup_bulk(data, indicator)

    ind_string = ""
    day = curr_dt
    while day >= before:
        key = day.strftime("%Y-%m-%d")
        value = lookup.get(key, "N/A: Not a trading day (weekend or holiday)")
        ind_string += f"{key}: {value}\n"
        day -= timedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + INDICATOR_DESCRIPTIONS.get(indicator, "No description available.")
    )
    return result_str


# --------------------------------------------------------------------------
# Fundamentals & financial statements (EastMoney)
# --------------------------------------------------------------------------


def _fund_not_applicable(label: str) -> str:
    return (
        f"Not applicable: '{label}' is an exchange-traded fund (ETF/LOF), so "
        f"corporate {label} data does not exist. Index-tracking vehicles are "
        "assessed through price action, discount/premium to NAV, and the "
        "underlying index constituents — analyze it with market/news tools "
        "instead."
    )

_VALUATION_FIELDS = [
    # (column in stock_value_em frame, output label)
    ("数据日期", "Valuation Date"),
    ("当日收盘价", "Close Price"),
    ("总市值", "Total Market Cap"),
    ("流通市值", "Float Market Cap"),
    ("PE(TTM)", "PE Ratio (TTM)"),
    ("PE(静)", "PE Ratio (Static)"),
    ("市净率", "Price to Book"),
    ("PEG值", "PEG Ratio"),
    ("市现率", "Price to Cash Flow"),
    ("市销率", "Price to Sales"),
]


def _company_name(acode: str) -> str | None:
    """Best-effort company name/industry; degrades silently when blocked."""
    try:
        info = _quiet(ak.stock_individual_info_em, symbol=acode)
        if info is None or info.empty:
            return None
        kv = dict(zip(info["item"], info["value"], strict=False))
        name = kv.get("股票简称")
        industry = kv.get("行业")
        if industry:
            return f"{name} ({industry})" if name else str(industry)
        return name
    except Exception as exc:
        logger.warning("akshare individual-info unavailable for %s: %s", acode, exc)
        return None


def get_fundamentals_akshare(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date YYYY-MM-DD (valuation snapshot cutoff)"] = None,
) -> str:
    """Company fundamentals overview from AKShare (EastMoney valuation)."""
    canonical = normalize_symbol(ticker)
    acode = to_acode(ticker)
    if is_fund_symbol(acode):
        return _fund_not_applicable("fundamentals")

    try:
        valuation = _quiet(ak.stock_value_em, symbol=acode)
    except VendorNotConfiguredError:
        raise
    except Exception as exc:
        return f"Error retrieving fundamentals for {ticker}: {exc}"

    if valuation is None or valuation.empty:
        raise NoMarketDataError(ticker, canonical, "no valuation history returned")

    if curr_date:
        valuation = valuation[pd.to_datetime(valuation["数据日期"]) <= pd.Timestamp(curr_date)]
    if valuation.empty:
        raise NoMarketDataError(ticker, canonical, f"no valuation rows on or before {curr_date}")

    latest = valuation.iloc[-1]

    lines = []
    name = _company_name(acode)
    if name:
        lines.append(f"Name: {name}")
    lines.append(f"Code: {acode}")
    for col, label in _VALUATION_FIELDS:
        value = latest.get(col)
        if value is not None and not pd.isna(value):
            lines.append(f"{label}: {value}")

    header = f"# Company Fundamentals for {canonical} ({acode})\n"
    header += "# Source: AKShare EastMoney daily valuation\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + "\n".join(lines)


def _pick_columns(row: pd.Series, mapping: list[tuple[str, str]]) -> list[tuple[str, object]]:
    """Extract mapped fields present in the report row, case-insensitive keys."""
    upper = {str(k).upper(): k for k in row.index}
    picked = []
    for field, label in mapping:
        col = upper.get(field.upper())
        if col is not None and pd.notna(row[col]):
            picked.append((label, row[col]))
    return picked


_INCOME_FIELDS = [
    ("TOTAL_OPERATE_INCOME", "营业总收入 Total Operating Revenue"),
    ("OPERATE_COST", "营业成本 Operating Cost"),
    ("OPERATE_PROFIT", "营业利润 Operating Profit"),
    ("TOTAL_PROFIT", "利润总额 Total Profit"),
    ("NETPROFIT", "净利润 Net Profit"),
    ("PARENT_NETPROFIT", "归母净利润 Net Profit Attributable to Parent"),
    ("BASIC_EPS", "基本每股收益 Basic EPS"),
    ("WEIGHTAVG_ROE", "加权平均ROE Weighted Avg ROE"),
]

_BALANCE_FIELDS = [
    ("TOTAL_ASSETS", "总资产 Total Assets"),
    ("TOTAL_CURRENT_ASSETS", "流动资产 Total Current Assets"),
    ("MONETARYFUNDS", "货币资金 Cash & Equivalents"),
    ("ACCOUNTS_RECE", "应收账款 Accounts Receivable"),
    ("INVENTORY", "存货 Inventory"),
    ("TOTAL_LIABILITIES", "总负债 Total Liabilities"),
    ("TOTAL_CURRENT_LIAB", "流动负债 Total Current Liabilities"),
    ("TOTAL_EQUITY", "股东权益合计 Total Equity"),
    ("TOTAL_PARENT_EQUITY", "归母股东权益 Equity Attributable to Parent"),
]

_CASHFLOW_FIELDS = [
    ("NETCASH_OPERATE", "经营活动现金流净额 Net Operating Cash Flow"),
    ("NETCASH_INVEST", "投资活动现金流净额 Net Investing Cash Flow"),
    ("NETCASH_FINANCE", "筹资活动现金流净额 Net Financing Cash Flow"),
    ("CCE_ADD", "现金及等价物净增加额 Net Change in Cash"),
]

_STATEMENT_META = {
    "balance_sheet": (
        "Balance Sheet",
        "stock_balance_sheet_by_report_em",
        _BALANCE_FIELDS,
    ),
    "cashflow": (
        "Cash Flow Statement",
        "stock_cash_flow_sheet_by_report_em",
        _CASHFLOW_FIELDS,
    ),
    "income_statement": (
        "Income Statement",
        "stock_profit_sheet_by_report_em",
        _INCOME_FIELDS,
    ),
}

_QUARTERLY_PERIODS = 8
_ANNUAL_PERIODS = 4


def _fetch_statement_report(symbol: str, kind: str) -> pd.DataFrame:
    fn_name = _STATEMENT_META[kind][1]
    fn = getattr(ak, fn_name, None)
    if fn is None:
        raise RuntimeError(f"AKShare has no endpoint {fn_name}; upgrade the akshare package")
    em_symbol = to_em_symbol(symbol)
    return _quiet(fn, symbol=em_symbol)


def _statement_periods(report: pd.DataFrame, freq: str, curr_date: str | None) -> pd.DataFrame:
    """Latest N reported periods, strictly look-ahead filtered by notice date.

    Uses NOTICE_DATE (公告日) when present rather than REPORT_DATE: figures not
    yet published on the analysis date must never reach the agents.
    """
    dates = pd.to_datetime(
        report["NOTICE_DATE"] if "NOTICE_DATE" in report.columns else report["REPORT_DATE"],
        errors="coerce",
    )
    cutoff = pd.Timestamp(curr_date) if curr_date else pd.Timestamp.max
    visible = report[dates.notna() & (dates <= cutoff)]

    if visible.empty:
        return visible

    parsed = pd.to_datetime(visible["REPORT_DATE"], errors="coerce")
    if freq.lower() != "quarterly":
        visible = visible[parsed.dt.month == 12]

    order = dates.loc[visible.index].sort_values(ascending=False)
    visible = visible.loc[order.index].head(
        _ANNUAL_PERIODS if freq.lower() != "quarterly" else _QUARTERLY_PERIODS
    )
    return visible.sort_index()


def _render_statement(
    ticker: str, canonical: str, kind: str, freq: str, curr_date: str | None
) -> str:
    title, _, fields = _STATEMENT_META[kind]

    try:
        report = _fetch_statement_report(ticker, kind)
    except VendorNotConfiguredError:
        raise
    except NoMarketDataError:
        raise
    except Exception as exc:
        return f"Error retrieving {title.lower()} for {ticker}: {exc}"

    if report is None or report.empty:
        raise NoMarketDataError(ticker, canonical, f"no {title.lower()} reports returned")

    visible = _statement_periods(report, freq, curr_date)
    if visible.empty:
        raise NoMarketDataError(
            ticker,
            canonical,
            f"no {title.lower()} periods published on or before {curr_date}",
        )

    blocks = []
    for _, row in visible.iterrows():
        report_date = str(pd.to_datetime(row["REPORT_DATE"]).date())
        notice_date = ""
        if "NOTICE_DATE" in row.index and pd.notna(row["NOTICE_DATE"]):
            notice_date = str(pd.to_datetime(row["NOTICE_DATE"]).date())
        picked = _pick_columns(row, fields)
        body = "\n".join(f"- {label}: {value}" for label, value in picked) or "- (fields unavailable)"
        suffix = f", disclosed {notice_date}" if notice_date else ""
        blocks.append(f"### Report period {report_date}{suffix}\n{body}")

    header = f"# {title} data for {canonical} ({freq}, reported)\n"
    header += "# Source: AKShare EastMoney periodic reports, amounts in CNY\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + "\n\n".join(blocks)


def get_balance_sheet_akshare(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Reported balance sheet periods for an A-share from AKShare."""
    canonical = normalize_symbol(ticker)
    if is_fund_symbol(to_acode(ticker)):
        return _fund_not_applicable("balance sheet")
    return _render_statement(ticker, canonical, "balance_sheet", freq, curr_date)


def get_cashflow_akshare(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Reported cash flow statement periods for an A-share from AKShare."""
    canonical = normalize_symbol(ticker)
    if is_fund_symbol(to_acode(ticker)):
        return _fund_not_applicable("cash flow statement")
    return _render_statement(ticker, canonical, "cashflow", freq, curr_date)


def get_income_statement_akshare(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Reported income statement periods for an A-share from AKShare."""
    canonical = normalize_symbol(ticker)
    if is_fund_symbol(to_acode(ticker)):
        return _fund_not_applicable("income statement")
    return _render_statement(ticker, canonical, "income_statement", freq, curr_date)


def get_insider_transactions_akshare(
    ticker: Annotated[str, "ticker symbol of the company"],
) -> str:
    """Executive/major-holder shareholding changes (A股高管及股东增减持).

    The closest A-share analogue to insider filings: major shareholder and
    executive stake changes from THS via AKShare.
    """
    canonical = normalize_symbol(ticker)
    acode = to_acode(ticker)
    if is_fund_symbol(acode):
        return _fund_not_applicable("shareholder changes")
    try:
        changes = _quiet(ak.stock_shareholder_change_ths, symbol=acode)
    except VendorNotConfiguredError:
        raise
    except Exception as exc:
        return f"Error retrieving shareholder changes for {ticker}: {exc}"

    if changes is None or changes.empty:
        return f"No shareholder/executive stake changes reported for '{acode}'"

    if "公告日期" in changes.columns:
        changes = changes.sort_values("公告日期", ascending=False)
    recent = changes.head(20)

    csv_string = recent.to_csv(index=False)
    header = f"# Shareholder & Executive Stake Changes for {canonical} ({acode})\n"
    header += "# Source: AKShare THS (增减持公告, most recent first)\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


__all__ = [
    "ak",
    "fetch_daily_ohlcv_akshare",
    "get_balance_sheet_akshare",
    "get_cashflow_akshare",
    "get_fundamentals_akshare",
    "get_income_statement_akshare",
    "get_indicators_akshare",
    "get_insider_transactions_akshare",
    "get_stock_data_akshare",
    "to_acode",
    "to_em_symbol",
]
