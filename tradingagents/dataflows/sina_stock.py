"""Sina Finance vendor: independent fallback family for the A-share preset.

Serves the same framework contracts as the AKShare vendor but through Sina /
Tencent host families, which sit on different infrastructure from EastMoney —
when the dashboard's declared chain ``akshare,sina,yfinance`` hits an
EastMoney throttle storm or Yahoo 429s, this vendor keeps prices flowing.

    framework method            endpoint
    ------------------------    ----------------------------------------
    get_stock_data              stock_zh_a_daily (qfq)
    get_indicators              stock_zh_a_daily + stockstats
    get_fundamentals            latest statements + daily close summary
    get_balance_sheet           stock_financial_report_sina(资产负债表)
    get_cashflow                stock_financial_report_sina(现金流量表)
    get_income_statement        stock_financial_report_sina(利润表)

Unregistered on purpose: news / insider endpoints in AKShare are EastMoney- or
THS-hosted, so those categories fall back elsewhere.

Look-ahead contract: Sina publishes no disclosure dates, so statement cutoffs
use the reporting-period end date — slightly looser than EastMoney's NOTICE_DATE
filtering. This is a deliberate, documented tradeoff for a *fallback* vendor;
the primary path keeps strict disclosure-date gating.

Prices are 前复权; volume arrives in shares (no lot conversion needed).
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
from .errors import NoMarketDataError, VendorNotConfiguredError
from .stockstats_utils import (
    MAX_OHLCV_STALE_DAYS,
    _assert_ohlcv_not_stale,
    _clean_dataframe,
    _needs_same_day_refresh,
)
from .symbol_utils import ashare_exchange, is_fund_symbol, normalize_symbol
from .utils import safe_ticker_component
from .y_finance import INDICATOR_DESCRIPTIONS

logger = logging.getLogger(__name__)

try:
    import akshare as ak
except ImportError:
    ak = None


def _require_ak():
    if ak is None:
        raise VendorNotConfiguredError(
            'akshare package is not installed: pip install "tradingagents[akshare]"'
        )


_QUIET_RETRIES = 3
_QUIET_BASE_DELAY = 2.0


def _quiet(fn, *args, **kwargs):
    """Retry wrapper mirroring the AKShare vendor (bound backoff, tqdm-hush)."""

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
        except Exception as exc:
            last = exc
            if attempt < _QUIET_RETRIES:
                delay = _QUIET_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Sina %s transient failure (attempt %d/%d): %s; retrying",
                    fn_name, attempt + 1, _QUIET_RETRIES, exc,
                )
                time.sleep(delay)
    assert last is not None
    raise last


def to_sina_symbol(symbol: str) -> str:
    """Framework symbol -> Sina ``sh600519`` / ``sz000001`` / ``sh510300`` form."""
    if not isinstance(symbol, str):
        raise NoMarketDataError(str(symbol), detail="symbol must be a string")
    code = to_acode(symbol)
    exchange = ashare_exchange(code) or ("SH" if code.startswith("6") else "SZ")
    return exchange.lower() + code


def to_acode(symbol: str) -> str:
    """Six-digit A-share / ETF code; raises NoMarketDataError otherwise."""
    s = str(symbol).strip().upper().rstrip("+")
    if len(s) == 9 and s[:6].isdigit() and s[6] == "." and s[7:] in ("SS", "SH", "SZ"):
        return s[:6]
    if len(s) == 6 and s.isdigit():
        return s
    raise NoMarketDataError(
        symbol, normalize_symbol(symbol),
        f"'{symbol}' is not an SSE/SZSE instrument; the sina vendor covers "
        "A-share stocks and in-market funds only",
    )


_DAILY_RENAMES = {
    "date": "Date", "open": "Open", "high": "High",
    "low": "Low", "close": "Close", "volume": "Volume",
}


def fetch_daily_ohlcv_sina(
    symbol: str,
    canonical: str,
    curr_date: str,
    max_stale_days: int = MAX_OHLCV_STALE_DAYS,
) -> pd.DataFrame:
    """Five-year qfq daily OHLCV via Sina; mirrors the other vendors' caching,
    look-ahead filtering and staleness guard under its own cache namespace."""
    _require_ak()
    scode = to_sina_symbol(canonical)

    config = get_config()
    curr_date_dt = pd.to_datetime(curr_date)
    today_date = pd.Timestamp.today()

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_ticker_component(canonical)}-Sina-data-"
        f"{(today_date - pd.DateOffset(years=5)).strftime('%Y-%m-%d')}-"
        f"{today_date.strftime('%Y-%m-%d')}.csv",
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
        raw = _quiet(_sina_daily_fn(scode), symbol=scode, adjust="qfq")
        data = _normalize_frame(raw, symbol, canonical)
        if data.empty or "Close" not in data.columns:
            raise NoMarketDataError(symbol, canonical, f"sina returned no rows for {scode}")
        data.to_csv(data_file, index=False, encoding="utf-8")

    data = _clean_dataframe(data)
    data = data[data["Date"] <= curr_date_dt]
    _assert_ohlcv_not_stale(data, curr_date, symbol, canonical, max_stale_days=max_stale_days)
    return data


def _sina_daily_fn(scode: str):
    """Dispatch between fund and stock daily-history endpoints."""
    if is_fund_symbol(scode[2:]):
        return lambda *, symbol, adjust: ak.fund_etf_hist_sina(symbol=symbol)
    return lambda *, symbol, adjust: ak.stock_zh_a_daily(symbol=symbol, adjust=adjust)


def _normalize_frame(raw: pd.DataFrame, symbol: str, canonical: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise NoMarketDataError(
            symbol, canonical, f"sina returned no rows for {to_sina_symbol(canonical)}"
        )
    renames = {c: r for c, r in _DAILY_RENAMES.items() if c in raw.columns}
    missing = {"Date", "Open", "High", "Low", "Close"} - set(renames.values())
    if missing:
        raise NoMarketDataError(symbol, canonical, f"daily frame missing columns: {sorted(missing)}")
    data = raw[list(renames)].rename(columns=renames).copy()
    for col in ("Open", "High", "Low", "Close"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["Volume"] = pd.to_numeric(data.get("Volume"), errors="coerce")
    return data.reset_index(drop=True)


def get_stock_data_sina(
    symbol: Annotated[str, "A-share ticker, e.g. 600519.SS / 510300.SS"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """OHLCV window for an A-share/ETF from Sina (qfq)."""
    datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    canonical = normalize_symbol(symbol)
    data = fetch_daily_ohlcv_sina(symbol, canonical, end_date)
    data = data[(data["Date"] >= pd.Timestamp(start_date)) & (data["Date"] <= end_dt)]
    if data.empty:
        raise NoMarketDataError(symbol, canonical, f"no rows between {start_date} and {end_date}")
    data = data.reset_index(drop=True)
    for col in ("Open", "High", "Low", "Close"):
        data[col] = data[col].round(2)

    csv_string = data.to_csv(index=False)
    header = (
        f"# Stock data for {canonical} [A-share {to_sina_symbol(canonical)}] "
        f"from {start_date} to {end_date}\n"
        "# Source: Sina Finance daily, qfq-adjusted, volume in shares\n"
        f"# Total records: {len(data)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


def get_indicators_sina(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to compute"],
    curr_date: Annotated[str, "current trading date YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Indicator history identical in shape to the other vendor adapters."""
    from stockstats import wrap

    if indicator not in INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. "
            f"Please choose from: {list(INDICATOR_DESCRIPTIONS.keys())}"
        )

    canonical = normalize_symbol(symbol)
    data = fetch_daily_ohlcv_sina(symbol, canonical, curr_date)

    df = wrap(data.copy())
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df[indicator]

    lookup = {}
    for _, row in df.iterrows():
        value = row[indicator]
        lookup[row["Date"]] = "N/A" if pd.isna(value) else str(value)

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_dt - pd.Timedelta(days=int(look_back_days))
    ind_string = ""
    day = curr_dt
    while day >= before:
        key = day.strftime("%Y-%m-%d")
        ind_string += f"{key}: {lookup.get(key, 'N/A: Not a trading day (weekend or holiday)')}\n"
        day -= pd.Timedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + ind_string
        + "\n\n"
        + INDICATOR_DESCRIPTIONS.get(indicator, "No description available.")
    )


# ---------------------------------------------------------------------------
# Statements & fundamentals (Sina periodic reports, Chinese wide tables)
# ---------------------------------------------------------------------------

_STATEMENT_KINDS = {
    "balance_sheet": ("资产负债表", "Balance Sheet"),
    "cashflow": ("现金流量表", "Cash Flow Statement"),
    "income_statement": ("利润表", "Income Statement"),
}

_STATEMENT_FIELDS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    # EN label -> ordered CN substring candidates (first match wins)
    "income_statement": [
        ("Total Operating Revenue", ("营业总收入",)),
        ("Operating Cost", ("营业成本",)),
        ("Operating Profit", ("营业利润",)),
        ("Net Profit", ("净利润",)),
        ("Net Profit Attributable to Parent", ("归属于母公司所有者的净利润", "归属母公司")),
        ("Basic EPS", ("基本每股收益",)),
    ],
    "balance_sheet": [
        ("Total Assets", ("资产总计",)),
        ("Total Liabilities", ("负债合计",)),
        ("Cash & Equivalents", ("货币资金",)),
        ("Accounts Receivable", ("应收账款",)),
        ("Inventory", ("存货",)),
        ("Total Equity", ("所有者权益(或股东权益)合计", "股东权益合计", "所有者权益")),
    ],
    "cashflow": [
        ("Net Operating Cash Flow", ("经营活动产生的现金流量净额", "经营活动")),
        ("Net Investing Cash Flow", ("投资活动产生的现金流量净额", "投资活动")),
        ("Net Financing Cash Flow", ("筹资活动产生的现金流量净额", "筹资活动")),
        ("Net Change in Cash", ("现金及现金等价物净增加额",)),
    ],
}

_STATEMENT_PERIODS = 6


def _fetch_statement_raw(scode: str, cn_kind: str) -> pd.DataFrame:
    """Rows = reported periods; drop the annoying trailing summary rows."""
    df = _quiet(ak.stock_financial_report_sina, stock=scode, symbol=cn_kind)
    if df is None or df.empty:
        raise NoMarketDataError(scode, scode, f"sina {cn_kind} empty")
    if "报告日" not in df.columns:
        raise NoMarketDataError(scode, scode, f"sina {cn_kind} missing 报告日")
    return df


def _parse_period(value) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(str(value), errors="coerce")
    except Exception:
        return None


def get_global_news_sina(
    curr_date: Annotated[str, "current date YYYY-MM-DD"],
    look_back_days: Annotated[int, "days to look back"] = None,
    limit: Annotated[int, "max headlines"] = None,
) -> str:
    """全球财经快讯（新浪，免限流）——新闻链的东财兜底。"""
    from .yfinance_news import _in_news_window

    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")

    raw = _quiet(ak.stock_info_global_sina)
    if raw is None or raw.empty:
        return f"No global news found between {start_dt:%Y-%m-%d} and {curr_date}"

    news_str = ""
    kept = 0
    for _, row in raw.iterrows():
        pub = None
        with contextlib.suppress(ValueError, TypeError):
            pub = datetime.strptime(str(row.get("时间", ""))[:19], "%Y-%m-%d %H:%M:%S")
        if not _in_news_window(pub, start_dt, end_dt):
            continue
        content = str(row.get("内容", "") or "").strip()
        if not content:
            continue
        news_str += f"### {content.splitlines()[0][:80]} (source: 新浪财经)\n{content}\n\n"
        kept += 1
        if kept >= limit:
            break

    if kept == 0:
        return f"No global news found between {start_dt:%Y-%m-%d} and {curr_date}"
    header = (
        f"## Global Market News (Sina), from {start_dt:%Y-%m-%d} to {curr_date}:\n\n"
    )
    return header + news_str


def get_income_statement_sina(ticker, freq="quarterly", curr_date=None) -> str:
    return _render_statement(ticker, "income_statement", freq, curr_date)


def get_balance_sheet_sina(ticker, freq="quarterly", curr_date=None) -> str:
    return _render_statement(ticker, "balance_sheet", freq, curr_date)


def get_cashflow_sina(ticker, freq="quarterly", curr_date=None) -> str:
    return _render_statement(ticker, "cashflow", freq, curr_date)


def _render_statement(ticker: str, kind: str, freq: str, curr_date: str | None) -> str:
    canonical = normalize_symbol(ticker)
    acode = to_acode(ticker)
    if is_fund_symbol(acode):
        return _fund_not_applicable(kind.replace("_", " "))

    title_en = _STATEMENT_KINDS[kind][1]
    try:
        raw = _fetch_statement_raw(to_sina_symbol(ticker), _STATEMENT_KINDS[kind][0])
    except VendorNotConfiguredError:
        raise
    except NoMarketDataError:
        raise
    except Exception as exc:
        return f"Error retrieving {title_en.lower()} for {ticker} via sina: {exc}"

    raw["_period"] = raw["报告日"].map(_parse_period)
    cutoff = pd.Timestamp(curr_date) if curr_date else pd.Timestamp.max
    visible = raw[raw["_period"].notna() & (raw["_period"] <= cutoff)]
    if visible.empty:
        raise NoMarketDataError(ticker, canonical, f"no published periods ≤ {curr_date}")

    parsed = visible["_period"]
    if freq.lower() != "quarterly":
        visible = visible[parsed.dt.month == 12]
        if visible.empty:
            raise NoMarketDataError(ticker, canonical, "no annual (12月) reports")
    visible = visible.sort_values("_period").tail(_STATEMENT_PERIODS)

    blocks = []
    fields = _STATEMENT_FIELDS[kind]
    for _, row in visible.iterrows():
        period = row["_period"].date()
        body_lines = []
        upper_cols = {str(c).replace(" ", ""): c for c in raw.columns}
        for en_label, candidates in fields:
            found = None
            for cand in candidates:
                key = next((orig for stripped, orig in upper_cols.items()
                            if cand in stripped), None)
                if key is not None:
                    found = key
                    break
            if found is not None and pd.notna(row[found]):
                body_lines.append(f"- {en_label}: {row[found]}")
        body = "\n".join(body_lines) or "- (fields unavailable)"
        blocks.append(f"### Report period {period}\n{body}")

    header = (
        f"# {title_en} data for {canonical} ({freq}, reported)\n"
        "# Source: Sina Finance periodic reports, amounts in CNY\n"
        "# Note: sina exposes no disclosure date; periods are gated by "
        "report-end date\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + "\n\n".join(blocks)


_FUND_NOTE_TMPL = (
    "Not applicable: '{label}' is an exchange-traded fund (ETF/LOF), so "
    "corporate statements do not exist. Assess it through price action and the "
    "underlying index instead."
)


def _fund_not_applicable(label: str) -> str:
    return _FUND_NOTE_TMPL.format(label=label)


def get_fundamentals_sina(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """Best-effort fundamentals from Sina + Baidu: price, daily market-cap/PB,
    quarterly financial ratios (ROE, margins, debt, EPS, BVPS)."""
    canonical = normalize_symbol(ticker)
    acode = to_acode(ticker)
    if is_fund_symbol(acode):
        return _fund_not_applicable("fundamentals")

    try:
        data = fetch_daily_ohlcv_sina(ticker, canonical, curr_date or
                                      pd.Timestamp.today().strftime("%Y-%m-%d"))
    except VendorNotConfiguredError:
        raise
    except NoMarketDataError:
        raise
    except Exception as exc:
        return f"Error retrieving fundamentals for {ticker} via sina: {exc}"

    lines = [f"Code: {acode}"]
    if not data.empty:
        latest = data.iloc[-1]
        prev_close = float(data["Close"].iloc[-2]) if len(data) >= 2 else None
        lines.append(f"Latest Close ({latest['Date'].date()}): {latest['Close']}")
        if prev_close:
            chg = (float(latest["Close"]) - prev_close) / prev_close * 100
            lines.append(f"Latest Daily Change: {chg:+.2f}%")
        if len(data) >= 21:
            ma20 = float(data["Close"].tail(20).mean())
            lines.append(f"MA20: {ma20:.2f}")
    else:
        raise NoMarketDataError(ticker, canonical, "no daily rows")

    # Baidu daily valuation (total market cap + PB, non-EM host)
    try:
        mktcap = _quiet(ak.stock_zh_valuation_baidu, symbol=acode,
                       indicator="总市值", period="近一年")
        pb = _quiet(ak.stock_zh_valuation_baidu, symbol=acode,
                    indicator="市净率", period="近一年")
        if mktcap is not None and not mktcap.empty:
            mc = pd.to_numeric(mktcap["value"], errors="coerce").dropna()
            if len(mc):
                lines.append(f"Total Market Cap (亿): {mc.iloc[-1]:.0f}")
                if len(mc) >= 6:
                    lines.append(f"Market Cap 5d Change: "
                                 f"{(mc.iloc[-1]/mc.iloc[-6]-1)*100:+.1f}%")
        if pb is not None and not pb.empty:
            pbs = pd.to_numeric(pb["value"], errors="coerce").dropna()
            if len(pbs):
                lines.append(f"Price to Book (PB): {pbs.iloc[-1]:.2f}")
    except Exception as exc:
        logger.warning("baidu valuation unavailable: %s", exc)

    # Sina quarterly financial ratios (EPS, BVPS, ROE, margins, debt)
    try:
        yr = str((pd.Timestamp(curr_date) if curr_date
                  else pd.Timestamp.today()).year - 1)
        ratios = _quiet(ak.stock_financial_analysis_indicator,
                        symbol=acode, start_year=yr)
        if ratios is not None and not ratios.empty:
            r = ratios.iloc[-1]
            for cn, en in [("摊薄每股收益(元)", "EPS (diluted)"),
                           ("每股净资产_调整前(元)", "Book Value Per Share"),
                           ("净资产收益率(%)", "ROE"),
                           ("销售净利率(%)", "Net Profit Margin"),
                           ("资产负债率(%)", "Debt to Asset Ratio"),
                           ("净利润增长率(%)", "Net Profit Growth YoY"),
                           ("流动比率", "Current Ratio")]:
                v = r.get(cn)
                if v is not None and pd.notna(v) and v != 0:
                    lines.append(f"{en}: {v}")
            # PE estimate from latest price / diluted EPS
            eps = r.get("摊薄每股收益(元)")
            if eps is not None and pd.notna(eps) and eps > 0 and not data.empty:
                pe_est = float(data["Close"].iloc[-1]) / float(eps)
                lines.append(f"PE (estimated, price/EPS): {pe_est:.1f}")
    except Exception as exc:
        logger.warning("sina financial indicators unavailable: %s", exc)

    header = (
        f"# Company Fundamentals for {canonical} ({acode})\n"
        "# Source: Sina + Baidu (unthrottled hosts)\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + "\n".join(lines)


__all__ = [
    "ak",
    "fetch_daily_ohlcv_sina",
    "get_balance_sheet_sina",
    "get_cashflow_sina",
    "get_fundamentals_sina",
    "get_income_statement_sina",
    "get_indicators_sina",
    "get_stock_data_sina",
    "to_acode",
    "to_sina_symbol",
]
