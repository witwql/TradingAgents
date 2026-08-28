"""Global macro factor tools: gold, crude oil, US Treasury yields, US equities.

Powers the optional Global Macro Analyst with hard numbers instead of vibes.
All series are Sina/中债-sourced (the unthrottled host family), keyless, and
strictly look-ahead filtered to the analysis date.

The quantitative centerpiece is :func:`get_factor_exposure`, which regresses
the target's daily returns against each factor's PREVIOUS-day move (overnight
transmission into the A-share session), producing correlations, betas and a
composite overnight factor score the analyst can cite.
"""

import contextlib
import io
import logging
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

from .akshare_lock import AKSHARE_LOCK
from .errors import NoMarketDataError, VendorNotConfiguredError
from .stockstats_utils import load_ohlcv

logger = logging.getLogger(__name__)

try:
    import akshare as ak
except ImportError:
    ak = None


_QUIET_RETRIES = 3
_QUIET_BASE_DELAY = 2.0


def _quiet(fn, *args, **kwargs):
    """Same retry + V8-lock discipline as the other AKShare-backed vendors."""
    import time

    if ak is None:
        raise VendorNotConfiguredError(
            'akshare package is not installed: pip install "tradingagents[akshare]"'
        )
    fn_name = getattr(fn, "__name__", str(fn))
    last: Exception | None = None
    for attempt in range(_QUIET_RETRIES + 1):
        try:
            with AKSHARE_LOCK, contextlib.redirect_stderr(io.StringIO()):
                return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            if attempt < _QUIET_RETRIES:
                delay = _QUIET_BASE_DELAY * (2**attempt)
                logger.warning(
                    "global-macro %s transient failure (%d/%d): %s; retrying",
                    fn_name, attempt + 1, _QUIET_RETRIES, exc,
                )
                time.sleep(delay)
    assert last is not None
    raise last


def _window(df: pd.DataFrame, date_col: str, curr_date: str, lookback_days: int) -> pd.DataFrame:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    cutoff = pd.Timestamp(curr_date)
    mask = dates.notna() & (dates <= cutoff) & (dates >= cutoff - pd.Timedelta(days=lookback_days))
    return df[mask].copy()


def _pct_change(values: pd.Series) -> float | None:
    clean = values.dropna().astype(float)
    if len(clean) < 2 or clean.iloc[-2] == 0:
        return None
    return (clean.iloc[-1] - clean.iloc[-2]) / clean.iloc[-2] * 100


def _chg_over(values: pd.Series, days: int) -> float | None:
    clean = values.dropna().astype(float)
    if len(clean) <= days or clean.iloc[-1 - days] == 0:
        return None
    return (clean.iloc[-1] - clean.iloc[-1 - days]) / clean.iloc[-1 - days] * 100


def _fmt_pct(v: float | None) -> str:
    return f"{v:+.2f}%" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Individual factor tools
# ---------------------------------------------------------------------------

@tool
def get_gold_price(
    curr_date: Annotated[str, "analysis date YYYY-MM-DD"],
    lookback_days: Annotated[int, "calendar days of history to include"] = 30,
) -> str:
    """COMEX gold front-month (Sina 外盘 GC): latest close, 1d/5d/20d moves."""
    raw = _quiet(ak.futures_foreign_hist, symbol="GC")
    df = _window(raw, "date", curr_date, max(lookback_days, 25))
    if df.empty:
        raise NoMarketDataError("GC", "GC", f"no gold rows on or before {curr_date}")
    close = pd.to_numeric(df["close"], errors="coerce")
    latest = df.iloc[-1]
    return (
        f"# 国际金价 (COMEX 黄金主力 GC, Sina)\n"
        f"- 截至 {pd.to_datetime(latest['date']).date()}\n"
        f"- 最新收盘: {latest['close']}\n"
        f"- 日变动: {_fmt_pct(_pct_change(close))}\n"
        f"- 5日累计: {_fmt_pct(_chg_over(close, 5))}\n"
        f"- 20日累计: {_fmt_pct(_chg_over(close, 20))}\n"
    )


@tool
def get_crude_oil_price(
    curr_date: Annotated[str, "analysis date YYYY-MM-DD"],
    lookback_days: Annotated[int, "calendar days of history to include"] = 30,
) -> str:
    """NYMEX WTI front-month (Sina 外盘 CL): latest close, 1d/5d/20d moves."""
    raw = _quiet(ak.futures_foreign_hist, symbol="CL")
    df = _window(raw, "date", curr_date, max(lookback_days, 25))
    if df.empty:
        raise NoMarketDataError("CL", "CL", f"no crude rows on or before {curr_date}")
    close = pd.to_numeric(df["close"], errors="coerce")
    latest = df.iloc[-1]
    return (
        f"# 国际原油 (NYMEX WTI 主力 CL, Sina)\n"
        f"- 截至 {pd.to_datetime(latest['date']).date()}\n"
        f"- 最新收盘: {latest['close']}\n"
        f"- 日变动: {_fmt_pct(_pct_change(close))}\n"
        f"- 5日累计: {_fmt_pct(_chg_over(close, 5))}\n"
        f"- 20日累计: {_fmt_pct(_chg_over(close, 20))}\n"
    )


@tool
def get_us_treasury_yields(
    curr_date: Annotated[str, "analysis date YYYY-MM-DD"],
    lookback_days: Annotated[int, "calendar days of history to include"] = 60,
) -> str:
    """US Treasury yields (中债网 via AKShare): 2y/10y levels + bp changes.

    Yields move in basis points, so changes are reported in bp rather than %.
    """
    raw = _quiet(
        ak.bond_zh_us_rate,
        start_date=(pd.Timestamp(curr_date) - pd.Timedelta(days=max(lookback_days, 40))).strftime("%Y%m%d"),
    )
    df = _window(raw, "日期", curr_date, max(lookback_days, 40)).dropna(subset=["美国国债收益率10年"])
    if df.empty:
        raise NoMarketDataError("US10Y", "US10Y", f"no yield rows on or before {curr_date}")

    def level(col: str) -> float:
        return float(df[col].iloc[-1])

    def bp(col: str, days: int) -> str:
        clean = df[col].dropna().astype(float)
        if len(clean) <= days:
            return "N/A"
        return f"{(clean.iloc[-1] - clean.iloc[-1 - days]) * 100:+.1f}bp"

    return (
        f"# 美国国债收益率 (中债网)\n"
        f"- 截至 {pd.to_datetime(df['日期'].iloc[-1]).date()}\n"
        f"- 10年期: {level('美国国债收益率10年'):.2f}% (1日 {bp('美国国债收益率10年', 1)}, "
        f"5日 {bp('美国国债收益率10年', 5)}, 20日 {bp('美国国债收益率10年', 20)})\n"
        f"- 2年期: {level('美国国债收益率2年'):.2f}% (1日 {bp('美国国债收益率2年', 1)}, "
        f"5日 {bp('美国国债收益率2年', 5)})\n"
        f"- 10Y-2Y利差: {(level('美国国债收益率10年') - level('美国国债收益率2年')):+.2f}百分点\n"
    )


_US_INDICES = [(".INX", "标普500"), (".IXIC", "纳斯达克"), (".DJI", "道琼斯")]


@tool
def get_us_stock_indices(
    curr_date: Annotated[str, "analysis date YYYY-MM-DD"],
    lookback_days: Annotated[int, "calendar days of history to include"] = 30,
) -> str:
    """US equity indices overnight session (Sina): close + 1d/5d momentum."""
    sections = []
    as_of = None
    for symbol, name in _US_INDICES:
        try:
            raw = _quiet(ak.index_us_stock_sina, symbol=symbol)
        except Exception as exc:
            logger.warning("us index %s unavailable: %s", symbol, exc)
            continue
        df = _window(raw, "date", curr_date, max(lookback_days, 25))
        if df.empty:
            continue
        close = pd.to_numeric(df["close"], errors="coerce")
        as_of = as_of or pd.to_datetime(df["date"].iloc[-1]).date()
        sections.append(
            f"- {name}: {close.iloc[-1]:.2f} "
            f"(日 {_fmt_pct(_pct_change(close))}, 5日 {_fmt_pct(_chg_over(close, 5))})"
        )
    if not sections:
        raise NoMarketDataError(".INX", ".INX", f"no US index rows on or before {curr_date}")
    return (
        f"# 美股行情 (Sina, 隔夜已收盘时段)\n"
        f"- 截至 {as_of}\n" + "\n".join(sections) + "\n"
    )


# ---------------------------------------------------------------------------
# Money flow (主力资金) — EastMoney per-stock daily history
# ---------------------------------------------------------------------------

@tool
def get_money_flow(
    symbol: Annotated[str, "A-share ticker, e.g. 600519.SS"],
    curr_date: Annotated[str, "analysis date YYYY-MM-DD"],
    lookback_days: Annotated[int, "calendar days of flow history to summarize"] = 30,
) -> str:
    """主力资金 (main-capital = 超大单+大单) daily net-flow history and derived
    signals: latest day, 1d/5d/20d aggregates, consecutive-flow streaks, and a
    price-vs-flow divergence flag (价涨资金流出 = distribution risk)."""
    code = str(symbol).strip().upper().rstrip("+")
    from .symbol_utils import ashare_exchange

    bare = code.split(".")[0]
    market = (ashare_exchange(bare) or "SZ").lower()

    raw = _quiet(ak.stock_individual_fund_flow, stock=bare, market=market)
    if raw is None or raw.empty:
        raise NoMarketDataError(symbol, code, "no money-flow rows returned")

    raw["_d"] = pd.to_datetime(raw["日期"], errors="coerce")
    raw = raw[raw["_d"].notna() & (raw["_d"] <= pd.Timestamp(curr_date))]
    raw = raw[raw["_d"] >= pd.Timestamp(curr_date) - pd.Timedelta(days=max(lookback_days, 25))]
    if raw.empty:
        raise NoMarketDataError(symbol, code, f"no flow rows on or before {curr_date}")

    main = pd.to_numeric(raw["主力净流入-净占比"], errors="coerce")
    main_amt = pd.to_numeric(raw["主力净流入-净额"], errors="coerce")
    super_amt = pd.to_numeric(raw.get("超大单净流入-净额"), errors="coerce")
    close = pd.to_numeric(raw["收盘价"], errors="coerce")

    days = raw["_d"]

    # 连续净流入/流出天数（以主力净额符号计）
    streak = 0
    for v in reversed(main_amt.dropna().tolist()):
        if streak == 0:
            streak = 1 if v > 0 else -1
        elif (v > 0) == (streak > 0):
            streak += 1 if v > 0 else -1
        else:
            break
    streak_txt = f"连续净流入 {streak} 天" if streak > 0 else (
        f"连续净流出 {-streak} 天" if streak < 0 else "方向未明")

    def yi(v):
        return f"{v / 1e8:+.2f}亿" if pd.notna(v) else "N/A"

    # 量价背离：近3日累计上涨但主力累计净流出
    up3 = close.iloc[-1] - close.iloc[-4] if len(close) >= 4 else None
    flow3 = main_amt.iloc[-3:].sum()
    divergence = ""
    if up3 is not None and pd.notna(flow3):
        if up3 > 0 and flow3 < 0:
            divergence = "\n- ⚠ 量价背离：近3日价格上行但主力资金累计净流出（" + yi(flow3) + "）——拉升缺乏主力配合，警惕出货"
        elif up3 < 0 and flow3 > 0:
            divergence = "\n- ⚠ 逆向吸筹信号：近3日价格下行但主力资金累计净流入（" + yi(flow3) + "）——可能是打压吸筹"

    return (
        f"# 主力资金流 (EastMoney, 主力=超大单+大单)\n"
        f"- 截至 {days.iloc[-1].date()}\n"
        f"- 最新主力净额: {yi(main_amt.iloc[-1])}（净占比 {main.iloc[-1]:+.2f}%）\n"
        f"- 其中超大单净额: {yi(super_amt.iloc[-1])}\n"
        f"- 5日主力净额合计: {yi(main_amt.iloc[-5:].sum())}\n"
        f"- 20日主力净额合计: {yi(main_amt.iloc[-20:].sum())}\n"
        f"- 连续性: {streak_txt}\n"
        f"{divergence}\n"
        f"解读指引：净占比>2% 视为显著进攻；连续净流入+价升=健康趋势；"
        f"价升资金流出=背离风险；低位连续净流入+价格滞涨=潜在吸筹。"
    )


# ---------------------------------------------------------------------------
# Quantitative factor-exposure model
# ---------------------------------------------------------------------------

def _factor_returns(symbol: str, curr_date: str, lookback_days: int) -> dict[str, pd.Series]:
    """Daily % returns per factor, indexed by date, SHIFTED +1 day.

    The shift encodes the transmission timing the model assumes: an overseas
    session move on A-share day t-1 evening shows up as an influence on day t.
    """
    out: dict[str, pd.Series] = {}

    raw = _quiet(ak.futures_foreign_hist, symbol="GC")
    df = _window(raw, "date", curr_date, lookback_days * 2)
    s = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100
    out["GOLD"] = pd.Series(s.values, index=pd.to_datetime(df["date"])).shift(1)

    raw = _quiet(ak.futures_foreign_hist, symbol="CL")
    df = _window(raw, "date", curr_date, lookback_days * 2)
    s = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100
    out["CRUDE"] = pd.Series(s.values, index=pd.to_datetime(df["date"])).shift(1)

    raw = _quiet(ak.bond_zh_us_rate, start_date=(pd.Timestamp(curr_date) - pd.Timedelta(days=lookback_days * 2)).strftime("%Y%m%d"))
    df = _window(raw, "日期", curr_date, lookback_days * 2).dropna(subset=["美国国债收益率10年"])
    yields = pd.to_numeric(df["美国国债收益率10年"], errors="coerce")
    s = yields.diff()  # bp-scale day change
    out["US10Y"] = pd.Series(s.values, index=pd.to_datetime(df["日期"])).shift(1)

    raw = _quiet(ak.index_us_stock_sina, symbol=".INX")
    df = _window(raw, "date", curr_date, lookback_days * 2)
    s = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100
    out["SPX"] = pd.Series(s.values, index=pd.to_datetime(df["date"])).shift(1)

    # 主力资金净占比（pct-point 日变化）——EM 限流时优雅跳过
    try:
        from .symbol_utils import ashare_exchange

        bare = to_bare(symbol)
        raw = _quiet(
            ak.stock_individual_fund_flow, stock=bare,
            market=(ashare_exchange(bare) or "SZ").lower(),
        )
        raw["_d"] = pd.to_datetime(raw["日期"], errors="coerce")
        raw = raw[raw["_d"].notna()]
        s = pd.to_numeric(raw["主力净流入-净占比"], errors="coerce").diff()
        out["MFLOW"] = pd.Series(s.values, index=raw["_d"]).shift(1)
    except Exception as exc:
        logger.info("factor exposure: money flow skipped (%s)", exc)

    return out


def to_bare(symbol: str) -> str:
    """'600519.SS'/'600519' -> '600519'."""
    return str(symbol).strip().upper().rstrip("+").split(".")[0]


_FACTOR_LABELS = {
    "GOLD": "国际金价",
    "CRUDE": "国际原油",
    "US10Y": "美债10Y收益率",
    "SPX": "美股标普500",
    "MFLOW": "主力资金净占比",
}


@tool
def get_factor_exposure(
    symbol: Annotated[str, "target ticker, e.g. 600519.SS or 510300.SS"],
    curr_date: Annotated[str, "analysis date YYYY-MM-DD"],
    lookback_days: Annotated[int, "trading-day window for correlation/beta estimation"] = 120,
) -> str:
    """Per-factor correlation, beta and composite overnight score vs the target.

    Transmission assumption: factor move on day t-1 (the overseas session) maps
    to the A-share move on day t. |corr| < 0.15 is reported but flagged as
    noise-level; the composite score weights factors by correlation strength.
    """
    ohlcv = load_ohlcv(symbol, curr_date)
    target = ohlcv.set_index(pd.to_datetime(ohlcv["Date"]))["Close"].astype(float)
    target_ret = target.pct_change() * 100
    window_start = pd.Timestamp(curr_date) - pd.Timedelta(days=int(lookback_days * 1.6))
    target_ret = target_ret[target_ret.index >= window_start]

    rows = []
    composite = 0.0
    weight_sum = 0.0
    latest_moves: dict[str, float | None] = {}

    for key, fact in _factor_returns(symbol, curr_date, lookback_days).items():
        label = _FACTOR_LABELS[key]
        aligned = pd.concat([target_ret, fact], axis=1, join="inner").dropna()
        aligned.columns = ["target", "factor"]
        if len(aligned) < 30:
            rows.append((label, "N/A", "N/A", "样本不足", 0.0))
            continue
        corr = float(aligned["target"].corr(aligned["factor"]))
        std_t = float(aligned["target"].std())
        std_f = float(aligned["factor"].std())
        beta = corr * (std_t / std_f) if std_f else 0.0
        latest_move = float(aligned["factor"].iloc[-1]) if len(aligned) else None
        latest_moves[key] = latest_move
        weight = max(0.0, abs(corr)) - 0.15 if abs(corr) >= 0.15 else 0.0
        if weight:
            composite += weight * beta * (latest_move or 0.0)
            weight_sum += weight
        strength = "显著" if abs(corr) >= 0.3 else ("中等" if abs(corr) >= 0.15 else "噪声级")
        rows.append((label, f"{corr:+.2f}", f"{beta:+.4f}", strength, corr))

    lines = [
        f"# 因子暴露度模型 — {symbol} (截至 {curr_date}, 估计窗口≈{lookback_days}个交易日)",
        "",
        "传导假设：海外时段(T-1晚)因子变动 → A股当日(T)收益。|相关|≥0.15 才计入合成得分。",
        "",
        "| 因子 | 相关系数 | β(每1%因子变动→标的%) | 显著性 |",
        "|---|---:|---:|---|",
    ]
    for label, corr, beta, strength, _c in rows:
        lines.append(f"| {label} | {corr} | {beta} | {strength} |")

    if weight_sum:
        lines += [
            "",
            f"**隔夜因子综合得分: {composite / weight_sum:+.2f}%** "
            "(加权 β × 最新因子变动; 正值=因子端偏利多)",
        ]
        for key, move in latest_moves.items():
            if move is not None:
                lines.append(f"- {_FACTOR_LABELS[key]} 最新变动: {move:+.2f} → 贡献 "
                             f"{'偏多' if move * _beta_sign(rows, _FACTOR_LABELS[key]) > 0 else '偏空'}")
    else:
        lines += ["", "四个因子与该标的的历史相关性均为噪声级——宏观因子解释力弱，"
                      "请以标的自身动量与基本面为主。"]

    lines += [
        "",
        "解读指引：金价上行通常利多黄金/避险资产、压制风险偏好；油价上行抬升"
        "航空/物流成本、利多上游资源；美债收益率上行压制成长股估值并影响红利"
        "资产相对吸引力；美股隔夜表现主导A股开盘情绪与风险偏好。",
    ]
    return "\n".join(lines)


def _beta_sign(rows: list, label: str) -> int:
    for r in rows:
        if r[0] == label:
            try:
                return 1 if float(r[2]) >= 0 else -1
            except (TypeError, ValueError):
                return 0
    return 0


__all__ = [
    "get_crude_oil_price",
    "get_factor_exposure",
    "get_gold_price",
    "get_us_stock_indices",
    "get_us_treasury_yields",
]
