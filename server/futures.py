"""Sina-sourced futures board: realtime quotes + daily-close trend series.

Realtime quotes come from one batched ``hq.sinajs.cn`` call — the same
Referer-gated Sina host the watchlist spot quotes already rely on (it stayed
unthrottled under heavy usage while other providers dropped connections).
Daily closes for the trend sparklines come from Sina's futures kline
endpoints, refreshed by a background daemon thread: daily bars change at most
once per session, so a ~6h TTL keeps the payload fresh without hammering.

Change/percent follow the Chinese futures convention of measuring against the
previous settlement price; sina's CFFEX realtime payload carries no
settlement field, so those rows use the previous close instead (verified
against the daily kline).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_SINA_HQ_HEADERS = {"Referer": "https://finance.sina.com.cn"}
_KLINE_URL = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php"
    "/var%20_/InnerFuturesNewService.getDailyKLine?symbol={symbol}"
)
_GLOBAL_KLINE_URL = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php"
    "/var%20_/GlobalFuturesService.getGlobalFuturesDailyKLine?symbol={symbol}"
)

_TREND_POINTS = 40  # sparkline window, in trading days

CATEGORY_LABELS = {"commodity": "商品", "financial": "金融", "global": "国际"}

# Curated liquid contracts. (sina_symbol, name, category, exchange, unit)
# sina_symbol is the hq.sinajs.cn ticker without the nf_/hf_ prefix; the
# "0" suffix means the exchange's main continuous contract.
FUTURES_UNIVERSE: list[tuple[str, str, str, str, str]] = [
    # ---- 上期所 ----
    ("AU0", "沪金", "commodity", "上期所", "元/克"),
    ("AG0", "沪银", "commodity", "上期所", "元/千克"),
    ("CU0", "沪铜", "commodity", "上期所", "元/吨"),
    ("AL0", "沪铝", "commodity", "上期所", "元/吨"),
    ("ZN0", "沪锌", "commodity", "上期所", "元/吨"),
    ("PB0", "沪铅", "commodity", "上期所", "元/吨"),
    ("NI0", "沪镍", "commodity", "上期所", "元/吨"),
    ("SN0", "沪锡", "commodity", "上期所", "元/吨"),
    ("SS0", "不锈钢", "commodity", "上期所", "元/吨"),
    ("AO0", "氧化铝", "commodity", "上期所", "元/吨"),
    ("RB0", "螺纹钢", "commodity", "上期所", "元/吨"),
    ("HC0", "热卷", "commodity", "上期所", "元/吨"),
    ("RU0", "天然橡胶", "commodity", "上期所", "元/吨"),
    ("BR0", "丁二烯橡胶", "commodity", "上期所", "元/吨"),
    ("FU0", "燃料油", "commodity", "上期所", "元/吨"),
    ("SP0", "纸浆", "commodity", "上期所", "元/吨"),
    # ---- 能源中心 ----
    ("SC0", "原油", "commodity", "上期能源", "元/桶"),
    ("LU0", "低硫燃料油", "commodity", "上期能源", "元/吨"),
    ("BC0", "国际铜", "commodity", "上期能源", "元/吨"),
    ("EC0", "集运指数(欧线)", "commodity", "上期能源", "点"),
    # ---- 大商所 ----
    ("I0", "铁矿石", "commodity", "大商所", "元/吨"),
    ("J0", "焦炭", "commodity", "大商所", "元/吨"),
    ("JM0", "焦煤", "commodity", "大商所", "元/吨"),
    ("M0", "豆粕", "commodity", "大商所", "元/吨"),
    ("Y0", "豆油", "commodity", "大商所", "元/吨"),
    ("C0", "玉米", "commodity", "大商所", "元/吨"),
    ("P0", "棕榈油", "commodity", "大商所", "元/吨"),
    ("L0", "塑料", "commodity", "大商所", "元/吨"),
    ("V0", "PVC", "commodity", "大商所", "元/吨"),
    ("PP0", "聚丙烯", "commodity", "大商所", "元/吨"),
    ("EG0", "乙二醇", "commodity", "大商所", "元/吨"),
    ("EB0", "苯乙烯", "commodity", "大商所", "元/吨"),
    ("PG0", "液化石油气", "commodity", "大商所", "元/吨"),
    ("JD0", "鸡蛋", "commodity", "大商所", "元/500千克"),
    ("LH0", "生猪", "commodity", "大商所", "元/吨"),
    # ---- 郑商所 ----
    ("TA0", "PTA", "commodity", "郑商所", "元/吨"),
    ("MA0", "甲醇", "commodity", "郑商所", "元/吨"),
    ("FG0", "玻璃", "commodity", "郑商所", "元/吨"),
    ("SA0", "纯碱", "commodity", "郑商所", "元/吨"),
    ("UR0", "尿素", "commodity", "郑商所", "元/吨"),
    ("PF0", "短纤", "commodity", "郑商所", "元/吨"),
    ("CF0", "棉花", "commodity", "郑商所", "元/吨"),
    ("AP0", "苹果", "commodity", "郑商所", "元/吨"),
    ("CJ0", "红枣", "commodity", "郑商所", "元/吨"),
    ("PK0", "花生", "commodity", "郑商所", "元/吨"),
    ("RM0", "菜粕", "commodity", "郑商所", "元/吨"),
    ("SM0", "锰硅", "commodity", "郑商所", "元/吨"),
    ("SF0", "硅铁", "commodity", "郑商所", "元/吨"),
    ("PS0", "多晶硅", "commodity", "郑商所", "元/吨"),
    # ---- 中金所 ----
    ("IF0", "沪深300", "financial", "中金所", "点"),
    ("IH0", "上证50", "financial", "中金所", "点"),
    ("IC0", "中证500", "financial", "中金所", "点"),
    ("IM0", "中证1000", "financial", "中金所", "点"),
    ("T0", "十债主力", "financial", "中金所", "元"),
    ("TF0", "五债主力", "financial", "中金所", "元"),
    ("TS0", "二债主力", "financial", "中金所", "元"),
    ("TL0", "三十债主力", "financial", "中金所", "元"),
    # ---- 外盘 ----
    ("CL", "NYMEX原油", "global", "NYMEX", "美元/桶"),
    ("OIL", "布伦特原油", "global", "ICE", "美元/桶"),
    ("NG", "NYMEX天然气", "global", "NYMEX", "美元/百万英热"),
    ("GC", "COMEX黄金", "global", "COMEX", "美元/盎司"),
    ("SI", "COMEX白银", "global", "COMEX", "美元/盎司"),
    ("HG", "COMEX铜", "global", "COMEX", "美分/磅"),
    ("S", "CBOT大豆", "global", "CBOT", "美分/蒲式耳"),
    ("W", "CBOT小麦", "global", "CBOT", "美分/蒲式耳"),
    ("C", "CBOT玉米", "global", "CBOT", "美分/蒲式耳"),
    ("BO", "CBOT豆油", "global", "CBOT", "美分/磅"),
    ("LHC", "CME瘦肉猪", "global", "CME", "美分/磅"),
]

_UNIVERSE_MAP = {sym: (name, cat, exch, unit)
                 for sym, name, cat, exch, unit in FUTURES_UNIVERSE}

_QUOTE_TTL = 20        # realtime quotes: one batched call, cheap to refresh
_TREND_TTL = 6 * 3600  # daily bars: at most one new bar per session
_TREND_WORKERS = 8
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CLOCK_RE = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")


def _num(text: str | None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_commodity(fields: list[str]) -> dict:
    """nf_ 商品期货：名称,时间,开,高,低,昨收,买,卖,最新,当日结算,昨结算,买量,卖量,持仓,成交量,…,日期"""
    price = _num(fields[8])
    prev_settle = _num(fields[10]) if len(fields) > 10 else None
    if not prev_settle:  # 空串/0：结算价缺失，退回昨收(5)
        prev_settle = _num(fields[5])
    return {
        "price": price,
        "prev_settle": prev_settle,
        "open": _num(fields[2]),
        "high": _num(fields[3]),
        "low": _num(fields[4]),
        "open_interest": _num(fields[13]) if len(fields) > 13 else None,
        "volume": _num(fields[14]) if len(fields) > 14 else None,
        "date": fields[17] if len(fields) > 17 else "",
        "time": _fmt_clock(fields[1]) if len(fields) > 1 else "",
    }


def _parse_cffex(fields: list[str]) -> dict:
    """nf_ 中金所：开,高,低,最新,成交量,成交额,持仓,…,昨收(13),…,日期,时间,…,名称(末位)"""
    date = next((f for f in fields if _DATE_RE.match(f)), "")
    clock = next((f for f in fields if _CLOCK_RE.match(f)), "")
    return {
        "price": _num(fields[3]),
        "prev_settle": _num(fields[13]),  # 中金所实时报文无昨结算，以昨收为基准
        "open": _num(fields[0]),
        "high": _num(fields[1]),
        "low": _num(fields[2]),
        "open_interest": _num(fields[6]),
        "volume": _num(fields[4]),
        "date": date,
        "time": clock,
    }


def _parse_global(fields: list[str]) -> dict:
    """hf_ 外盘：最新,?,买,卖,高,低,时间,昨结算,开盘,?,?,?,日期,中文名,…"""
    return {
        "price": _num(fields[0]),
        "prev_settle": _num(fields[7]),
        "open": _num(fields[8]) if len(fields) > 8 else None,
        "high": _num(fields[4]) if len(fields) > 4 else None,
        "low": _num(fields[5]) if len(fields) > 5 else None,
        "open_interest": None,  # hf_ 报文持仓字段位置不统一，宁缺毋滥
        "volume": None,
        "date": fields[12] if len(fields) > 12 else "",
        "time": fields[6] if len(fields) > 6 else "",
    }


def _fmt_clock(raw: str) -> str:
    return f"{raw[:2]}:{raw[2:4]}:{raw[4:6]}" if len(raw) == 6 else raw


def parse_quotes_payload(text: str) -> dict[str, dict]:
    """Parse a hq.sinajs.cn response body into per-symbol quote dicts."""
    quotes: dict[str, dict] = {}
    for line in text.strip().splitlines():
        try:
            key, payload = line.split("=", 1)
            symbol = key.removeprefix("var hq_str_").strip()
            fields = payload.strip().strip('"').split(",")
            # 停牌/不存在的品种返回空串报文
            if fields[0] == "" or not symbol:
                continue
            if symbol.startswith("nf_"):
                raw = symbol[3:]
                # 商品报文首字段是中文名；中金所首字段直接是数字
                quote = (_parse_cffex(fields) if _num(fields[0]) is not None
                         else _parse_commodity(fields))
            elif symbol.startswith("hf_"):
                raw, quote = symbol[3:], _parse_global(fields)
            else:
                continue
            if not raw:
                continue
            quotes[raw] = quote
        except Exception:
            logger.debug("sina futures quote line unparsed: %r", line, exc_info=True)
    return quotes


def fetch_futures_quotes(symbols: list[str]) -> dict[str, dict]:
    """One batched realtime call for all board symbols (GB18030, Referer-gated)."""
    import requests

    if not symbols:
        return {}
    listing = ",".join(
        ("hf_" if _UNIVERSE_MAP[sym][1] == "global" else "nf_") + sym
        for sym in symbols
    )
    resp = requests.get(
        f"https://hq.sinajs.cn/list={listing}",
        headers=_SINA_HQ_HEADERS,
        timeout=8,
    )
    resp.raise_for_status()
    return parse_quotes_payload(resp.content.decode("gb18030", errors="replace"))


def fetch_futures_daily_closes(symbol: str, category: str) -> list[float]:
    """Last N daily closes for the trend sparkline (best-effort, may be [])."""
    import requests

    url = (_GLOBAL_KLINE_URL if category == "global" else _KLINE_URL).format(symbol=symbol)
    resp = requests.get(url, headers=_SINA_HQ_HEADERS, timeout=10)
    resp.raise_for_status()
    start, end = resp.text.find("["), resp.text.rfind("]")
    if start < 0 or end <= start:
        return []
    bars = json.loads(resp.text[start:end + 1])
    closes = []
    for bar in bars:
        try:
            close = float(bar.get("c") or bar.get("close") or 0)
        except (TypeError, ValueError):
            continue
        if close > 0:
            closes.append(close)
    return closes[-_TREND_POINTS:]


class FuturesBoard:
    """Quotes fetched on demand (one batched call, TTL-gated), trends by a
    background thread. Reads are lock-protected snapshots; failures keep
    serving the previous data (stale-while-revalidate)."""

    def __init__(self, quote_fetcher=fetch_futures_quotes,
                 trend_fetcher=fetch_futures_daily_closes,
                 quote_ttl: int = _QUOTE_TTL, trend_ttl: int = _TREND_TTL):
        self._fetch_quotes = quote_fetcher
        self._fetch_trend = trend_fetcher
        self._quote_ttl = max(5, int(quote_ttl))
        self._trend_ttl = max(60, int(trend_ttl))
        self._quotes: dict[str, dict] = {}
        self._quotes_ts = 0.0
        self._quotes_ready = False
        self._trends: dict[str, list[float]] = {}
        self._trends_ts = 0.0
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._trend_loop, name="futures-trends", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _trend_loop(self):
        while not self._stop.is_set():
            loaded = self._refresh_trends()
            # 首轮全失败（断网/服务未就绪）时快速重试，成功后按 TTL 节奏走
            self._stop.wait(30 if loaded == 0 else self._trend_ttl)

    def _refresh_trends(self) -> int:
        with self._lock:
            stale = time.time() - self._trends_ts >= self._trend_ttl
        targets = (list(_UNIVERSE_MAP) if stale
                   else [s for s in _UNIVERSE_MAP if s not in self._trends])
        if not targets:
            return len(self._trends)
        fetched = 0
        with ThreadPoolExecutor(max_workers=_TREND_WORKERS) as pool:
            futures = {sym: pool.submit(self._fetch_trend, sym, _UNIVERSE_MAP[sym][1])
                       for sym in targets}
            for sym, fut in futures.items():
                try:
                    closes = fut.result()
                except Exception:
                    continue
                if closes:
                    self._trends[sym] = closes
                    fetched += 1
        with self._lock:
            if fetched:
                self._trends_ts = time.time()
        if fetched < len(targets):
            logger.info("futures trends: %d/%d symbols fetched", fetched, len(targets))
        return fetched

    # ---------------------------------------------------------------- reads
    def _ensure_quotes(self):
        """Refresh the batched quote payload if stale; single-flight guarded."""
        with self._lock:
            fresh = time.time() - self._quotes_ts < self._quote_ttl
        if fresh or not self._refresh_lock.acquire(blocking=False):
            return
        try:
            rows = self._fetch_quotes(list(_UNIVERSE_MAP))
            if rows:
                with self._lock:
                    self._quotes = rows
                    self._quotes_ts = time.time()
                    self._quotes_ready = True
            elif not self._quotes_ready:
                logger.warning("futures quotes: empty payload (board offline?)")
        except Exception as exc:
            logger.warning("futures quotes refresh failed: %s", exc)
        finally:
            self._refresh_lock.release()

    def snapshot(self) -> dict:
        """Board state: contracts in curated order + cache metadata."""
        self._ensure_quotes()
        with self._lock:
            quotes = dict(self._quotes)
            trends = {k: list(v) for k, v in self._trends.items()}
            quotes_ts, quotes_ready = self._quotes_ts, self._quotes_ready
            trends_ts = self._trends_ts
        contracts = []
        for sym, (name, cat, exch, unit) in _UNIVERSE_MAP.items():
            quote = quotes.get(sym, {})
            price = quote.get("price")
            baseline = quote.get("prev_settle")
            change = pct = None
            if price is not None and baseline:
                change = price - baseline
                pct = change / baseline * 100
            contracts.append({
                "symbol": sym,
                "name": name,
                "category": cat,
                "category_label": CATEGORY_LABELS[cat],
                "exchange": exch,
                "unit": unit,
                "price": price,
                "prev_settle": baseline,
                "change": change,
                "pct": pct,
                **{k: quote.get(k) for k in ("open", "high", "low", "volume",
                                             "open_interest", "date", "time")},
                "trend": trends.get(sym, []),
            })
        return {
            "contracts": contracts,
            "quotes_ready": quotes_ready,
            "quote_ts": quotes_ts,
            "refresh_seconds": self._quote_ttl,
            "trend_ready": bool(trends),
            "trend_ts": trends_ts,
            "trend_count": len(trends),
            "trend_points": _TREND_POINTS,
        }
