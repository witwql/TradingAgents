"""AKShare news vendor: per-stock and global/macro Chinese financial news.

    framework method            AKShare endpoint
    ------------------------    ------------------------------------------
    get_news                    stock_news_em        (EastMoney stock news)
    get_global_news             stock_info_global_em (EastMoney global flash)

The same look-ahead-safe window semantics as the yfinance vendor apply:
articles outside ``[start_date, end_date + 1 day)`` are dropped so backtests
never see future news.
"""

import contextlib
import logging
from datetime import datetime, timedelta

import pandas as pd

from .akshare_stock import _quiet, ak, to_acode
from .config import get_config
from .yfinance_news import _in_news_window

logger = logging.getLogger(__name__)


def _parse_cn_timestamp(raw) -> datetime | None:
    """Parse EastMoney ``YYYY-MM-DD HH:MM:SS`` stamps; None when absent/bad."""
    if raw is None:
        return None
    with contextlib.suppress(ValueError, TypeError):
        return datetime.strptime(str(raw)[:19], "%Y-%m-%d %H:%M:%S")
    with contextlib.suppress(ValueError, TypeError):
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d")
    return None


def get_news_akshare(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Recent A-share news headlines for one ticker from EastMoney via AKShare."""
    config = get_config()
    article_limit = config["news_article_limit"]
    acode = to_acode(ticker)

    try:
        raw = _quiet(ak.stock_news_em, symbol=acode)
    except Exception as exc:
        return f"Error fetching news for {ticker}: {exc}"

    if raw is None or raw.empty:
        return f"No news found for {acode}"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    news_str = ""
    kept = 0
    for _, row in raw.iterrows():
        pub_date = _parse_cn_timestamp(row.get("发布时间"))

        # Look-ahead-safe window filter shared with the yfinance vendor.
        if not _in_news_window(pub_date, start_dt, end_dt):
            continue

        title = str(row.get("新闻标题", "No title"))
        source = str(row.get("文章来源", "Unknown"))
        content = str(row.get("新闻内容", "") or "").strip()
        link = str(row.get("新闻链接", "") or "")

        news_str += f"### {title} (source: {source})\n"
        if content:
            news_str += f"{content}\n"
        if link:
            news_str += f"Link: {link}\n"
        news_str += "\n"
        kept += 1
        if kept >= article_limit:
            break

    if kept == 0:
        return f"No news found for {acode} between {start_date} and {end_date}"

    header = f"## {acode} News (AKShare EastMoney), from {start_date} to {end_date}:\n\n"
    return header + news_str


def get_global_news_akshare(
    curr_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    """China & global macro market flash headlines from EastMoney via AKShare.

    Headlines publish in Chinese with timestamps and are filtered to the same
    look-ahead-safe window as every other news vendor.
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)

    try:
        raw = _quiet(ak.stock_info_global_em)
    except Exception as exc:
        logger.warning("AKShare global news unavailable: %s", exc)
        raw = pd.DataFrame()

    news_str = ""
    kept = 0
    if raw is not None and not raw.empty:
        for _, row in raw.iterrows():
            pub_date = _parse_cn_timestamp(row.get("发布时间"))

            if not _in_news_window(pub_date, start_dt, curr_dt):
                continue

            title = str(row.get("标题", "")).strip()
            summary = str(row.get("摘要", "") or "").strip()
            link = str(row.get("链接", "") or "")

            if not title:
                continue

            news_str += f"### {title} (source: EastMoney 快讯)\n"
            if summary:
                news_str += f"{summary}\n"
            if link:
                news_str += f"Link: {link}\n"
            news_str += "\n"
            kept += 1
            if kept >= limit:
                break

    if kept == 0:
        return f"No global news found between {start_dt.strftime('%Y-%m-%d')} and {curr_date}"

    header = (
        f"## Global Market News (AKShare EastMoney), "
        f"from {start_dt.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
    )
    return header + news_str


__all__ = ["get_global_news_akshare", "get_news_akshare"]
