import contextlib
import glob
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd

logger = logging.getLogger(__name__)

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]

# Janitor retention for data_cache_dir. Cache files are re-fetched on demand,
# so pruning only costs a network round-trip on the next use; 14 days covers
# everything touched by a weekly workflow while bounding disk usage.
CACHE_RETENTION_DAYS = 14

# Tickers can contain letters, digits, dot, dash, underscore, caret
# (index symbols like ^GSPC), equals (futures like GC=F), and plus
# (forex/CFD symbols like XAUUSD+). None of these enable directory
# traversal, so the value never escapes a containing directory when
# interpolated into a path. Anything else is rejected.
_TICKER_PATH_RE = re.compile(r"^[A-Za-z0-9._\-\^=+]+$")


def safe_ticker_component(value: str, *, max_len: int = 32) -> str:
    """Validate ``value`` is safe to interpolate into a filesystem path.

    Tickers come from user CLI input or from LLM tool calls, both of which
    can be influenced by attacker-controlled content (e.g. prompt injection
    embedded in fetched news). Without validation, a value like
    ``"../../../etc/foo"`` flows into ``os.path.join`` / ``Path /`` and
    escapes the configured cache, checkpoint, or results directory.

    Returns ``value`` unchanged when it matches the allowed pattern; raises
    ``ValueError`` otherwise.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"ticker must be a non-empty string, got {value!r}")
    if len(value) > max_len:
        raise ValueError(f"ticker exceeds {max_len} chars: {value!r}")
    if not _TICKER_PATH_RE.fullmatch(value):
        raise ValueError(
            f"ticker contains characters not allowed in a filesystem path: {value!r}"
        )
    # The regex above allows '.', so values like '.', '..', '...' would pass,
    # and as a path component they traverse the parent directory. Reject any
    # value that's only dots.
    if set(value) == {"."}:
        raise ValueError(f"ticker cannot consist solely of dots: {value!r}")
    return value


def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        data.to_csv(save_path, encoding="utf-8")
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date):

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date


# ---------------------------------------------------------------------------
# Cache housekeeping
# ---------------------------------------------------------------------------

# OHLCV/statement caches key their filename on "today", so every day would
# otherwise mint a fresh multi-MB file while superseded ones linger forever
# (~250 stocks/day of screening adds tens of GB a year). Write-time pruning
# keeps the hot set at one file per (ticker, vendor); the mtime janitor at
# server startup reclaims files for tickers no longer being analyzed.


def atomic_csv_write(df: "pd.DataFrame", path: str | Path, **to_csv_kwargs) -> None:
    """Write a cache CSV atomically: temp file in the same dir, then rename.

    Parallel task workers (and the screener threads) can read a cache file
    while another writer is mid-``to_csv`` — a torn read would surface as a
    truncated frame and spurious staleness failures. ``os.replace`` is atomic
    on the same filesystem, so readers see either the old or the new file.
    """
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        df.to_csv(tmp, **to_csv_kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def prune_superseded_cache_files(current_path: str | Path, pattern: str) -> int:
    """Delete cache files matching ``pattern`` except ``current_path``.

    Called right after writing a fresh cache file so the newest wins and its
    older siblings (yesterday's windows, last quarter's statement dump) go
    away. Never raises: a failed unlink only means one stale file lingers.
    Returns the number of files removed.
    """
    removed = 0
    keep = str(current_path)
    try:
        for candidate in glob.glob(pattern):
            if os.path.abspath(candidate) == os.path.abspath(keep):
                continue
            try:
                os.unlink(candidate)
                removed += 1
            except OSError:
                logger.debug("cache prune skipped %s", candidate, exc_info=True)
    except Exception:
        logger.debug("cache prune failed for pattern %s", pattern, exc_info=True)
    return removed


# Filename families where only the newest member per (ticker [, kind]) is
# ever read again: the name embeds the fetch day, so every run mints another
# file and superseded siblings are dead weight. Explicit list — a novel cache
# family must opt in, it is never collapsed by accident.
_COLLAPSE_FAMILIES = (
    ("-Sina-data-", False),
    ("-AkShare-data-", False),
    ("-Sina-stmt-", True),   # group also by statement kind (利润表/资产负债表/…)
)

_TRAILING_DATE_RE = re.compile(r"-\d{4}-\d{2}-\d{2}\.csv$")


def _collapse_windowed_cache_files(cache_dir: str) -> int:
    """Keep only the newest file per (ticker [, statement kind]) family."""
    groups: dict[str, list[str]] = {}
    for marker, kind_aware in _COLLAPSE_FAMILIES:
        for path in glob.glob(os.path.join(cache_dir, f"*{marker}*.csv")):
            base = os.path.basename(path)
            head, _, tail = base.partition(marker)
            if kind_aware:
                head = f"{head}|{_TRAILING_DATE_RE.sub('', tail)}"
            groups.setdefault(head, []).append(path)

    removed = 0
    for paths in groups.values():
        if len(paths) <= 1:
            continue
        newest = max(paths, key=os.path.getmtime)
        for path in paths:
            if path == newest:
                continue
            try:
                os.unlink(path)
                removed += 1
            except OSError:
                logger.debug("cache collapse skipped %s", path, exc_info=True)
    return removed


def prune_stale_cache_files(cache_dir: str | Path | None = None,
                            max_age_days: int = CACHE_RETENTION_DAYS) -> int:
    """Janitor: collapse per-day cache families, then drop stale leftovers.

    Two passes:
    1. Collapse windowed families (`*-Sina-data-*`, `*-AkShare-data-*`,
       `*-Sina-stmt-*`): these embed the fetch day in the filename, so only
       the newest file per (ticker [, statement kind]) is ever read again.
       This reclaims pre-existing backlogs the moment the server starts.
    2. Unlink any remaining cache file untouched for ``max_age_days`` — the
       catch-all for files whose ticker stopped being analyzed (flow caches,
       future families).

    Directory-safe (only regular files are removed) and never raises — the
    dashboard calls this at startup, where a cache hiccup must not block
    boot. Returns the number of files removed.
    """
    if cache_dir is None:
        from .config import get_config

        cache_dir = get_config()["data_cache_dir"]
    removed = _collapse_windowed_cache_files(str(cache_dir))
    cutoff = time.time() - max_age_days * 86400
    try:
        entries = os.scandir(cache_dir)
    except OSError:
        return removed
    with entries:
        for entry in entries:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if entry.stat().st_mtime >= cutoff:
                    continue
                os.unlink(entry.path)
                removed += 1
            except OSError:
                logger.debug("cache janitor skipped %s", entry.path, exc_info=True)
    return removed
