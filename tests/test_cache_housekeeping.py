"""Cache housekeeping tests: superseded-file pruning and the mtime janitor."""
import os
import time

import pytest

from tradingagents.dataflows.utils import (
    prune_stale_cache_files,
    prune_superseded_cache_files,
)


@pytest.mark.unit
class TestSupersededPruneTests:
    def test_keeps_current_deletes_siblings(self, tmp_path):
        keep = tmp_path / "600519.SS-Sina-data-2026-08-31.csv"
        keep.write_text("Date,Close\n")
        for name in ("600519.SS-Sina-data-2026-08-30.csv",
                     "600519.SS-Sina-data-2026-08-29.csv"):
            (tmp_path / name).write_text("Date,Close\n")

        removed = prune_superseded_cache_files(
            keep, str(tmp_path / "600519.SS-Sina-data-*.csv"))

        assert removed == 2
        assert keep.exists()
        assert list(tmp_path.glob("*.csv")) == [keep]

    def test_other_tickers_untouched(self, tmp_path):
        keep = tmp_path / "600519.SS-Sina-data-2026-08-31.csv"
        keep.write_text("x")
        other = tmp_path / "000001.SZ-Sina-data-2026-08-30.csv"
        other.write_text("x")

        prune_superseded_cache_files(
            keep, str(tmp_path / "600519.SS-Sina-data-*.csv"))

        assert other.exists()

    def test_missing_current_is_fine(self, tmp_path):
        # prune keyed on a not-yet-written path (e.g. write failed) removes
        # siblings but must not blow up
        (tmp_path / "a-Sina-data-1.csv").write_text("x")
        removed = prune_superseded_cache_files(
            tmp_path / "a-Sina-data-2.csv", str(tmp_path / "a-Sina-data-*.csv"))
        assert removed == 1


@pytest.mark.unit
class TestWindowedCollapseTests:
    def test_collapses_to_newest_per_ticker(self, tmp_path):
        for day in ("2026-08-29", "2026-08-30", "2026-08-31"):
            (tmp_path / f"600519.SS-Sina-data-2021-08-31-{day}.csv").write_text("x")

        removed = prune_stale_cache_files(tmp_path)

        assert removed == 2
        remaining = list(tmp_path.glob("*.csv"))
        assert len(remaining) == 1
        assert "2026-08-31" in remaining[0].name

    def test_tickers_collapse_independently(self, tmp_path):
        for t in ("600519.SS", "000001.SZ"):
            (tmp_path / f"{t}-Sina-data-2021-x-2026-08-30.csv").write_text("x")
            (tmp_path / f"{t}-Sina-data-2021-x-2026-08-31.csv").write_text("x")

        removed = prune_stale_cache_files(tmp_path)

        assert removed == 2
        assert len(list(tmp_path.glob("*-Sina-data-*.csv"))) == 2

    def test_statement_kinds_collapse_separately(self, tmp_path):
        for day in ("2026-08-30", "2026-08-31"):
            (tmp_path / f"sh600519-Sina-stmt-利润表-{day}.csv").write_text("x")
            (tmp_path / f"sh600519-Sina-stmt-资产负债表-{day}.csv").write_text("x")

        removed = prune_stale_cache_files(tmp_path)

        assert removed == 2   # newest of each kind kept
        kept = sorted(p.name for p in tmp_path.glob("*.csv"))
        assert all("2026-08-31" in n for n in kept) and len(kept) == 2

    def test_akshare_family_collapsed_too(self, tmp_path):
        (tmp_path / "600519.SS-AkShare-data-2021-x-2026-08-30.csv").write_text("x")
        (tmp_path / "600519.SS-AkShare-data-2021-x-2026-08-31.csv").write_text("x")

        assert prune_stale_cache_files(tmp_path) == 1

    def test_unknown_family_not_collapsed(self, tmp_path):
        # a family that never opted in must not be collapsed by accident
        (tmp_path / "600519.SS-Mystery-2026-08-30.csv").write_text("x")
        (tmp_path / "600519.SS-Mystery-2026-08-31.csv").write_text("x")

        assert prune_stale_cache_files(tmp_path) == 0


@pytest.mark.unit
class TestStaleCacheJanitorTests:
    def test_old_files_removed_fresh_kept(self, tmp_path):
        old = tmp_path / "000001.SZ-Flow-2026-07-01.csv"
        old.write_text("x")
        past = time.time() - 30 * 86400
        os.utime(old, (past, past))
        fresh = tmp_path / "000002.SZ-Flow-2026-08-31.csv"
        fresh.write_text("x")

        removed = prune_stale_cache_files(tmp_path, max_age_days=14)

        assert removed == 1
        assert not old.exists()
        assert fresh.exists()

    def test_directories_untouched(self, tmp_path):
        subdir = tmp_path / "checkpoints"
        subdir.mkdir()
        stale_inside = subdir / "old.db"
        stale_inside.write_text("x")
        past = time.time() - 60 * 86400
        os.utime(stale_inside, (past, past))

        removed = prune_stale_cache_files(tmp_path, max_age_days=14)

        assert removed == 0          # janitor never descends into subdirs
        assert stale_inside.exists()

    def test_missing_dir_is_noop(self, tmp_path):
        assert prune_stale_cache_files(tmp_path / "nope") == 0
