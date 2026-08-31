"""选股复盘 tests: settlement math, lazy fetch behavior, DB upserts, API."""
import json
import time
from unittest import mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.db import Database
from server.review import realized_returns, review_summary, settle_run


def _frame(closes, start="2026-08-10"):
    """Daily closes starting on a Monday, business-day spaced."""
    dates = pd.bdate_range(start=start, periods=len(closes))
    return pd.DataFrame({
        "Date": dates, "Open": closes, "High": closes,
        "Low": closes, "Close": [float(c) for c in closes],
        "Volume": [1e5] * len(closes),
    })


@pytest.mark.unit
class TestRealizedReturnsTests:
    def test_basic_t1_t5(self):
        # baseline 10 (idx0); T+1 = idx1, T+5 = idx5 → 11.8/10 - 1 = 18%
        frame = _frame([10, 10.5, 11, 11.2, 11.5, 11.8, 12, 12])
        baseline, r1, r5 = realized_returns(frame, "2026-08-10")
        assert baseline == 10.0
        assert r1 == pytest.approx(0.05)
        assert r5 == pytest.approx(0.18)

    def test_baseline_uses_last_close_on_or_before(self):
        # settlement of a run made on a non-trading boundary: baseline is the
        # Friday close, future bars start the next trading day
        frame = _frame([10, 10, 11, 12, 12, 12, 12, 12], start="2026-08-10")
        baseline, r1, _ = realized_returns(frame, "2026-08-11")  # Tue
        assert baseline == 10.0
        assert r1 == pytest.approx(0.10)  # Wednesday's close

    def test_horizon_shorter_than_5_bars(self):
        frame = _frame([10, 10.2, 10.4])  # only 2 future bars
        _, r1, r5 = realized_returns(frame, "2026-08-10")
        assert r1 == pytest.approx(0.02)
        assert r5 is None

    def test_trade_date_before_series(self):
        frame = _frame([10, 11])
        assert realized_returns(frame, "2026-01-01") == (None, None, None)


def _seed_run(db, table, run_id, trade_date="2026-08-10", picks=None,
              status="done", created_at=None):
    db.execute(
        f"INSERT INTO {table} (id, created_at, status, trade_date, results)"
        " VALUES (?,?,?,?,?)",
        (run_id, created_at or time.time(), status, trade_date,
         json.dumps({"picks": picks or []}, ensure_ascii=False)),
    )


def _picks():
    return [
        {"code": "600519", "name": "贵州茅台", "close": 1500.0, "probability": 0.9},
        {"code": "000001", "name": "平安银行", "close": 10.0, "probability": 0.85},
    ]


def _price_loader_factory(frames_by_code, calls):
    def loader(code):
        calls.append(code)
        return frames_by_code.get(code)
    return loader


@pytest.mark.unit
class TestSettleRunTests:
    def test_settles_and_persists(self, tmp_path):
        db = Database(tmp_path / "dash.db")
        _seed_run(db, "screen_runs", "r1", picks=_picks())
        frames = {
            "600519": _frame([1500, 1530, 1545, 1560, 1575, 1590, 1600]),
            "000001": _frame([10, 9.8, 9.9, 10.1, 10.0, 10.2, 10.1]),
        }
        calls = []
        result = settle_run(db, "screen", "r1",
                            price_loader=_price_loader_factory(frames, calls))

        assert result["exists"] is True
        rows = {r["code"]: r for r in result["picks"]}
        assert rows["600519"]["ret_1d"] == pytest.approx(0.02)
        assert rows["600519"]["ret_5d"] == pytest.approx(0.06)
        assert rows["600519"]["baseline_price"] == 1500.0
        assert rows["600519"]["pick_price"] == 1500.0
        assert rows["600519"]["score"] == 0.9
        assert rows["600519"]["rank"] == 1
        assert rows["000001"]["ret_1d"] == pytest.approx(-0.02)
        assert calls == ["600519", "000001"]

        # re-settle: terminal rows must not refetch
        result2 = settle_run(db, "screen", "r1",
                             price_loader=_price_loader_factory(frames, calls))
        assert calls == ["600519", "000001"]
        assert len(result2["picks"]) == 2

    def test_failed_fetch_stays_pending(self, tmp_path):
        db = Database(tmp_path / "dash.db")
        db.execute(
            "INSERT INTO screen_runs (id, created_at, status, trade_date, results)"
            " VALUES ('r1', ?, 'done', '2026-08-10', ?)",
            (time.time(), json.dumps({"picks": _picks()[:1]}, ensure_ascii=False)),
        )
        calls = []
        result = settle_run(db, "screen", "r1",
                            price_loader=_price_loader_factory({}, calls))
        row = result["picks"][0]
        assert row["ret_1d"] is None and row["ret_5d"] is None
        assert row["baseline_price"] is None
        # non-terminal row is retried on the next view
        settle_run(db, "screen", "r1", price_loader=_price_loader_factory({}, calls))
        assert calls == ["600519", "600519"]

    def test_unknown_run(self, tmp_path):
        db = Database(tmp_path / "dash.db")
        result = settle_run(db, "screen", "nope")
        assert result["exists"] is False and result["picks"] == []

    def test_value_run_score_and_price_fields(self, tmp_path):
        db = Database(tmp_path / "dash.db")
        db.execute(
            "INSERT INTO value_runs (id, created_at, status, trade_date, results)"
            " VALUES ('v1', ?, 'done', '2026-08-10', ?)",
            (time.time(), json.dumps({"picks": [
                {"code": "002466", "name": "天齐锂业", "price": 50.0, "score": 12.5},
            ]}, ensure_ascii=False)),
        )
        frames = {"002466": _frame([50, 52, 51, 53, 52, 54, 55])}
        result = settle_run(db, "value", "v1",
                            price_loader=_price_loader_factory(frames, []))
        row = result["picks"][0]
        assert row["score"] == 12.5 and row["pick_price"] == 50.0
        assert row["ret_5d"] == pytest.approx(0.08)  # T+5 close = 54


@pytest.mark.unit
class TestReviewSummaryTests:
    def test_merges_both_screeners_without_network(self, tmp_path):
        db = Database(tmp_path / "dash.db")
        now = time.time()
        db.execute(
            "INSERT INTO screen_runs (id, created_at, status, trade_date, results)"
            " VALUES ('s1', ?, 'done', '2026-08-10', ?)",
            (now, json.dumps({"picks": _picks()}, ensure_ascii=False)),
        )
        db.execute(
            "INSERT INTO value_runs (id, created_at, status, trade_date, results)"
            " VALUES ('v1', ?, 'done', '2026-08-10', ?)",
            (now + 1, json.dumps({"picks": [{"code": "002466", "name": "x", "price": 5, "score": 9}]},
                                  ensure_ascii=False)),
        )
        db.upsert_pick_return({
            "run_type": "screen", "run_id": "s1", "trade_date": "2026-08-10",
            "code": "600519", "name": "贵州茅台", "pick_price": 1500.0,
            "baseline_price": 1500.0, "score": 0.9, "rank": 1,
            "ret_1d": 0.02, "ret_5d": 0.06, "settled_at": now,
        })
        runs = review_summary(db)
        by_id = {r["run_id"]: r for r in runs}
        assert by_id["s1"]["label"] == "动量精选"
        assert by_id["s1"]["n_picks"] == 2
        assert by_id["s1"]["settled_5d"] == 1
        assert by_id["s1"]["avg_5d"] == pytest.approx(0.06)
        assert by_id["s1"]["hit_rate_5d"] == 1.0
        assert by_id["v1"]["label"] == "价值精选"
        assert by_id["v1"]["settled_5d"] == 0
        assert by_id["v1"]["hit_rate_5d"] is None


@pytest.mark.unit
class TestReviewApiTests:
    def test_endpoints(self, tmp_path):
        db = Database(tmp_path / "dash.db")
        db.execute(
            "INSERT INTO screen_runs (id, created_at, status, trade_date, results)"
            " VALUES ('r1', ?, 'done', '2026-08-10', ?)",
            (time.time(), json.dumps({"picks": _picks()[:1]}, ensure_ascii=False)),
        )
        app = create_app(db=db, start_spot=False, queue=None)
        client = TestClient(app)

        summary = client.get("/api/review/summary").json()
        assert summary["runs"][0]["run_id"] == "r1"

        frames = {"600519": _frame([1500, 1545, 1560, 1575, 1590, 1600, 1610])}
        with mock.patch("server.review._load_prices",
                        side_effect=_price_loader_factory(frames, [])):
            detail = client.get("/api/review/screen/r1").json()
        assert detail["picks"][0]["ret_1d"] == pytest.approx(0.03)

        assert client.get("/api/review/bogus/r1").status_code == 404
