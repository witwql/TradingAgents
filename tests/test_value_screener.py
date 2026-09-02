"""Value screener evaluation tests: ROE annualization + look-ahead gating,
the 距52周低点 price dimension, tie-break ordering, and run-over-run rotation."""
import json
import tempfile
from unittest import mock

import pandas as pd
import pytest

import server.value_screener as vs
from server.db import Database


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch):
    """Keep the statement cache out of the real ~/.tradingagents/cache."""
    import tradingagents.dataflows.config as config_module

    monkeypatch.setattr(config_module, "_config",
                        {**config_module.get_config(), "data_cache_dir": tempfile.mkdtemp()})


def _ratio_row(period, roe):
    return {"日期": pd.Timestamp(period), "净资产收益率(%)": roe,
            "销售净利率(%)": 20.0, "主营业务收入增长率(%)": 10.0,
            "净利润增长率(%)": 30.0, "资产负债率(%)": 40.0,
            "流动比率": 2.0}


def _ratios(rows):
    return pd.DataFrame(rows)


def _evaluate(rows, curr_date, baidu_value=5.0, ttm_np=40e8, mc_values=None):
    fake = mock.Mock()
    fake.stock_financial_analysis_indicator.return_value = _ratios(rows)
    if mc_values is None:
        fake.stock_zh_valuation_baidu.return_value = pd.DataFrame({"value": [baidu_value]})
    else:
        # distinct 总市值 history; 市净率 keeps the single-point default
        fake.stock_zh_valuation_baidu.side_effect = (
            lambda symbol, indicator, period: (
                pd.DataFrame({"value": list(mc_values)})
                if indicator == "总市值"
                else pd.DataFrame({"value": [baidu_value]})
            )
        )
    fake.stock_financial_report_sina.return_value = pd.DataFrame([
        {"报告日": "20260630", "归属于母公司所有者的净利润": ttm_np},
        {"报告日": "20251231", "归属于母公司所有者的净利润": ttm_np},
        {"报告日": "20250630", "归属于母公司所有者的净利润": ttm_np},
        {"报告日": "20250331", "归属于母公司所有者的净利润": ttm_np},
    ])
    # both module-level `ak` bindings: the ratio pass reads vs.ak, while
    # compute_ttm_net_profit reads sina_stock.ak
    with mock.patch.object(vs, "ak", fake), \
            mock.patch("tradingagents.dataflows.sina_stock.ak", fake):
        return vs.evaluate_value_stock("600519", "贵州茅台", 1500.0, curr_date)


@pytest.mark.unit
class TestValueScreenerEvaluationTests:
    def test_roe_annualized_for_interim_period(self):
        # H1'26 ROE 8.77 (6-month cumulative) → annualized 17.54 → score 2
        result = _evaluate([_ratio_row("2026-06-30", 8.77)], "2026-08-31")
        roe = result["metrics"]["ROE(年化)"]
        assert roe["value"] == 17.54
        assert roe["score"] == 2
        # the raw cumulative value must not be scored as-is
        assert "ROE" not in result["metrics"]

    def test_annual_period_roe_not_scaled(self):
        # December report → already annual, ×12/12 = unchanged
        result = _evaluate([_ratio_row("2025-12-31", 8.77)], "2026-03-01")
        assert result["metrics"]["ROE(年化)"]["value"] == 8.77

    def test_future_ratio_row_gated(self):
        # as of Q1, the H1'26 row (disclosed later) must not be scored
        result = _evaluate([
            _ratio_row("2026-03-31", 6.0),
            _ratio_row("2026-06-30", 30.0),
        ], "2026-03-31")
        assert result["metrics"]["ROE(年化)"]["value"] == 24.0  # 6.0 × 12/3

    def test_low_roe_rejected(self):
        assert _evaluate([_ratio_row("2026-06-30", 2.0)], "2026-08-31") is None

    def test_pe_ttm_uses_market_cap_and_ttm_profit(self):
        # 858亿? no — baidu_value=5.0(亿) hmm; use 800亿: PE = 800e8/40e8 = 20
        result = _evaluate([_ratio_row("2026-06-30", 8.77)], "2026-08-31",
                           baidu_value=800.0, ttm_np=40e8)
        assert result["metrics"]["PE-TTM"]["value"] == 20.0


@pytest.mark.unit
class TestLow52PriceDimension:
    """The only daily-moving scored input: without it the picks board stays
    frozen between reporting seasons."""

    def test_far_from_low_scores_zero(self):
        result = _evaluate([_ratio_row("2026-06-30", 8.77)], "2026-08-31",
                           mc_values=[500.0, 700.0, 1000.0])
        assert result["metrics"]["距52周低点"] == {"value": 100.0, "score": 0}
        assert result["max_score"] == 18

    def test_near_low_scores_two(self):
        result = _evaluate([_ratio_row("2026-06-30", 8.77)], "2026-08-31",
                           mc_values=[100.0, 105.0, 108.0])
        assert result["metrics"]["距52周低点"]["value"] == 8.0
        assert result["metrics"]["距52周低点"]["score"] == 2

    def test_mid_distance_scores_one(self):
        result = _evaluate([_ratio_row("2026-06-30", 8.77)], "2026-08-31",
                           mc_values=[100.0, 105.0, 125.0])
        assert result["metrics"]["距52周低点"]["score"] == 1

    def test_single_point_history_degrades_without_low52(self):
        # one Baidu row: PE-TTM still computed from the last point, but a
        # 52w low needs a window → dimension scores 0
        result = _evaluate([_ratio_row("2026-06-30", 8.77)], "2026-08-31",
                           mc_values=[800.0])
        assert result["metrics"]["PE-TTM"]["value"] == 20.0
        assert result["metrics"]["距52周低点"] == {"value": None, "score": 0}


@pytest.mark.unit
class TestTieBreakOrdering:
    def _pick(self, code, score, pe=None, cap=None):
        return {"code": code, "score": score,
                "metrics": {"PE-TTM": {"value": pe}, "总市值": {"value": cap}}}

    def test_ties_break_by_pe_then_cap_then_code(self):
        picks = [
            self._pick("600000", 14, pe=30.0, cap=100),
            self._pick("600002", 14, pe=10.0, cap=500),
            self._pick("600001", 14, pe=10.0, cap=300),
            self._pick("600003", 14, pe=None, cap=10),   # missing PE → last
            self._pick("600004", 14, pe=10.0, cap=None),  # missing cap after pe
        ]
        ordered = sorted(picks, key=vs._pick_sort_key)
        # pe 10 < 30 < inf, then cap 300 < 500 < inf, then code
        assert [p["code"] for p in ordered] == \
            ["600001", "600002", "600004", "600000", "600003"]

    def test_higher_score_always_first(self):
        picks = [self._pick("600001", 14, pe=1.0), self._pick("600002", 16, pe=99.0)]
        ordered = sorted(picks, key=vs._pick_sort_key)
        assert [p["code"] for p in ordered] == ["600002", "600001"]


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "value.db")


def _insert_done_run(db, run_id, picks, *, created_at):
    payload = json.dumps({"evaluated": len(picks), "qualifying": len(picks),
                          "picks": picks, "watchlist": []}, ensure_ascii=False)
    db.execute(
        "INSERT INTO value_runs (id, created_at, finished_at, status, results)"
        " VALUES (?,?,?,?,?)",
        (run_id, created_at, created_at + 60, "done", payload),
    )


@pytest.mark.unit
class TestValueRunChanges:
    def test_none_without_previous_run(self, db):
        _insert_done_run(db, "r1", [{"code": "600519", "name": "贵州茅台", "score": 16}],
                         created_at=100)
        assert vs.value_run_changes(db, "r1", [{"code": "600519"}]) is None

    def test_entered_and_exited(self, db):
        _insert_done_run(db, "r1", [
            {"code": "600519", "name": "贵州茅台", "score": 16},
            {"code": "000807", "name": "云铝股份", "score": 15},
        ], created_at=100)
        _insert_done_run(db, "r2", [
            {"code": "000807", "name": "云铝股份", "score": 16},
            {"code": "601899", "name": "紫金矿业", "score": 14},
        ], created_at=200)
        changes = vs.value_run_changes(db, "r2", [{"code": "000807"}, {"code": "601899"}])
        assert changes["entered"] == ["601899"]
        assert [(x["code"], x["name"]) for x in changes["exited"]] == [("600519", "贵州茅台")]

    def test_diffs_against_latest_previous_run(self, db):
        # a same-day rerun becomes the comparison base for the next run
        _insert_done_run(db, "r1", [{"code": "600519", "name": "贵州茅台", "score": 16}], created_at=100)
        _insert_done_run(db, "r2", [{"code": "601899", "name": "紫金矿业", "score": 16}], created_at=160)
        _insert_done_run(db, "r3", [{"code": "601899", "name": "紫金矿业", "score": 16}], created_at=220)
        changes = vs.value_run_changes(db, "r3", [{"code": "601899"}])
        assert changes == {"entered": [], "exited": []}


@pytest.mark.unit
class TestFuturesContextAnnotation:
    def test_annotate_attaches_context_for_mapped_names(self):
        snap = {"AL0": {"name": "沪铝", "kind": "domestic",
                        "moves": {"1d": 1.5, "5d": 3.0, "20d": 6.0}}}
        picks = [{"code": "000807", "name": "云铝股份"},
                 {"code": "600519", "name": "贵州茅台"}]
        vs.annotate_futures_context(picks, snap)
        assert picks[0]["futures_context"] and "沪铝" in picks[0]["futures_context"]
        assert picks[1]["futures_context"] is None
