"""Value screener evaluation tests: ROE annualization + look-ahead gating."""
import tempfile
from unittest import mock

import pandas as pd
import pytest

import server.value_screener as vs


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


def _evaluate(rows, curr_date, baidu_value=5.0, ttm_np=40e8):
    fake = mock.Mock()
    fake.stock_financial_analysis_indicator.return_value = _ratios(rows)
    fake.stock_zh_valuation_baidu.return_value = pd.DataFrame({"value": [baidu_value]})
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
