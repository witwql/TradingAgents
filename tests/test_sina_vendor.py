"""Sina fallback vendor unit tests — no network unless marked integration."""
import copy
import glob
import os
import tempfile
from unittest import mock

import pandas as pd
import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.dataflows.sina_stock as sina
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.interface import VENDOR_METHODS

DATES = pd.to_datetime(
    ["2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25"]
).normalize()


def _reset(tmp_cache=None):
    cfg = copy.deepcopy(__import__("tradingagents.default_config", fromlist=["DEFAULT_CONFIG"]).DEFAULT_CONFIG)
    if tmp_cache:
        cfg["data_cache_dir"] = tmp_cache
    config_module._config = cfg


def _frame():
    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in DATES],
        "open": [10.0, 10.5, 11.0, 11.5],
        "high": [11.0, 11.5, 12.0, 12.5],
        "low": [9.5, 10.0, 10.5, 11.0],
        "close": [10.5, 11.0, 11.5, 12.0],
        "volume": [100000.0] * 4,
        "amount": [1e7] * 4,
    })


@pytest.mark.unit
class TestSymbolTests:
    def test_sina_symbols(self):
        assert sina.to_sina_symbol("600519.SS") == "sh600519"
        assert sina.to_sina_symbol("600519") == "sh600519"
        assert sina.to_sina_symbol("000001.SZ") == "sz000001"
        assert sina.to_sina_symbol("300750") == "sz300750"

    def test_fund_symbols(self):
        assert sina.to_sina_symbol("510300") == "sh510300"
        assert sina.to_sina_symbol("159994") == "sz159994"
        assert sina.to_sina_symbol("562500") == "sh562500"

    def test_reject_non_ashare(self):
        with pytest.raises(NoMarketDataError):
            sina.to_acode("AAPL")


@pytest.mark.unit
class TestDailyDataTests:
    def test_stock_daily_uses_zh_a_endpoint(self):
        cache = tempfile.mkdtemp()
        fake = mock.Mock()
        fake.stock_zh_a_daily.return_value = _frame()
        with mock.patch.object(sina, "ak", fake), \
                mock.patch.object(config_module, "_config",
                                  {**copy.deepcopy(config_module._config if config_module._config else {}),
                                   **_cfg(cache)}):
            out = sina.get_stock_data_sina("600519.SS", "2026-06-22", "2026-06-25")
        fake.stock_zh_a_daily.assert_called_once()
        kwargs = fake.stock_zh_a_daily.call_args.kwargs
        assert kwargs["symbol"] == "sh600519" and kwargs["adjust"] == "qfq"
        assert "Date,Open,High,Low,Close,Volume" in out
        first_row = out.splitlines()[6]
        assert first_row.startswith("2026-06-22")
        assert out.endswith("2026-06-25,11.5,12.5,11.0,12.0,100000.0\n")

    def test_fund_daily_uses_etf_endpoint(self):
        cache = tempfile.mkdtemp()
        fake = mock.Mock()
        fake.fund_etf_hist_sina.return_value = _frame()
        with mock.patch.object(sina, "ak", fake), _cache_ctx(cache):
            sina.get_stock_data_sina("510300", "2026-06-22", "2026-06-25")
        fake.fund_etf_hist_sina.assert_called_once_with(symbol="sh510300")
        fake.stock_zh_a_daily.assert_not_called()

    def test_missing_price_columns_raise(self):
        bad = _frame().drop(columns=["close"])
        with pytest.raises(NoMarketDataError):
            sina._normalize_frame(bad, "600519.SS", "600519")


@pytest.mark.unit
class TestStatementTests:
    def _income(self):
        rows = []
        for period, rev, np_, eps in [
            ("20251231", 1500.0, 800.0, 10.0),
            ("20260331", 300.0, 200.0, 2.5),
        ]:
            rows.append({"报告日": period, "营业总收入": rev,
                         "归属于母公司所有者的净利润": np_, "基本每股收益": eps})
        return pd.DataFrame(rows)

    def test_income_subset_and_lookahead_cutoff(self):
        cache = tempfile.mkdtemp()
        fake = mock.Mock()
        fake.stock_financial_report_sina.return_value = self._income()
        with mock.patch.object(sina, "ak", fake), _cache_ctx(cache):
            out = sina.get_income_statement_sina("600519.SS", "quarterly", "2026-03-01")
        assert "Report period 2025-12-31" in out
        assert "Total Operating Revenue: 1500.0" in out
        assert "2026-03-31" not in out  # future period gated by report-end date

    def test_annual_filter_keeps_december(self):
        df = pd.concat([self._income(),
                        pd.DataFrame([{"报告日": "20260930", "营业总收入": 700.0,
                                       "归属于母公司所有者的净利润": 400.0,
                                       "基本每股收益": 5.0}])], ignore_index=True)
        fake = mock.Mock()
        fake.stock_financial_report_sina.return_value = df
        with mock.patch.object(sina, "ak", fake), _cache_ctx(tempfile.mkdtemp()):
            out = sina.get_income_statement_sina("600519.SS", "annual")
        assert "09-30" not in out


@pytest.mark.unit
class TestFundamentalsValuationTests:
    """Valuation must come from primary data (market cap / TTM net profit),
    never from Sina's financial-analysis-indicator EPS, whose share-count
    denominator goes stale after placements (002466: 4.06 vs true ~2.7)."""

    def _daily(self, end="2026-08-31"):
        end_ts = pd.Timestamp(end)
        days = [end_ts - pd.Timedelta(days=1), end_ts]
        return pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in days],
            "open": [10.0, 10.5], "high": [11.0, 11.5],
            "low": [9.5, 10.0], "close": [10.5, 11.0],
            "volume": [1e5, 1e5],
        })

    def _ratio_row(self, period, eps, roe):
        return {"日期": pd.Timestamp(period), "摊薄每股收益(元)": eps,
                "每股净资产_调整前(元)": 5.0, "净资产收益率(%)": roe,
                "销售净利率(%)": 20.0, "资产负债率(%)": 40.0,
                "净利润增长率(%)": 30.0, "流动比率": 2.0}

    def _ratios(self):
        # Q1'26 and H1'26 rows; the H1 row is "the future" for curr_date in Q1
        return pd.DataFrame([
            self._ratio_row("2026-03-31", 1.0, 5.0),
            self._ratio_row("2026-06-30", 4.06, 8.77),
        ])

    def _income(self):
        return pd.DataFrame([
            {"报告日": "20260630", "归属于母公司所有者的净利润": 30e8},
            {"报告日": "20260331", "归属于母公司所有者的净利润": 10e8},
            {"报告日": "20251231", "归属于母公司所有者的净利润": 40e8},
            {"报告日": "20250630", "归属于母公司所有者的净利润": 15e8},
            {"报告日": "20250331", "归属于母公司所有者的净利润": 8e8},
        ])

    def _baidu(self, *args, **kwargs):
        return pd.DataFrame({"value": [858.0]})

    def _fundamentals(self, curr_date, ratios=None, daily_end="2026-08-31"):
        cache = tempfile.mkdtemp()
        fake = mock.Mock()
        fake.stock_zh_a_daily.return_value = self._daily(daily_end)
        fake.stock_financial_analysis_indicator.return_value = (
            self._ratios() if ratios is None else ratios)
        fake.stock_financial_report_sina.return_value = self._income()
        fake.stock_zh_valuation_baidu.side_effect = self._baidu
        with mock.patch.object(sina, "ak", fake), _cache_ctx(cache):
            return sina.get_fundamentals_sina("600519.SS", curr_date=curr_date)

    def test_no_stale_share_count_eps(self):
        out = self._fundamentals("2026-08-31")
        assert "EPS (diluted)" not in out
        assert "4.06" not in out                      # stale-share column value
        assert "EPS (TTM" in out
        # shares = 858亿 / 11.0 → EPS-TTM = TTM净利 55亿 / 78亿股 = 0.71
        assert "EPS (TTM, TTM净利润/当前总股本): 0.71" in out
        # PE-TTM = 858e8 / 55e8 = 15.6
        assert "PE-TTM (总市值/TTM净利润): 15.6" in out

    def test_ratio_row_gated_by_curr_date(self):
        out = self._fundamentals("2026-03-31", daily_end="2026-03-31")
        assert "2026-06-30" not in out                # H1 row must not leak
        assert "2026-03-31" in out                    # Q1 row is the visible one

    def test_ratio_period_label_present(self):
        out = self._fundamentals("2026-08-31")
        assert "NOT annualized" in out
        assert "cumulative reporting period 2026-06-30" in out

    def test_negative_ttm_says_na(self):
        income = pd.DataFrame([
            {"报告日": "20260630", "归属于母公司所有者的净利润": -5e8},
            {"报告日": "20251231", "归属于母公司所有者的净利润": -20e8},
            {"报告日": "20250630", "归属于母公司所有者的净利润": -8e8},
        ])
        cache = tempfile.mkdtemp()
        fake = mock.Mock()
        fake.stock_zh_a_daily.return_value = self._daily()
        fake.stock_financial_analysis_indicator.return_value = self._ratios()
        fake.stock_financial_report_sina.return_value = income
        fake.stock_zh_valuation_baidu.side_effect = self._baidu
        with mock.patch.object(sina, "ak", fake), _cache_ctx(cache):
            out = sina.get_fundamentals_sina("600519.SS", curr_date="2026-08-31")
        assert "PE-TTM: N/A (TTM net profit is negative)" in out

    def test_ttm_net_profit_curr_date_gate(self):
        fake = mock.Mock()
        fake.stock_financial_report_sina.return_value = self._income()
        with mock.patch.object(sina, "ak", fake), _cache_ctx(tempfile.mkdtemp()):
            # as of 2026-07-01 only H1'26 visible: TTM = 30 + 40 - 15 = 55亿
            assert sina.compute_ttm_net_profit("sh600519", "2026-07-01") == 55e8
            # as of 2026-04-01 only Q1'26 visible: TTM = 10 + 40 - 8 = 42亿
            assert sina.compute_ttm_net_profit("sh600519", "2026-04-01") == 42e8

    def test_statement_fetched_once_per_day(self):
        """Second same-day call (any scode form) must reuse the disk cache."""
        fake = mock.Mock()
        fake.stock_financial_report_sina.return_value = self._income()
        with mock.patch.object(sina, "ak", fake), _cache_ctx(tempfile.mkdtemp()):
            assert sina.compute_ttm_net_profit("sh600519", "2026-07-01") == 55e8
            # prefixed and bare forms share one cache key
            assert sina.compute_ttm_net_profit("600519", "2026-07-01") == 55e8
            assert sina.compute_ttm_net_profit("600519") == 55e8
            assert fake.stock_financial_report_sina.call_count == 1

    def test_statement_cache_supersedes_old_files(self):
        fake = mock.Mock()
        fake.stock_financial_report_sina.return_value = self._income()
        cache = tempfile.mkdtemp()
        stale = os.path.join(cache, "sh600519-Sina-stmt-利润表-2020-01-01.csv")
        open(stale, "w").close()
        with mock.patch.object(sina, "ak", fake), _cache_ctx(cache):
            sina.compute_ttm_net_profit("sh600519")
        assert not os.path.exists(stale)
        assert len(glob.glob(os.path.join(cache, "sh600519-Sina-stmt-*.csv"))) == 1

    def test_daily_ohlcv_cache_supersedes_old_windows(self):
        fake = mock.Mock()
        fake.stock_zh_a_daily.return_value = _frame()
        cache = tempfile.mkdtemp()
        stale = os.path.join(cache, "600519.SS-Sina-data-2020-01-01-2025-01-01.csv")
        open(stale, "w").close()
        with mock.patch.object(sina, "ak", fake), _cache_ctx(cache):
            sina.get_stock_data_sina("600519.SS", "2026-06-22", "2026-06-25")
        assert not os.path.exists(stale)
        assert len(glob.glob(os.path.join(cache, "600519.SS-Sina-data-*.csv"))) == 1


@pytest.mark.unit
class TestRouterRegistrationTests:
    def test_sina_registered_everywhere_expected(self):
        for method in ("get_stock_data", "get_indicators", "get_fundamentals",
                       "get_balance_sheet", "get_cashflow", "get_income_statement"):
            assert "sina" in VENDOR_METHODS[method], method

    def test_ohlcv_source_chain_routes_to_sina_adapter(self):
        import tradingagents.dataflows.sina_stock as ss
        from tradingagents.dataflows import stockstats_utils as su

        frame = _frame()
        normalized = pd.DataFrame({
            "Date": pd.to_datetime(frame["date"]),
            "Open": frame["open"], "High": frame["high"],
            "Low": frame["low"], "Close": frame["close"],
            "Volume": frame["volume"],
        })
        delegate = mock.Mock(return_value=normalized)
        set_config({"data_vendors": {"technical_indicators": "sina"}})

        def fail_download(*a, **k):
            raise AssertionError("yfinance must not run when sina is primary")

        with mock.patch.object(su, "yf", mock.Mock(download=fail_download)), \
                mock.patch.object(ss, "fetch_daily_ohlcv_sina", delegate):
            data = su.load_ohlcv("600519.SS", "2026-06-25")
        delegate.assert_called_once()
        assert {"Date", "Open", "High", "Low", "Close", "Volume"} <= set(data.columns)


# ---------------------------------------------------------------------------
def _cfg(cache):
    base = copy.deepcopy(_default())
    base["data_cache_dir"] = cache
    return base


def _cache_ctx(cache):
    return mock.patch.object(config_module, "_config", _cfg(cache))


_D = None
def _default():
    global _D
    if _D is None:
        import tradingagents.default_config as dc
        _D = dc.DEFAULT_CONFIG
    return _D
