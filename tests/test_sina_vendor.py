"""Sina fallback vendor unit tests — no network unless marked integration."""
import copy
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
