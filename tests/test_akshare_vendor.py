"""AKShare vendor unit tests (no network — all endpoints mocked).

Covers the vendor contract end to end at unit level:
- symbol conversion & non-A-share rejection (so the router can fall back),
- OHLCV normalization (column rename, lot->share volume, window filters),
- indicator/validator dispatch when akshare is the configured OHLCV source,
- look-ahead filtering for fundamentals/statements/news windows,
- router registration and configured-chain routing.
"""
import copy
import unittest
from unittest import mock

import pandas as pd
import pytest

import tradingagents.dataflows.akshare_stock as aks
import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.symbol_utils import NoMarketDataError, normalize_symbol


def _reset_config():
    # Hard reset like test_vendor_routing: set_config() merges nested dicts.
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _hist_frame(dates, base=10.0):
    n = len(dates)
    return pd.DataFrame(
        {
            "日期": [d.strftime("%Y-%m-%d") for d in dates],
            "股票代码": ["600519"] * n,
            "开盘": [base + i for i in range(n)],
            "收盘": [base + i + 0.5 for i in range(n)],
            "最高": [base + i + 1.0 for i in range(n)],
            "最低": [base + i - 0.5 for i in range(n)],
            "成交量": [1000 + i for i in range(n)],  # 手
        }
    )


DATES = pd.to_datetime(["2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25"]).normalize()


@pytest.mark.unit
class SymbolConversionTests(unittest.TestCase):
    def test_acode_accepts_suffix_and_bare_forms(self):
        assert aks.to_acode("600519.SS") == "600519"
        assert aks.to_acode("600519.SH") == "600519"
        assert aks.to_acode("600519") == "600519"
        assert aks.to_acode("000001.sz") == "000001"
        assert aks.to_acode("300750") == "300750"

    def test_non_ashare_raises_no_market_data(self):
        with pytest.raises(NoMarketDataError):
            aks.to_acode("AAPL")
        with pytest.raises(NoMarketDataError):
            aks.to_acode("830799.BJ")  # BSE has no auto mapping

    def test_em_symbol_prefixes(self):
        assert aks.to_em_symbol("600519.SS") == "SH600519"
        assert aks.to_em_symbol("000001") == "SZ000001"

    def test_normalize_symbol_ashare_rules(self):
        assert normalize_symbol("600519") == "600519.SS"
        assert normalize_symbol("600519.SH") == "600519.SS"
        assert normalize_symbol("000001") == "000001.SZ"
        assert normalize_symbol("AAPL") == "AAPL"  # untouched


@pytest.mark.unit
class StockDataTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_get_stock_data_renames_columns_and_converts_lots(self):
        import tempfile

        cache_dir = tempfile.mkdtemp(prefix="ta-akshare-test-")
        fake_ak = mock.Mock()
        fake_ak.stock_zh_a_hist.return_value = _hist_frame(DATES)
        cfg = {**copy.deepcopy(default_config.DEFAULT_CONFIG), "data_cache_dir": cache_dir}
        with mock.patch.object(aks, "ak", fake_ak), \
                mock.patch.object(config_module, "_config", cfg):
            out = aks.get_stock_data_akshare("600519.SS", "2026-06-22", "2026-06-25")

        assert "# Stock data for 600519.SS [A-share 600519]" in out
        assert ",Open,High,Low,Close,Volume" in out
        # first row: O=10.0 H=11.0 L=9.5 C=10.5; volume converted from lots to shares
        assert ",10.0,11.0,9.5,10.5,100000\n" in out
        called = fake_ak.stock_zh_a_hist.call_args.kwargs
        assert called["symbol"] == "600519"
        assert called["adjust"] == "qfq"

    def test_window_filter_is_inclusive(self):
        frame = _hist_frame(DATES)
        data = aks._normalize_eastmoney_frame(frame, "600519.SS", "600519")
        data["Date"] = pd.to_datetime(data["Date"])
        win = data[
            (data["Date"] >= pd.Timestamp("2026-06-23"))
            & (data["Date"] <= pd.Timestamp("2026-06-24"))
        ]
        assert list(win["Date"].dt.strftime("%Y-%m-%d")) == ["2026-06-23", "2026-06-24"]

    def test_stale_history_raises_no_market_data(self):
        from tradingagents.dataflows.stockstats_utils import _assert_ohlcv_not_stale

        old = aks._normalize_eastmoney_frame(
            _hist_frame(pd.to_datetime(["2024-01-02"]).normalize()), "600519.SS", "600519"
        )
        old["Date"] = pd.to_datetime(old["Date"])
        with pytest.raises(NoMarketDataError):
            _assert_ohlcv_not_stale(old, "2026-06-25", "600519.SS")

    def test_missing_columns_raises_no_market_data(self):
        bad = _hist_frame(DATES).drop(columns=["收盘"])
        with pytest.raises(NoMarketDataError):
            aks._normalize_eastmoney_frame(bad, "600519.SS", "600519")


@pytest.mark.unit
class IndicatorTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_unsupported_indicator_raises_value_error(self):
        with pytest.raises(ValueError):
            aks.get_indicators_akshare("600519.SS", "not_an_indicator", "2026-06-25", 3)

    def test_indicator_report_shape(self):
        long_dates = pd.bdate_range(end="2026-06-24", periods=260)
        synthetic = pd.DataFrame(
            {
                "Date": long_dates,
                "Open": 10.0,
                "High": 11.0,
                "Low": 9.5,
                "Close": range(len(long_dates)),
                "Volume": 1e6,
            }
        )
        with mock.patch.object(aks, "fetch_daily_ohlcv_akshare", return_value=synthetic):
            out = aks.get_indicators_akshare("600519.SS", "rsi", "2026-06-24", 3)

        assert out.startswith("## rsi values from 2026-06-21 to 2026-06-24:")
        for d in ("2026-06-24", "2026-06-23", "2026-06-22", "2026-06-21"):
            assert f"{d}: " in out
        assert "RSI:" in out  # shared usage description included


@pytest.mark.unit
class StatementAndFundamentalsTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def _fake_income(self):
        return pd.DataFrame(
            {
                "REPORT_DATE": ["2026-03-31 00:00:00", "2025-12-31 00:00:00"],
                "NOTICE_DATE": ["2026-04-30 00:00:00", "2026-01-15 00:00:00"],
                "TOTAL_OPERATE_INCOME": [300.0, 1500.0],
                "PARENT_NETPROFIT": [200.0, 800.0],
                "BASIC_EPS": [2.5, 10.0],
            }
        )

    def test_income_statement_lookahead_filtered_by_notice_date(self):
        with mock.patch.object(aks, "_fetch_statement_report", return_value=self._fake_income()):
            out = aks.get_income_statement_akshare("600519.SS", "quarterly", "2026-03-01")
        # Q1 notice lands Apr 30 -> must be invisible on Mar 1
        assert "2026-03-31" not in out
        assert "Report period 2025-12-31, disclosed 2026-01-15" in out
        assert "营业总收入 Total Operating Revenue: 1500.0" in out

    def test_annual_filter_keeps_only_december_periods(self):
        report = self._fake_income()
        extra = report.copy()
        extra["REPORT_DATE"] = [
            "2025-09-30 00:00:00" if i % 2 else extra["REPORT_DATE"][i]
            for i in range(len(extra))
        ]
        big = pd.concat([report, extra], ignore_index=True)
        with mock.patch.object(aks, "_fetch_statement_report", return_value=big):
            out = aks.get_income_statement_akshare("600519.SS", "annual", None)
        assert "09-30" not in out

    def test_fundamentals_snapshot_respects_curr_date(self):
        valuation = pd.DataFrame(
            {
                "数据日期": ["2026-06-20", "2026-06-24"],
                "当日收盘价": [1400.0, 1420.0],
                "总市值": [1.76e12, 1.78e12],
                "PE(TTM)": [21.0, 21.5],
            }
        )
        info = pd.DataFrame(
            {"item": ["股票简称", "行业"], "value": ["贵州茅台", "酿酒行业"]}
        )
        fake_ak = mock.Mock()
        fake_ak.stock_value_em.return_value = valuation
        fake_ak.stock_individual_info_em.return_value = info
        with mock.patch.object(aks, "ak", fake_ak):
            out = aks.get_fundamentals_akshare("600519.SS", curr_date="2026-06-23")

        assert "Valuation Date: 2026-06-20" in out
        assert "Close Price: 1400.0" in out
        assert "PE Ratio (TTM): 21.0" in out
        assert "贵州茅台 (酿酒行业)" in out

    def test_insider_changes_listing(self):
        changes = pd.DataFrame(
            {
                "公告日期": ["2025-01-05", "2024-06-08"],
                "变动股东": ["张三", "李四"],
                "变动数量": ["增持1万股", "减持2万股"],
            }
        )
        fake_ak = mock.Mock()
        fake_ak.stock_shareholder_change_ths.return_value = changes
        with mock.patch.object(aks, "ak", fake_ak):
            out = aks.get_insider_transactions_akshare("600519.SS")
        assert "Shareholder & Executive Stake Changes" in out
        assert "增持1万股" in out
        assert out.index("2025-01-05") < out.index("2024-06-08")


@pytest.mark.unit
class NewsTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def _news_frame(self):
        return pd.DataFrame(
            {
                "关键词": ["600519"] * 3,
                "新闻标题": ["past news", "in-window news", "future news"],
                "新闻内容": ["a", "b", "c"],
                "发布时间": [
                    "2026-06-20 09:00:00",
                    "2026-06-24 10:00:00",
                    "2026-07-01 10:00:00",
                ],
                "文章来源": ["s1", "s2", "s3"],
                "新闻链接": ["l1", "l2", "l3"],
            }
        )

    def test_stock_news_window_filter(self):
        from tradingagents.dataflows import akshare_news as akn

        fake_ak = mock.Mock()
        fake_ak.stock_news_em.return_value = self._news_frame()
        with mock.patch.object(akn, "ak", fake_ak):
            out = akn.get_news_akshare("600519.SS", "2026-06-23", "2026-06-25")
        assert "in-window news" in out
        assert "past news" not in out
        assert "future news" not in out

    def test_global_news_window_filter(self):
        from tradingagents.dataflows import akshare_news as akn

        raw = pd.DataFrame(
            {
                "标题": ["宏观A", "宏观B", "宏观C"],
                "摘要": ["sa", "sb", "sc"],
                "链接": ["x1", "x2", "x3"],
                "发布时间": [
                    "2026-06-28 08:00:00",  # inside 7-day window
                    "2020-01-01 08:00:00",  # too old
                    "2026-08-01 08:00:00",  # future vs curr_date
                ],
            }
        )
        fake_ak = mock.Mock()
        fake_ak.stock_info_global_em.return_value = raw
        with mock.patch.object(akn, "ak", fake_ak):
            out = akn.get_global_news_akshare("2026-06-29", look_back_days=7, limit=10)
        assert "宏观A" in out
        assert "宏观B" not in out
        assert "宏观C" not in out


@pytest.mark.unit
class RouterIntegrationTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_all_market_methods_register_akshare(self):
        for method in (
            "get_stock_data",
            "get_indicators",
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ):
            assert "akshare" in interface.VENDOR_METHODS[method], method

    def test_router_dispatches_to_akshare_when_configured(self):
        set_config({"data_vendors": {"core_stock_apis": "akshare"}})
        sentinel_impl = mock.Mock(return_value="AK_DATA")
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {**interface.VENDOR_METHODS["get_stock_data"],
                                "akshare": sentinel_impl}},
        ):
            result = interface.route_to_vendor("get_stock_data", "600519.SS", "2026-06-01", "2026-06-25")
        assert result == "AK_DATA"
        sentinel_impl.assert_called_once_with("600519.SS", "2026-06-01", "2026-06-25")

    def test_load_ohlcv_delegates_to_akshare_source_for_validator(self):
        """Verified snapshot + indicators follow the configured OHLCV source."""
        from tradingagents.dataflows import stockstats_utils as su
        from tradingagents.dataflows.market_data_validator import build_verified_market_snapshot

        dates = pd.bdate_range(end="2026-06-24", periods=260)
        frame = pd.DataFrame(
            {
                "Date": dates,
                "Open": 10.0,
                "High": 11.0,
                "Low": 9.5,
                "Close": range(len(dates)),
                "Volume": 1e6,
            }
        )
        delegate = mock.Mock(return_value=frame)
        set_config({"data_vendors": {"technical_indicators": "akshare"}})

        def _fail_download(*a, **k):
            raise AssertionError("yfinance must not be used under the akshare source")

        with mock.patch.object(su, "yf", mock.Mock(download=_fail_download)), \
                mock.patch.object(aks, "fetch_daily_ohlcv_akshare", delegate):
            snapshot = build_verified_market_snapshot("600519.SS", "2026-06-24", look_back_days=5)

        assert "Verified market data snapshot for 600519.SS" in snapshot
        delegate.assert_called_once()
