"""Vendor router must respect the configured chain and never silently hide a
broken primary.

Regressions for #988 (explicit single-vendor config still fell back to others),
#289 (fallback ran for unchosen vendors), and #989 (serious primary failures
were swallowed without a trace).
"""
import copy
import unittest
from unittest import mock

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _reset_config():
    # Hard reset: set_config() merges, so empty DEFAULT dicts (e.g. tool_vendors)
    # don't clear keys leaked by other tests. Replace the global outright.
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _no_data(symbol, *a, **k):
    raise NoMarketDataError(symbol, symbol, "no rows")


def _returns(value):
    def impl(symbol, *a, **k):
        return value
    return impl


def _raises(exc):
    def impl(symbol, *a, **k):
        raise exc
    return impl


@pytest.mark.unit
class VendorRoutingTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def _route(self, vendors_for_get_stock_data):
        return mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": vendors_for_get_stock_data},
            clear=False,
        )

    def test_explicit_single_vendor_does_not_fall_back(self):
        # #988: with yfinance pinned, a healthy alpha_vantage must NOT be used.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        av = mock.Mock(side_effect=_returns("AV_DATA"))
        with self._route({"yfinance": _no_data, "alpha_vantage": av}):
            result = interface.route_to_vendor("get_stock_data", "FAKE", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)
        av.assert_not_called()  # the unchosen vendor was never tried

    def test_explicit_multi_vendor_falls_back_within_chain(self):
        # Listing both vendors opts in to ordered fallback.
        set_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}})
        with self._route({"yfinance": _no_data, "alpha_vantage": _returns("AV_DATA")}):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "AV_DATA")

    def test_primary_error_is_logged_not_masked(self):
        # #989: primary errors + fallback no-data -> NO_DATA, but the failure
        # must be visible in logs (broken primary not hidden).
        set_config({"data_vendors": {"core_stock_apis": "yfinance,alpha_vantage"}})
        with self._route({"yfinance": _raises(ValueError("boom")), "alpha_vantage": _no_data}), \
                self.assertLogs("tradingagents.dataflows.interface", level="WARNING") as cm:
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIn("NO_DATA_AVAILABLE", result)
        joined = "\n".join(cm.output)
        self.assertIn("boom", joined)            # the real error surfaced in logs
        self.assertIn("yfinance", joined)

    def test_unknown_configured_vendor_raises(self):
        set_config({"data_vendors": {"core_stock_apis": "bogus_vendor"}})
        with self.assertRaises(ValueError) as ctx:
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertIn("bogus_vendor", str(ctx.exception))

    def test_default_sentinel_uses_all_vendors(self):
        # No explicit choice ("default") keeps the resilient full-chain behavior.
        set_config({"data_vendors": {"core_stock_apis": "default"}})
        with self._route({"yfinance": _no_data, "alpha_vantage": _returns("AV_DATA")}):
            result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "AV_DATA")

    def _route_method(self, method, vendors):
        return mock.patch.dict(interface.VENDOR_METHODS, {method: vendors}, clear=False)

    def test_optional_category_degrades_instead_of_raising(self):
        # An optional enrichment vendor (FRED macro) that raises must NOT abort
        # the run — the router returns a sentinel so the analysis proceeds.
        set_config({"data_vendors": {"macro_data": "fred"}})
        with self._route_method(
            "get_macro_indicators", {"fred": _raises(ValueError("FRED 400: bad series"))}
        ):
            result = interface.route_to_vendor("get_macro_indicators", "cpi", "2026-01-01")
        self.assertIn("DATA_UNAVAILABLE", result)
        self.assertIn("macro_data", result)

    def test_core_category_still_raises_on_error(self):
        # A core category (single configured vendor) propagates the error so a
        # broken primary is loud, not silently degraded.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        with self._route({"yfinance": _raises(ValueError("boom"))}), \
                self.assertRaises(ValueError):
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")


if __name__ == "__main__":
    unittest.main()


def _route_dict(impls):
    return mock.patch.dict(interface.VENDOR_METHODS,
                           {"get_stock_data": {**interface.VENDOR_METHODS["get_stock_data"], **impls}},
                           clear=False)


@pytest.mark.unit
class ThrottleDegradationTests(unittest.TestCase):
    """All-vendors-throttled chains degrade to an instructive sentinel instead
    of killing hour-long analyses; real failures still raise loudly."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_all_vendors_rate_limited_returns_sentinel(self):
        from tradingagents.dataflows.errors import VendorRateLimitError

        set_config({"data_vendors": {"core_stock_apis": "akshare,yfinance"}})
        aks = mock.Mock(side_effect=_raises(VendorRateLimitError("EM dropped x5")))
        yf = mock.Mock(side_effect=_raises(VendorRateLimitError("429 exhausted")))
        with _route_dict({"akshare": aks, "yfinance": yf}):
            result = interface.route_to_vendor(
                "get_stock_data", "600519.SS", "2026-01-01", "2026-01-10"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("rate-limited", result)

    def test_real_failure_still_raises(self):
        # A genuine non-throttle error on the only vendor must raise, not degrade.
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        broken = mock.Mock(side_effect=_raises(ValueError("auth boom")))
        with (
            _route_dict({"yfinance": broken}),
            self.assertRaises(ValueError),
        ):
            interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")

    def test_no_data_beats_rate_limit_sentinel(self):
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        impl = mock.Mock(side_effect=_no_data)
        with _route_dict({"yfinance": impl}):
            result = interface.route_to_vendor(
                "get_stock_data", "FAKE", "2026-01-01", "2026-01-10"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)


@pytest.mark.unit
class ConfigScopeTests(unittest.TestCase):
    """config_scope isolates per-run config without mutating the global."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_scope_overrides_and_restores(self):
        from tradingagents.dataflows.config import config_scope, get_config

        assert get_config()["data_vendors"]["core_stock_apis"] == "yfinance"
        with config_scope({**get_config(), "data_vendors": {"core_stock_apis": "sina"}}):
            assert get_config()["data_vendors"]["core_stock_apis"] == "sina"
        assert get_config()["data_vendors"]["core_stock_apis"] == "yfinance"

    def test_routed_calls_honor_scope(self):
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})
        impls = {
            "sina": _returns("SINA_DATA"),
            "yfinance": _returns("YF_DATA"),
        }
        from tradingagents.dataflows.config import config_scope

        from tradingagents.dataflows.config import get_config as _gc

        scoped = {**_gc(), "data_vendors": {"core_stock_apis": "sina"}}
        with (
            config_scope(scoped),
            _route_dict(impls),
        ):
            result = interface.route_to_vendor(
                "get_stock_data", "600519.SS", "2026-01-01", "2026-01-10"
            )
        assert result == "SINA_DATA"
        assert _gc()["data_vendors"]["core_stock_apis"] == "yfinance"  # global untouched
