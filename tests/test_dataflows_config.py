"""Config isolation: get/set must not leak nested-dict references."""

import copy
import unittest

import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows.config import config_scope, get_config, set_config


@pytest.mark.unit
class DataflowsConfigIsolationTests(unittest.TestCase):
    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_get_config_returns_deep_copy(self):
        cfg = get_config()
        cfg["data_vendors"]["core_stock_apis"] = "alpha_vantage"
        cfg["tool_vendors"]["get_stock_data"] = "alpha_vantage"

        fresh = get_config()
        self.assertEqual(fresh["data_vendors"]["core_stock_apis"], "yfinance")
        self.assertNotIn("get_stock_data", fresh["tool_vendors"])

    def test_set_config_does_not_alias_caller_nested_dicts(self):
        custom = copy.deepcopy(default_config.DEFAULT_CONFIG)
        custom["data_vendors"]["core_stock_apis"] = "alpha_vantage"
        custom["tool_vendors"]["get_stock_data"] = "alpha_vantage"

        set_config(custom)

        custom["data_vendors"]["core_stock_apis"] = "yfinance"
        custom["tool_vendors"]["get_stock_data"] = "yfinance"

        fresh = get_config()
        self.assertEqual(fresh["data_vendors"]["core_stock_apis"], "alpha_vantage")
        self.assertEqual(fresh["tool_vendors"]["get_stock_data"], "alpha_vantage")

    def test_partial_nested_update_preserves_existing_defaults(self):
        set_config(
            {
                "data_vendors": {
                    "core_stock_apis": "alpha_vantage",
                }
            }
        )

        fresh = get_config()
        self.assertEqual(fresh["data_vendors"]["core_stock_apis"], "alpha_vantage")
        self.assertEqual(fresh["data_vendors"]["technical_indicators"], "yfinance")
        self.assertEqual(fresh["data_vendors"]["fundamental_data"], "yfinance")
        self.assertEqual(fresh["data_vendors"]["news_data"], "yfinance")

    def test_nested_dict_updates_merge_one_level_deep(self):
        set_config({"tool_vendors": {"get_stock_data": "alpha_vantage"}})
        set_config({"tool_vendors": {"get_news": "alpha_vantage"}})

        fresh = get_config()
        self.assertEqual(fresh["tool_vendors"]["get_stock_data"], "alpha_vantage")
        self.assertEqual(fresh["tool_vendors"]["get_news"], "alpha_vantage")


@pytest.mark.unit
class ScopeAwareSetConfigTests(unittest.TestCase):
    """set_config inside a config_scope must not leak to the process global —

    this is the property that makes parallel dashboard task workers safe."""

    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_outside_scope_still_updates_global(self):
        set_config({"llm_provider": "google"})
        assert get_config()["llm_provider"] == "google"
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_inside_scope_global_untouched(self):
        with config_scope({"llm_provider": "glm-cn", "temperature": 0.3}):
            set_config({"llm_provider": "deepseek"})
            # scoped view sees the update
            assert get_config()["llm_provider"] == "deepseek"
        # global keeps its pre-scope value
        assert get_config()["llm_provider"] == default_config.DEFAULT_CONFIG["llm_provider"]

    def test_nested_dict_merge_applies_to_scope_only(self):
        vendors = copy.deepcopy(default_config.DEFAULT_CONFIG["data_vendors"])
        set_config({"data_vendors": vendors})
        scoped_vendors = dict(vendors)
        scoped_vendors["news_data"] = "akshare"
        with config_scope({"data_vendors": scoped_vendors}):
            set_config({"data_vendors": {"news_data": "sina"}})
            assert get_config()["data_vendors"]["news_data"] == "sina"
        assert get_config()["data_vendors"]["news_data"] == "yfinance"

    def test_two_scopes_isolated(self):
        with config_scope({"llm_provider": "glm-cn"}):
            set_config({"temperature": 0.1})
            inner = get_config()
        complete = copy.deepcopy(default_config.DEFAULT_CONFIG)
        with config_scope(complete):
            assert get_config()["temperature"] is None
        assert inner["temperature"] == 0.1
