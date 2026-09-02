"""Global macro analyst: factor tools, exposure model, and graph wiring."""
import unittest
from unittest import mock

import pandas as pd
import pytest

import numpy as np
import tradingagents.dataflows.global_macro as gm
from tradingagents.dataflows.errors import NoMarketDataError


def _daily_frame(dates, closes):
    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1000] * len(closes),
    })


def _mock_ak(monkeypatch):
    """Deterministic factor series: gold +1%/day, crude -1%/day."""
    dates = pd.bdate_range(end="2026-06-24", periods=260)
    gold = _daily_frame(dates, [100 + i for i in range(len(dates))])
    crude = _daily_frame(dates, [100 - i * 0.5 for i in range(len(dates))])
    yields = pd.DataFrame({
        "日期": [d.strftime("%Y-%m-%d") for d in dates],
        "美国国债收益率2年": [4.0] * len(dates),
        "美国国债收益率10年": [4.2 + i * 0.001 for i in range(len(dates))],
    })
    spx = _daily_frame(dates, [5000 + i * 2 for i in range(len(dates))])

    fake = mock.Mock()
    fake.futures_foreign_hist.side_effect = lambda symbol: {"GC": gold, "CL": crude}[symbol]
    fake.bond_zh_us_rate.return_value = yields
    fake.index_us_stock_sina.side_effect = lambda symbol: {
        ".INX": spx, ".IXIC": spx, ".DJI": spx}[symbol]
    monkeypatch.setattr(gm, "ak", fake)


@pytest.fixture(autouse=True)
def _futures_basket_isolation(monkeypatch):
    """The snapshot cache is process-wide — reset it between tests — and a
    missing basket symbol must fail fast (real retries sleep 2/4/8s)."""
    monkeypatch.setattr(gm, "_QUIET_RETRIES", 0)
    monkeypatch.setattr(gm, "_FUTURES_CACHE", {})
    monkeypatch.setattr(gm, "_FUTURES_CACHE_TS", 0.0)


@pytest.mark.unit
class TestFactorTools:
    def test_gold_report_contains_moves(self, monkeypatch):
        _mock_ak(monkeypatch)
        out = gm.get_gold_price.func("2026-06-24", 30)
        assert "国际金价" in out and "日变动" in out and "20日累计" in out

    def test_yields_reported_in_bp(self, monkeypatch):
        _mock_ak(monkeypatch)
        out = gm.get_us_treasury_yields.func("2026-06-24")
        assert "10年期" in out and "bp" in out and "10Y-2Y利差" in out

    def test_lookahead_cutoff(self, monkeypatch):
        _mock_ak(monkeypatch)
        out = gm.get_gold_price.func("2026-03-01", 30)  # 03-01 is a Sunday
        assert "截至 2026-02-27" in out  # last trading day before the cutoff
        assert "2026-06" not in out

    def test_us_indices_section(self, monkeypatch):
        _mock_ak(monkeypatch)
        out = gm.get_us_stock_indices.func("2026-06-24")
        for name in ("标普500", "纳斯达克", "道琼斯"):
            assert name in out


@pytest.mark.unit
class TestExposureModel:
    def test_correlated_factor_scores_significant(self, monkeypatch):
        """Target returns equal the previous day's gold returns -> corr ≈ +1."""
        dates = pd.bdate_range(end="2026-06-24", periods=260)
        import math

        # Wavy gold so returns have variance (a perfectly linear series has none)
        gold_closes = [100 + 10 * math.sin(i / 5) + 0.2 * i for i in range(len(dates))]
        gold = _daily_frame(dates, gold_closes)
        crude = _daily_frame(dates, [50 - i * 0.1 for i in range(len(dates))])
        yields = pd.DataFrame({
            "日期": [d.strftime("%Y-%m-%d") for d in dates],
            "美国国债收益率2年": [4.0] * len(dates),
            "美国国债收益率10年": [4.2] * len(dates),
        })
        spx = _daily_frame(dates, [5000] * len(dates))
        fake = mock.Mock()
        fake.futures_foreign_hist.side_effect = lambda symbol: {"GC": gold, "CL": crude}[symbol]
        fake.bond_zh_us_rate.return_value = yields
        fake.index_us_stock_sina.side_effect = lambda symbol: spx
        monkeypatch.setattr(gm, "ak", fake)

        # Target close(t) = gold close(t-1) * 2  ->  target_ret(t) == gold_ret(t-1)
        target_closes = [gold_closes[0] * 2] + [g * 2 for g in gold_closes[:-1]]
        frame = pd.DataFrame({
            "Date": dates, "Open": target_closes, "High": target_closes,
            "Low": target_closes, "Close": target_closes, "Volume": [1e6] * len(dates),
        })
        with mock.patch.object(gm, "load_ohlcv", return_value=frame):
            out = gm.get_factor_exposure.func("600519.SS", "2026-06-24", lookback_days=120)
        assert "因子暴露度模型" in out
        gold_line = next(l for l in out.splitlines() if l.startswith("| COMEX黄金"))
        corr_value = float(gold_line.split("|")[2].strip().replace("+", ""))
        assert corr_value > 0.9, gold_line
        assert "隔夜因子综合得分" in out

    def test_insufficient_overlap_reported(self, monkeypatch):
        _mock_ak(monkeypatch)
        tiny = pd.DataFrame({
            "Date": pd.bdate_range(end="2026-06-25", periods=10),
            "Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5,
            "Volume": 1e6,
        })
        with mock.patch.object(gm, "load_ohlcv", return_value=tiny):
            out = gm.get_factor_exposure.func("600519.SS", "2026-06-24")
        assert "样本不足" in out


@pytest.mark.unit
class TestFuturesBasket:
    """15-symbol basket: name mapping, snapshot cache, look-ahead, exposure."""

    def _fake_ak(self, dates, periods=None):
        n = periods or len(dates)
        foreign = {
            "GC": _daily_frame(dates[:n], [1800 + i for i in range(n)]),
            "SI": _daily_frame(dates[:n], [30.0 + i * 0.01 for i in range(n)]),
            "HG": _daily_frame(dates[:n], [4.0 + i * 0.001 for i in range(n)]),
            "CL": _daily_frame(dates[:n], [70.0 - i * 0.01 for i in range(n)]),
            "NG": _daily_frame(dates[:n], [2.5] * n),
        }
        domestic_dates = [d.strftime("%Y-%m-%d") for d in dates[:n]]
        domestic = {
            key: pd.DataFrame({"日期": domestic_dates,
                               "收盘价": [100 + i * 0.2 for i in range(n)]})
            for key in ("AU0", "AG0", "CU0", "AL0", "ZN0", "RB0", "I0", "M0", "JM0", "LH0")
        }
        fake = mock.Mock()
        fake.futures_foreign_hist.side_effect = lambda symbol: foreign[symbol]
        fake.futures_main_sina.side_effect = lambda symbol: domestic[symbol]
        fake.index_us_stock_sina.side_effect = lambda symbol: _daily_frame(dates[:n], [5000.0] * n)
        return fake

    @pytest.mark.parametrize("name,expected", [
        ("云铝股份", ["AL0"]),
        ("江西铜业", ["CU0", "HG"]),
        ("山东黄金", ["AU0", "GC"]),
        ("中国海油", ["CL"]),
        ("新天然气", ["NG"]),
        ("牧原股份", ["LH0", "M0"]),
        ("宝钢股份", ["RB0"]),
        # precision over recall: a bare 金 is wind power / real estate, not gold
        ("金风科技", []),
        ("金地集团", []),
        ("贵州茅台", []),
    ])
    def test_stock_futures_map(self, name, expected):
        assert gm.stock_futures_map(name) == expected

    def test_snapshot_moves_kind_and_lookahead(self, monkeypatch):
        dates = pd.bdate_range(end="2026-06-24", periods=30)
        monkeypatch.setattr(gm, "ak", self._fake_ak(dates))
        snap = gm.futures_snapshot("2026-06-24")
        assert snap["GC"]["name"] == "COMEX黄金" and snap["GC"]["kind"] == "global"
        assert snap["AL0"]["kind"] == "domestic"
        # steadily rising series → positive moves
        assert snap["GC"]["moves"]["1d"] > 0 and snap["GC"]["moves"]["5d"] > 0
        early = gm.futures_snapshot("2026-06-10")
        assert early["GC"]["close"].index.max() <= pd.Timestamp("2026-06-10")

    def test_snapshot_cached_across_calls(self, monkeypatch):
        dates = pd.bdate_range(end="2026-06-24", periods=30)
        fake = self._fake_ak(dates)
        monkeypatch.setattr(gm, "ak", fake)
        gm.futures_snapshot("2026-06-24")
        assert fake.futures_foreign_hist.call_count == 5
        assert fake.futures_main_sina.call_count == 10
        gm.futures_snapshot("2026-06-24")
        assert fake.futures_foreign_hist.call_count == 5   # no refetch within TTL

    def test_snapshot_partial_failure_degrades(self, monkeypatch):
        dates = pd.bdate_range(end="2026-06-24", periods=30)
        fake = self._fake_ak(dates)
        original = fake.futures_foreign_hist.side_effect

        def flaky(symbol):
            if symbol == "NG":
                raise RuntimeError("boom")
            return original(symbol)

        fake.futures_foreign_hist.side_effect = flaky
        monkeypatch.setattr(gm, "ak", fake)
        snap = gm.futures_snapshot("2026-06-24")
        assert "NG" not in snap and "GC" in snap and "AL0" in snap

    def test_context_line_lists_mapped_futures(self, monkeypatch):
        dates = pd.bdate_range(end="2026-06-24", periods=30)
        monkeypatch.setattr(gm, "ak", self._fake_ak(dates))
        snap = gm.futures_snapshot("2026-06-24")
        line = gm.futures_context_line("云铝股份", snap)
        assert line and "沪铝" in line
        assert gm.futures_context_line("贵州茅台", snap) is None

    def test_exposure_table_contains_futures_rows(self, monkeypatch):
        dates = pd.bdate_range(end="2026-06-24", periods=260)
        fake = self._fake_ak(dates)
        yields = pd.DataFrame({
            "日期": [d.strftime("%Y-%m-%d") for d in dates],
            "美国国债收益率2年": [4.0] * len(dates),
            "美国国债收益率10年": [4.2] * len(dates),
        })
        fake.bond_zh_us_rate.return_value = yields
        monkeypatch.setattr(gm, "ak", fake)
        target = pd.DataFrame({
            "Date": dates, "Open": 10.0, "High": 11.0, "Low": 9.0,
            "Close": [10 + 0.05 * ((-1) ** i) for i in range(len(dates))],
            "Volume": 1e6,
        })
        with mock.patch.object(gm, "load_ohlcv", return_value=target):
            with mock.patch("tradingagents.dataflows.money_flow.fetch_money_flow",
                            side_effect=RuntimeError("skip")):
                out = gm.get_factor_exposure.func("600519.SS", "2026-06-24", 120)
        for label in ("COMEX黄金", "NYMEX原油", "沪铜", "沪铝", "螺纹钢", "生猪"):
            assert f"| {label}" in out
        assert "联动最强的期货" in out or "噪声级" in out



    def test_macro_spec_registered(self):
        from tradingagents.graph.analyst_execution import (
            ANALYST_NODE_SPECS,
            build_analyst_execution_plan,
        )

        spec = ANALYST_NODE_SPECS["macro"]
        assert spec.agent_node == "Global Macro Analyst"
        assert spec.tool_node == "tools_macro"
        assert spec.report_key == "macro_report"

        plan = build_analyst_execution_plan(("market", "macro"))
        keys = [s.key for s in plan.specs]
        assert keys == ["market", "macro"]

    def test_trading_graph_has_macro_tool_node(self):
        import inspect

        from tradingagents.graph.trading_graph import TradingAgentsGraph

        src = inspect.getsource(TradingAgentsGraph._create_tool_nodes)
        assert '"macro"' in src and "get_factor_exposure" in src

    def test_runner_stage_mapping(self):
        from server.runner import ALL_ANALYST_KEYS, NODE_TO_STAGE, STAGE_LABELS, build_stages

        assert "macro" in ALL_ANALYST_KEYS
        assert NODE_TO_STAGE["Global Macro Analyst"] == "macro"
        assert STAGE_LABELS["macro"] == "全球宏观分析师"
        stages = [s for s, _ in build_stages(list(ALL_ANALYST_KEYS))]
        assert "macro" in stages
        assert stages.index("macro") < stages.index("bull_researcher")

    def test_reporting_tree_includes_macro(self):
        import tempfile

        from tradingagents.reporting import write_report_tree

        state = {
            "market_report": "M", "macro_report": "MACRO",
            "investment_debate_state": {"bull_history": "", "bear_history": "",
                                        "judge_decision": ""},
            "risk_debate_state": {},
            "trader_investment_plan": "", "investment_plan": "",
            "final_trade_decision": "",
        }
        out = write_report_tree(state, "T", tempfile.mkdtemp())
        assert (out.parent / "1_analysts" / "macro.md").read_text(encoding="utf-8") == "MACRO"

    def test_macro_section_helper_empty_when_absent(self):
        from tradingagents.agents.utils.agent_utils import get_macro_section

        assert get_macro_section({}) == ""
        assert get_macro_section({"macro_report": ""}) == ""
        section = get_macro_section({"macro_report": "综合：利多"})
        assert "综合：利多" in section and "Global macro" in section


class TestMoneyFlow:
    def _flow_frame(self):
        dates = pd.bdate_range(end="2026-06-24", periods=40)
        # 主力连续 5 天净流入，近3日价格上行 → 健康趋势场景
        amounts = [-2e8] * 35 + [1.5e8, 2.0e8, 2.5e8, 3.0e8, 3.5e8]
        return pd.DataFrame({
            "日期": [d.date() for d in dates],
            "收盘价": np.linspace(100, 104, len(dates)),
            "涨跌幅": [0.5] * len(dates),
            "主力净流入-净额": amounts,
            "主力净流入-净占比": [a / 1e9 * 10 for a in amounts],
            "超大单净流入-净额": [a / 3 for a in amounts],
            "大单净流入-净额": [a * 2 / 3 for a in amounts],
        })

    def test_flow_report_streak_and_levels(self, monkeypatch):
        import tradingagents.dataflows.money_flow as mf

        fake = mock.Mock()
        fake.stock_individual_fund_flow.return_value = self._flow_frame()
        monkeypatch.setattr(mf, "ak", fake)
        monkeypatch.setattr(mf, "get_config",
                            lambda: {"data_cache_dir": __import__("tempfile").mkdtemp()})
        import tradingagents.dataflows.global_macro as g
        out = g.get_money_flow.func("600519.SS", "2026-06-24", 30)
        assert "主力资金流" in out
        assert "连续净流入 5 天" in out
        assert "5日主力净额合计" in out and "亿" in out

    def test_flow_lookahead_cutoff(self, monkeypatch):
        import tradingagents.dataflows.money_flow as mf

        fake = mock.Mock()
        fake.stock_individual_fund_flow.return_value = self._flow_frame()
        monkeypatch.setattr(mf, "ak", fake)
        monkeypatch.setattr(mf, "get_config",
                            lambda: {"data_cache_dir": __import__("tempfile").mkdtemp()})
        import tradingagents.dataflows.global_macro as g
        out = g.get_money_flow.func("600519.SS", "2026-06-10", 30)
        assert "截至 2026-06-10" in out
        assert "2026-06-24" not in out

    def test_divergence_warning(self, monkeypatch):
        import tradingagents.dataflows.money_flow as mf

        frame = self._flow_frame()
        # 近3日价跌 + 资金净流入 → 吸筹信号分支
        frame.loc[frame.index[-3:], "收盘价"] = [100.0, 99.0, 98.0]
        fake = mock.Mock()
        fake.stock_individual_fund_flow.return_value = frame
        monkeypatch.setattr(mf, "ak", fake)
        monkeypatch.setattr(mf, "get_config",
                            lambda: {"data_cache_dir": __import__("tempfile").mkdtemp()})
        import tradingagents.dataflows.global_macro as g
        import tradingagents.dataflows.global_macro as g
        out = g.get_money_flow.func("600519.SS", "2026-06-24", 30)
        assert "逆向吸筹" in out

    def test_exposure_model_includes_mflow(self, monkeypatch):
        """EM 资金流可用时，暴露度模型应含主力资金因子行。"""
        import math

        dates = pd.bdate_range(end="2026-06-24", periods=260)
        gold = _daily_frame(dates, [100 + 10 * math.sin(i / 5) + 0.2 * i for i in range(len(dates))])
        crude = _daily_frame(dates, [50] * len(dates))
        yld = pd.DataFrame({"日期": [d.strftime("%Y-%m-%d") for d in dates],
                            "美国国债收益率2年": [4.0] * len(dates),
                            "美国国债收益率10年": [4.2] * len(dates)})
        spx = _daily_frame(dates, [5000] * len(dates))
        flow = pd.DataFrame({
            "日期": [d.date() for d in dates],
            "主力净流入-净占比": [2 + math.sin(i / 4) for i in range(len(dates))],
        })
        fake = mock.Mock()
        fake.futures_foreign_hist.side_effect = lambda *, symbol: {"GC": gold, "CL": crude}[symbol]
        fake.bond_zh_us_rate.return_value = yld
        fake.index_us_stock_sina.side_effect = lambda *, symbol: spx
        fake.stock_individual_fund_flow.return_value = flow
        monkeypatch.setattr(gm, "ak", fake)
        import tradingagents.dataflows.money_flow as mf
        monkeypatch.setattr(mf, "ak", fake)
        monkeypatch.setattr(mf, "get_config",
                            lambda: {"data_cache_dir": __import__("tempfile").mkdtemp()})

        target = pd.DataFrame({
            "Date": dates, "Open": 10.0, "High": 11.0, "Low": 9.0,
            "Close": [10 + 0.05 * math.sin(i / 4) for i in range(len(dates))],
            "Volume": 1e6,
        })
        with mock.patch.object(gm, "load_ohlcv", return_value=target):
            out = gm.get_factor_exposure.func("600519.SS", "2026-06-24", 120)
        assert "主力资金净占比" in out


class TestThsFallback:
    def test_ths_snapshot_fallback_when_em_blocked(self, monkeypatch):
        """EM 冷却期 → THS 榜单兜底返回单行快照。"""
        import tradingagents.dataflows.money_flow as mf

        # 触发冷却
        mf._em_fail_until = __import__("time").time() + 600

        ths = pd.DataFrame([
            {"股票代码": "600519", "股票简称": "贵州茅台", "最新价": 1290.0,
             "流入资金": "5.15亿", "流出资金": "3.65亿", "净额": "1.50亿",
             "成交额": "8.80亿"},
        ])
        fake = mock.Mock()
        fake.stock_fund_flow_individual.return_value = ths
        monkeypatch.setattr(mf, "ak", fake)
        monkeypatch.setattr(mf, "get_config",
                            lambda: {"data_cache_dir": __import__("tempfile").mkdtemp()})

        # anchor to "today" so the snapshot row (stamped Timestamp.today()
        # inside _ths_snapshot_frame) always passes the <= curr_date filter
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        df = mf.fetch_money_flow("600519.SS", today, 30, retries=1)
        assert len(df) == 1
        assert df["_source"].iloc[-1] == "ths_snapshot"
        assert abs(df["主力净流入-净额"].iloc[-1] - 1.5e8) < 1
        assert abs(df["主力净流入-净占比"].iloc[-1] - (1.5e8 / 8.8e8 * 100)) < 0.1

    def test_ths_miss_raises_no_data(self, monkeypatch):
        import tradingagents.dataflows.money_flow as mf

        mf._em_fail_until = __import__("time").time() + 600
        ths = pd.DataFrame([{"股票代码": "000001", "净额": "1亿", "成交额": "5亿",
                             "最新价": 12.0}])
        fake = mock.Mock()
        fake.stock_fund_flow_individual.return_value = ths
        monkeypatch.setattr(mf, "ak", fake)
        monkeypatch.setattr(mf, "get_config",
                            lambda: {"data_cache_dir": __import__("tempfile").mkdtemp()})
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        with pytest.raises(NoMarketDataError):
            mf.fetch_money_flow("600519.SS", today, 30, retries=1)


@pytest.mark.unit
class TestMacroToolDegradation:
    """Tool bodies must never raise through ToolNode: a NoMarketDataError from
    get_money_flow (EM throttled + THS snapshot miss on 000792.SZ) once killed
    a live run at the macro stage. Every macro tool now degrades to a sentinel
    the analyst is told to cite honestly."""

    def test_money_flow_degrades_on_no_market_data(self, monkeypatch):
        from tradingagents.dataflows import money_flow
        from tradingagents.dataflows.errors import NoMarketDataError

        monkeypatch.setattr(
            money_flow, "fetch_money_flow",
            mock.Mock(side_effect=NoMarketDataError("000792.SZ", "000792", "no rows")),
        )
        out = gm.get_money_flow.func("000792.SZ", "2026-09-01", 30)
        assert "数据暂不可用" in out and "get_money_flow" in out

    def test_gold_price_degrades_when_vendor_down(self, monkeypatch):
        fake = mock.Mock()
        fake.futures_foreign_hist.side_effect = RuntimeError("sina down")
        monkeypatch.setattr(gm, "ak", fake)
        out = gm.get_gold_price.func("2026-09-01", 30)
        assert "数据暂不可用" in out and "get_gold_price" in out

    def test_factor_exposure_degrades_without_ohlcv(self, monkeypatch):
        from tradingagents.dataflows.errors import NoMarketDataError

        _mock_ak(monkeypatch)
        with mock.patch.object(gm, "load_ohlcv",
                               side_effect=NoMarketDataError("x", "x", "none")):
            out = gm.get_factor_exposure.func("600519.SS", "2026-06-24", 120)
        assert "数据暂不可用" in out
