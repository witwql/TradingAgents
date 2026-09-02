"""Screener engine tests: factors, probability blend, universe, top-N."""
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import server.screener as sc


def _hist(n=260, base=20.0, drift=0.02, seed=7):
    """Deterministic OHLCV with mild uptrend + sine wiggle (variance present)."""
    rng = np.random.default_rng(seed)
    closes = [base]
    for i in range(1, n):
        closes.append(closes[-1] * (1 + drift + 0.01 * np.sin(i / 4) + rng.normal(0, 0.004)))
    closes = np.array(closes)
    dates = pd.bdate_range(end="2026-06-24", periods=n)
    return pd.DataFrame({
        "Date": dates,
        "Open": closes * 0.995,
        "High": closes * 1.01,
        "Low": closes * 0.99,
        "Close": closes,
        "Volume": np.abs(rng.normal(5e6, 1e6, n)),
    })


@pytest.mark.unit
class TestFactorEngineTests:
    def test_all_factor_columns_present(self):
        d = sc.compute_factors(_hist())
        for col in sc.PRICE_FACTOR_COLUMNS:   # 价格五因子必在
            assert col in d.columns
        assert d["f1_pullback"].notna().all()
        assert "f6_money" not in d.columns    # f6 仅在注入资金流后出现

    def test_factor_stats_sample_floor(self):
        d = sc.compute_factors(_hist())
        stats = sc._factor_stats(d)
        for _col, (p, n) in stats.items():
            assert isinstance(n, int)
            if n >= sc.MIN_FACTOR_SAMPLES:
                assert 0.0 <= p <= 1.0

    def test_composite_requires_min_factors(self):
        d = sc.compute_factors(_hist())
        # Force today's factors off -> prob None
        for col in sc.FACTOR_COLUMNS:
            d[col] = False
        prob, contrib, fired = sc._composite_probability(d, sc._factor_stats(d))
        assert prob is None and fired == 0

    def test_probability_clamped(self):
        d = sc.compute_factors(_hist())
        stats = dict.fromkeys(sc.FACTOR_COLUMNS, (0.99, 400))
        d["f6_money"] = True
        for col in sc.FACTOR_COLUMNS:
            d[col] = True
        prob, _, fired = sc._composite_probability(d, stats)
        assert prob is not None and prob <= 0.95 and fired == len(sc.FACTOR_COLUMNS)

    def test_low_sample_factor_not_counted_toward_weight(self):
        d = sc.compute_factors(_hist())
        stats = dict.fromkeys(sc.FACTOR_COLUMNS, (0.9, 400))
        stats["f1_pullback"] = (0.9, 10)  # below floor
        d["f6_money"] = True
        for col in sc.FACTOR_COLUMNS:
            d[col] = True
        prob, contrib, fired = sc._composite_probability(d, stats)
        assert fired == len(sc.FACTOR_COLUMNS)
        unused = [c for c in contrib if c["factor"] == sc.FACTOR_LABELS["f1_pullback"]]
        assert unused and unused[0]["used"] is False


@pytest.mark.unit
class TestUniverseTests:
    def test_prefix_and_st_filtering(self, monkeypatch):
        spot = pd.DataFrame([
            {"代码": "sh600519", "名称": "贵州茅台", "最新价": 1290.0, "成交额": 3e9, "成交量": 2e6},
            {"代码": "sh688981", "名称": "中芯国际", "最新价": 90.0, "成交额": 5e9, "成交量": 3e6},   # 科创板
            {"代码": "sz300750", "名称": "宁德时代", "最新价": 260.0, "成交额": 6e9, "成交量": 4e6},  # 创业板
            {"代码": "sh601318", "名称": "ST平安", "最新价": 45.0, "成交额": 4e9, "成交量": 5e6},     # ST
            {"代码": "sz000001", "名称": "平安银行", "最新价": 1.2, "成交额": 9e8, "成交量": 9e7},    # 低价
            {"代码": "sz000002", "名称": "万科A", "最新价": 8.0, "成交额": 1e8, "成交量": 1e7},       # 成交额不足
            {"代码": "sh603259", "名称": "药明康德", "最新价": 65.0, "成交额": 2.5e9, "成交量": 3e7},
            {"代码": "bj920000", "名称": "安徽凤凰", "最新价": 13.7, "成交额": 3e8, "成交量": 2e6},   # 北交所
        ])
        import akshare as ak
        monkeypatch.setattr(ak, "stock_zh_a_spot", lambda: spot)
        monkeypatch.setattr(sc, "AKSHARE_LOCK", __import__("threading").Lock())

        universe = sc.fetch_universe()
        codes = [u["code"] for u in universe]
        assert codes == ["600519", "603259"]  # sorted by turnover desc

    def test_max_candidates_cap(self, monkeypatch):
        rows = [
            {"代码": f"sh60{i:04d}", "名称": f"股{i}", "最新价": 10.0,
             "成交额": 1e9 + i, "成交量": 1e6}
            for i in range(300)
        ]
        import akshare as ak
        monkeypatch.setattr(ak, "stock_zh_a_spot", lambda: pd.DataFrame(rows))
        monkeypatch.setattr(sc, "AKSHARE_LOCK", __import__("threading").Lock())
        assert len(sc.fetch_universe()) == sc.MAX_CANDIDATES


@pytest.mark.unit
class TestSelectionTests:
    def test_top_n_and_threshold(self):
        results = [
            {"code": f"60000{i}", "name": f"X{i}", "close": 10.0,
             "probability": p, "factors_fired": 4,
             "contributions": [], "resonance_hit_rate": 0.7,
             "resonance_samples": 9, "history_days": 500}
            for i, p in enumerate([0.91, 0.85, 0.82, 0.79, 0.95, 0.88, 0.81])
        ]
        qualifying = sorted(
            (r for r in results if r["probability"] >= sc.PROBABILITY_THRESHOLD),
            key=lambda x: x["probability"], reverse=True,
        )
        picks = qualifying[:sc.MAX_PICKS]
        assert [p["probability"] for p in picks] == [0.95, 0.91, 0.88, 0.85, 0.82]
        assert all(p["probability"] >= 0.80 for p in picks)

    def test_calibration_low_sample_flagged(self):
        d = sc.compute_factors(_hist())
        hit, n = sc._resonance_calibration(d)
        assert n == 0 or (hit is None) == (n < 5)


@pytest.mark.unit
class TestEvaluateStockIntegration:
    def test_evaluate_stock_with_mocked_history(self, monkeypatch):
        # Engineered resonance: mild uptrend (MA stack + MACD + RSI in range)
        # then a last-day +2.5% volume-spike break of the prior 20-day high.
        n = 260
        rng = np.random.default_rng(0)
        closes = [20.0]
        for i in range(1, n - 1):
            closes.append(closes[-1] * (1 + 0.001 + 0.008 * np.sin(i / 3) + rng.normal(0, 0.003)))
        closes.append(closes[-1] * 1.025)  # breakout day
        closes = np.array(closes)
        dates = pd.bdate_range(end="2026-06-24", periods=n)
        volumes = np.abs(rng.normal(5e6, 5e5, n))
        volumes[-6:-1] = 1.0e6          # thin week before the spike
        volumes[-1] = 3.0e6             # breakout volume
        frame = pd.DataFrame({
            "Date": dates.astype(str),
            "Open": closes * 0.998, "High": closes * 1.004,
            "Low": closes * 0.996, "Close": closes,
            "Volume": volumes,
        })

        import tradingagents.dataflows.sina_stock as ss
        monkeypatch.setattr(ss, "fetch_daily_ohlcv_sina", lambda *a, **k: frame)
        result = sc.evaluate_stock("600519", "贵州茅台", float(closes[-1]), "2026-06-24")
        assert result is not None, "engineered resonance must produce a pick"
        assert result["code"] == "600519"
        assert 0.05 <= result["probability"] <= 0.95
        assert result["factors_fired"] >= 3
        assert result["history_days"] == n

    def test_unusable_history_returns_none(self, monkeypatch):
        import tradingagents.dataflows.sina_stock as ss

        def bad_fetch(*a, **k):
            from tradingagents.dataflows.errors import NoMarketDataError
            raise NoMarketDataError("600519", "600519", "no rows")

        monkeypatch.setattr(ss, "fetch_daily_ohlcv_sina", bad_fetch)
        assert sc.evaluate_stock("600519", "贵州茅台", 1290.0, "2026-06-24") is None


class TestLatestRunResilience:
    def test_corrupt_results_payload_does_not_raise(self, tmp_path):
        from server.db import Database
        from server.screener import latest_run

        db = Database(tmp_path / "s.db")
        db.execute(
            "INSERT INTO screen_runs (id, created_at, status, trade_date, results)"
            " VALUES ('bad', 1, 'done', '2026-08-28', ?)",
            ("{invalid json with True}",),
        )
        run = latest_run(db)
        assert run["id"] == "bad"
        assert run["results"] is None

    def test_valid_results_are_parsed(self, tmp_path):
        import json as _json

        from server.db import Database
        from server.screener import latest_run

        db = Database(tmp_path / "s.db")
        db.execute(
            "INSERT INTO screen_runs (id, created_at, status, trade_date, results)"
            " VALUES ('ok', 1, 'done', '2026-08-28', ?)",
            (_json.dumps({"evaluated": 1, "qualifying": 0, "picks": [], "watchlist": []}),),
        )
        run = latest_run(db)
        assert run["results"]["evaluated"] == 1


class TestScreenRunAtomicity:
    """只验证 DB 原子语义；线程体 mock 掉——否则后台线程会真实拉网络，
    并通过模块级 ss.ak 泄漏 mock 计数污染后续测试（复盘实测踩中）。"""

    @pytest.fixture(autouse=True)
    def _no_worker(self, monkeypatch):
        monkeypatch.setattr(sc, "_run_blocking", lambda *a, **k: None)

    def test_double_start_reuses_running_run(self, tmp_path):
        from server.db import Database
        from server.screener import run_screening

        db = Database(tmp_path / "s.db")
        id1, reused1 = run_screening(db)
        id2, reused2 = run_screening(db)
        assert reused1 is False and reused2 is True
        assert id1 == id2

    def test_after_done_new_run_spawns(self, tmp_path):
        from server.db import Database
        from server.screener import run_screening

        db = Database(tmp_path / "s.db")
        id1, _ = run_screening(db)
        db.execute("UPDATE screen_runs SET status='done' WHERE id=?", (id1,))
        id2, reused = run_screening(db)
        assert reused is False and id2 != id1


class TestMoneyFlowFactor:
    def _flow(self, dates, pct, amt):
        import pandas as pd

        return pd.DataFrame({
            "_d": pd.to_datetime(dates),
            "主力净流入-净占比": pct,
            "主力净流入-净额": amt,
        })

    def test_append_factor_fires_on_inflow_streak(self):
        frame = sc.compute_factors(_hist())
        dates = pd.to_datetime(frame["Date"])
        n = len(frame)
        flow = self._flow(dates, [2.0] * n, [1e8] * n)
        d = sc.append_money_flow_factor(frame, flow)
        assert d["f6_money"].iloc[-1] is True or bool(d["f6_money"].iloc[-1])
        stats = sc._factor_stats(d)
        assert "f6_money" in stats

    def test_flow_missing_leaves_factor_absent(self):
        frame = sc.compute_factors(_hist())
        d = sc.append_money_flow_factor(frame, None)
        assert "f6_money" not in d.columns
        stats = sc._factor_stats(d)
        assert "f6_money" not in stats

    def test_outflow_days_do_not_fire(self):
        frame = sc.compute_factors(_hist())
        dates = pd.to_datetime(frame["Date"])
        flow = self._flow(dates, [-2.0] * len(dates), [-1e8] * len(dates))
        d = sc.append_money_flow_factor(frame, flow)
        assert not bool(d["f6_money"].iloc[-1])

    def test_evaluate_stock_uses_flow_factor(self, monkeypatch):
        # 历史 + 资金流双 mock：f6 应出现在 contributions 中。
        n = 260
        rng = np.random.default_rng(0)
        closes = [20.0]
        for i in range(1, n - 1):
            closes.append(closes[-1] * (1 + 0.001 + 0.008 * np.sin(i / 3) + rng.normal(0, 0.003)))
        closes.append(closes[-1] * 1.025)
        closes = np.array(closes)
        dates = pd.bdate_range(end="2026-06-24", periods=n)
        frame = pd.DataFrame({
            "Date": dates.astype(str), "Open": closes * 0.998,
            "High": closes * 1.004, "Low": closes * 0.996,
            "Close": closes, "Volume": np.abs(rng.normal(5e6, 5e5, n)),
        })
        flow = pd.DataFrame({
            "_d": dates,
            "主力净流入-净占比": [2.0] * n,
            "主力净流入-净额": [1e8] * n,
        })
        import tradingagents.dataflows.money_flow as mf
        import tradingagents.dataflows.sina_stock as ss
        monkeypatch.setattr(ss, "fetch_daily_ohlcv_sina", lambda *a, **k: frame)
        monkeypatch.setattr(mf, "fetch_money_flow", lambda *a, **k: flow)
        result = sc.evaluate_stock("600519", "贵州茅台", float(closes[-1]), "2026-06-24")
        assert result is not None
        fired = [c["factor"] for c in result["contributions"] if c["fired"]]
        assert "主力资金净流入配合" in fired


class TestScreeningScheduler:
    """注入时钟与触发器，纯逻辑验证。"""

    WED = __import__("datetime").datetime(2026, 8, 26, 15, 31)   # 周三
    WED_NOON = __import__("datetime").datetime(2026, 8, 26, 12, 0)
    SAT = __import__("datetime").datetime(2026, 8, 29, 16, 0)

    def _make(self, settings, now):
        from server.scheduler import ScreeningScheduler

        fired = []

        def trigger(*a, **k):
            fired.append(a)
            return ("id", False)

        db = mock.Mock()
        db.get_settings.return_value = settings
        db.fetchone.return_value = None
        s = ScreeningScheduler(db, now_fn=lambda: now, trigger=trigger)
        return s, fired

    def test_fires_once_after_target_on_weekday(self):
        s, fired = self._make({"auto_screen_time": "15:30"}, self.WED)
        s._tick()
        assert len(fired) == 1
        s._tick()
        assert len(fired) == 1  # 当日只触发一次

    def test_before_target_does_not_fire(self):
        s, fired = self._make({"auto_screen_time": "15:30"}, self.WED_NOON)
        s._tick()
        assert fired == []

    def test_off_disables(self):
        s, fired = self._make({"auto_screen_time": "off"}, self.WED)
        s._tick()
        assert fired == []

    def test_weekend_does_not_fire(self):
        s, fired = self._make({"auto_screen_time": "15:30"}, self.SAT)
        s._tick()
        assert fired == []

    def test_db_guard_skips_when_today_run_exists(self):
        from server.scheduler import ScreeningScheduler

        db = mock.Mock()
        db.get_settings.return_value = {"auto_screen_time": "15:30"}
        db.fetchone.return_value = {"id": "today-run"}   # 今日已有运行
        s = ScreeningScheduler(db, now_fn=lambda: self.WED,
                               trigger=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fire")))
        s._tick(lambda *a, **k: None)
        assert s._last_fired_date == "2026-08-26"


class TestInterruptedSweepAndCancel:
    def test_sweep_fails_stale_running_rows(self, tmp_path):
        import time as _t

        from server.db import Database

        db = Database(tmp_path / "s.db")
        # 屏幕运行：旧的 running 应被清扫，新的保留
        db.execute(
            "INSERT INTO screen_runs (id, created_at, status) VALUES ('old', ?, 'running')",
            (_t.time() - 7200,),
        )
        db.execute(
            "INSERT INTO screen_runs (id, created_at, status) VALUES ('fresh', ?, 'running')",
            (_t.time(),),
        )
        # 分析任务：新旧 running 都要清扫 —— 重启后没有任何 worker 持有
        # running 行，年龄过滤会让刚重启时的幽灵任务永远挂在 UI 上
        # （事故：任务挂 27 分钟后用户重启服务，任务当时"太新"未被清扫）。
        tid_old = db.create_task({"ticker": "600519", "trade_date": "2026-08-28"})
        db.update_task(tid_old, status="running", started_at=_t.time() - 7200)
        tid_fresh = db.create_task({"ticker": "600938", "trade_date": "2026-09-02"})
        db.update_task(tid_fresh, status="running", started_at=_t.time() - 60)

        counts = db.sweep_interrupted(max_running_age=1800)
        assert counts["screen_runs"] == 1
        assert counts["tasks"] == 2
        assert db.fetchone("SELECT status FROM screen_runs WHERE id='old'")["status"] == "failed"
        assert db.fetchone("SELECT status FROM screen_runs WHERE id='fresh'")["status"] == "running"
        assert db.get_task(tid_old)["status"] == "failed"
        assert db.get_task(tid_fresh)["status"] == "failed"

    def test_cancel_flag_request(self, tmp_path):
        from server.db import Database

        db = Database(tmp_path / "s.db")
        db.execute(
            "INSERT INTO screen_runs (id, created_at, status) VALUES ('r1', ?, 'running')",
            (__import__("time").time(),),
        )
        assert db.request_screen_cancel("r1") is True
        assert db.fetchone("SELECT cancel_requested FROM screen_runs WHERE id='r1'")["cancel_requested"] == 1
        # 已结束的运行不可取消
        db.execute("UPDATE screen_runs SET status='done' WHERE id='r1'")
        assert db.request_screen_cancel("r1") is False

    def test_run_blocking_honours_cancel_flag(self, tmp_path, monkeypatch):
        """预置取消标志 → 线程体立即终止为 cancelled。"""
        import time as _t

        import server.screener as sc
        from server.db import Database

        db = Database(tmp_path / "s.db")
        run_id = "rc1"
        db.execute(
            "INSERT INTO screen_runs (id, created_at, status, trade_date, stage, cancel_requested)"
            " VALUES (?, ?, 'running', '2026-08-28', 'analyzing', 1)",
            (run_id, _t.time()),
        )
        monkeypatch.setattr(sc, "fetch_universe", lambda: [
            {"code": "600519", "name": "贵州茅台", "price": 1290.0, "turnover": 3e9}])
        monkeypatch.setattr(sc, "evaluate_stock", lambda *a, **k: None)  # never needed

        sc._run_blocking(db, run_id, "2026-08-28")
        row = db.fetchone("SELECT status, results FROM screen_runs WHERE id=?", (run_id,))
        assert row["status"] == "cancelled"
        assert '"cancelled": true' in row["results"]


class TestFuturesFactor:
    """F7 相关期货利多脉冲: mapped futures day-move ≥ +1% fires the factor."""

    @staticmethod
    def _frame(n=260):
        # Same engineered resonance as TestEvaluateStockIntegration: mild
        # uptrend then a last-day breakout, so a composite actually forms.
        rng = np.random.default_rng(0)
        closes = [20.0]
        for i in range(1, n - 1):
            closes.append(closes[-1] * (1 + 0.001 + 0.008 * np.sin(i / 3) + rng.normal(0, 0.003)))
        closes.append(closes[-1] * 1.025)
        closes = np.array(closes)
        dates = pd.bdate_range(end="2026-06-24", periods=n)
        volumes = np.abs(rng.normal(5e6, 5e5, n))
        volumes[-6:-1] = 1.0e6
        volumes[-1] = 3.0e6
        return pd.DataFrame({
            "Date": dates.astype(str),
            "Open": closes * 0.998, "High": closes * 1.004,
            "Low": closes * 0.996, "Close": closes,
            "Volume": volumes,
        })

    @staticmethod
    def _fut(dates, closes):
        return {"AL0": {"name": "沪铝", "kind": "domestic",
                        "close": pd.Series(closes, index=dates),
                        "moves": {"1d": 2.0, "5d": 3.0, "20d": 4.0}}}

    def test_f7_fires_on_pulse_day_only(self):
        d = sc.compute_factors(self._frame())
        dates = pd.bdate_range(end="2026-06-24", periods=120)
        closes = [100.0] * 119 + [102.0]           # +2% on the last bar only
        out = sc.append_futures_factor(d, self._fut(dates, closes))
        assert bool(out["f7_futures"].iloc[-1])
        assert not bool(out["f7_futures"].iloc[-2])
        assert not bool(out["f7_futures"].iloc[0])  # pre-history days → False

    def test_f7_absent_without_snapshot(self):
        d = sc.compute_factors(self._frame())
        assert "f7_futures" not in sc.append_futures_factor(d, {}).columns
        assert "f7_futures" not in sc.append_futures_factor(d, None).columns

    def test_evaluate_stock_annotates_and_appends_factor(self, monkeypatch):
        """云铝股份 + AL0 snapshot: futures_context attached, f7 in the table."""
        frame = self._frame()
        import tradingagents.dataflows.sina_stock as ss
        monkeypatch.setattr(ss, "fetch_daily_ohlcv_sina", lambda *a, **k: frame)
        # keep the money-flow leg off the network; f6 simply stays absent
        monkeypatch.setattr("tradingagents.dataflows.money_flow.fetch_money_flow",
                            mock.Mock(side_effect=RuntimeError("skip")))

        dates = pd.to_datetime(frame["Date"])
        closes = [100.0] * (len(dates) - 1) + [102.5]  # pulse on the last day
        result = sc.evaluate_stock("000807", "云铝股份", 12.0, "2026-06-24",
                                   futures=self._fut(dates, closes))
        assert result is not None
        assert result["futures_context"] and "沪铝" in result["futures_context"]
        assert any(c["factor"].startswith("相关期货") for c in result["contributions"])
        assert result["factors_available"] == 6   # 5 price factors + f7 (f6 off)

    def test_evaluate_stock_without_mapping_stays_clean(self, monkeypatch):
        frame = self._frame()
        import tradingagents.dataflows.sina_stock as ss
        monkeypatch.setattr(ss, "fetch_daily_ohlcv_sina", lambda *a, **k: frame)

        monkeypatch.setattr("tradingagents.dataflows.money_flow.fetch_money_flow",
                            mock.Mock(side_effect=RuntimeError("skip")))
        dates = pd.to_datetime(frame["Date"])
        fut = self._fut(dates, [100.0] * len(dates))
        result = sc.evaluate_stock("600519", "贵州茅台", 12.0, "2026-06-24",
                                   futures=fut)
        assert result is not None
        assert result["futures_context"] is None
        assert not any(c["factor"].startswith("相关期货") for c in result["contributions"])
        assert result["factors_available"] == 5   # f6/f7 never appended
