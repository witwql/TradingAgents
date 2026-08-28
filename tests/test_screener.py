"""Screener engine tests: factors, probability blend, universe, top-N."""
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
        for col in sc.FACTOR_COLUMNS:
            assert col in d.columns
        assert d["f1_pullback"].notna().all()

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
        for col in sc.FACTOR_COLUMNS:
            d[col] = True
        prob, _, fired = sc._composite_probability(d, stats)
        assert prob is not None and prob <= 0.95 and fired == 5

    def test_low_sample_factor_not_counted_toward_weight(self):
        d = sc.compute_factors(_hist())
        stats = dict.fromkeys(sc.FACTOR_COLUMNS, (0.9, 400))
        stats["f1_pullback"] = (0.9, 10)  # below floor
        for col in sc.FACTOR_COLUMNS:
            d[col] = True
        prob, contrib, fired = sc._composite_probability(d, stats)
        assert fired == 5
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
