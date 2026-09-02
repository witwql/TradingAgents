"""Futures board tests: Sina payload parsing, change baselines, board API.

All quote payloads are verbatim captures from hq.sinajs.cn (2026-09-01) so a
silent upstream format drift breaks the suite instead of the page.
"""
import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.db import Database
from server.futures import (
    FUTURES_UNIVERSE,
    FuturesBoard,
    fetch_futures_quotes,
    parse_quotes_payload,
)
from server.queue import TaskQueue

# --- verbatim Sina captures -------------------------------------------------

RAW_NF_RB0 = (
    'var hq_str_nf_RB0="螺纹钢连续,150000,3197.000,3200.000,3158.000,3174.000,'
    '3174.000,3175.000,3174.000,3175.000,3186.000,24,129,1290303.000,710896,沪,'
    '螺纹钢,2026-09-01,1,,,,,,,,,3175.669,0.000,0,0.000,0,0.000,0,0.000,0,0.000,'
    '0,0.000,0,0.000,0";'
)
RAW_NF_IF0 = (
    'var hq_str_nf_IF0="4602.000,4617.800,4576.000,4597.200,59140,271657781.600,'
    '137290.000,4597.200,0.000,5059.600,4140.000,0.000,0.000,4601.600,4599.800,'
    '145044.000,4597.000,3,0.000,0,0.000,0,0.000,0,0.000,0,4597.800,1,0.000,0,'
    '0.000,0,0.000,0,0.000,0,2026-09-01,15:00:00,200,1,,,,,,,,,4593.469,'
    '沪深300指数期货连续";'
)
RAW_HF_CL = (
    'var hq_str_hf_CL="87.213,,87.200,87.210,87.240,86.130,15:54:06,85.760,'
    '86.310,0,2,1,2026-09-01,纽约原油,0";'
)
RAW_EMPTY = 'var hq_str_nf_WH0="";'


@pytest.mark.unit
class TestParseQuotesPayload:
    def test_commodity_fields_and_prev_settle_baseline(self):
        quotes = parse_quotes_payload(RAW_NF_RB0)
        assert set(quotes) == {"RB0"}
        q = quotes["RB0"]
        assert q["price"] == 3174.0
        assert q["prev_settle"] == 3186.0        # 昨结算，非昨收(3174)
        assert q["open"] == 3197.0
        assert q["high"] == 3200.0
        assert q["low"] == 3158.0
        assert q["open_interest"] == 1290303.0
        assert q["volume"] == 710896.0
        assert q["date"] == "2026-09-01"
        assert q["time"] == "15:00:00"

    def test_cffex_fields_use_prev_close_baseline(self):
        quotes = parse_quotes_payload(RAW_NF_IF0)
        q = quotes["IF0"]
        assert q["price"] == 4597.2
        assert q["prev_settle"] == 4601.6        # 昨收（中金所实时报文无昨结算）
        assert q["volume"] == 59140.0
        assert q["open_interest"] == 137290.0
        assert q["date"] == "2026-09-01"
        assert q["time"] == "15:00:00"

    def test_global_fields(self):
        quotes = parse_quotes_payload(RAW_HF_CL)
        q = quotes["CL"]
        assert q["price"] == 87.213
        assert q["prev_settle"] == 85.76
        assert q["open"] == 86.31
        assert q["high"] == 87.24
        assert q["low"] == 86.13
        assert q["date"] == "2026-09-01"

    def test_empty_payload_skipped(self):
        quotes = parse_quotes_payload(RAW_EMPTY)
        assert quotes == {}

    def test_mixed_payload(self):
        payload = "\n".join([RAW_NF_RB0, RAW_NF_IF0, RAW_HF_CL, RAW_EMPTY]) + "\n"
        quotes = parse_quotes_payload(payload)
        assert set(quotes) == {"RB0", "IF0", "CL"}


@pytest.mark.unit
class TestFetchFuturesQuotes:
    def test_nf_hf_prefix_and_decoding(self, monkeypatch):
        captured = {}

        class FakeResp:
            content = (RAW_NF_RB0 + "\n" + RAW_HF_CL).encode("gb18030")

            def raise_for_status(self):
                pass

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers")
            return FakeResp()

        monkeypatch.setattr("requests.get", fake_get)
        quotes = fetch_futures_quotes(["RB0", "CL"])
        assert "list=nf_RB0,hf_CL" in captured["url"]
        assert captured["headers"]["Referer"].startswith("https://finance.sina")
        assert set(quotes) == {"RB0", "CL"}


def _fake_quote_fetcher(symbols):
    """Two symbols with distinct moves + one stale symbol without a quote."""
    rows = {}
    for sym in symbols:
        if sym == "RB0":
            rows[sym] = {"price": 3174.0, "prev_settle": 3186.0, "open": 3197.0,
                         "high": 3200.0, "low": 3158.0, "volume": 710896.0,
                         "open_interest": 1290303.0, "date": "2026-09-01",
                         "time": "15:00:00"}
        elif sym == "CL":
            rows[sym] = {"price": 87.213, "prev_settle": 85.76, "open": 86.31,
                         "high": 87.24, "low": 86.13, "volume": None,
                         "open_interest": None, "date": "2026-09-01",
                         "time": "15:54:06"}
    return rows


def _fake_trend_fetcher(symbol, category):
    return [100.0 + i for i in range(40)]


def _board(**kwargs) -> FuturesBoard:
    defaults = {
        "quote_fetcher": _fake_quote_fetcher,
        "trend_fetcher": _fake_trend_fetcher,
        "quote_ttl": 3600,
        "trend_ttl": 3600,
    }
    defaults.update(kwargs)
    return FuturesBoard(**defaults)


@pytest.mark.unit
class TestFuturesBoard:
    def test_snapshot_rows_and_change_math(self):
        board = _board()
        board._refresh_trends()  # 生产环境由后台线程执行；测试同步驱动
        snap = board.snapshot()
        assert len(snap["contracts"]) == len(FUTURES_UNIVERSE)
        by_sym = {c["symbol"]: c for c in snap["contracts"]}
        rb = by_sym["RB0"]
        assert rb["name"] == "螺纹钢"
        assert rb["category_label"] == "商品"
        assert rb["change"] == pytest.approx(-12.0)
        assert rb["pct"] == pytest.approx(-12.0 / 3186.0 * 100, abs=1e-9)
        assert rb["trend"] == _fake_trend_fetcher("RB0", "commodity")
        cl = by_sym["CL"]
        assert cl["unit"] == "美元/桶"
        assert cl["pct"] == pytest.approx((87.213 - 85.76) / 85.76 * 100, abs=1e-9)
        assert snap["quotes_ready"] is True
        assert snap["trend_ready"] is True
        assert snap["trend_count"] == len(FUTURES_UNIVERSE)

    def test_missing_quote_leaves_nulls_not_crash(self):
        board = _board(quote_fetcher=lambda symbols: {})
        board._refresh_trends()
        snap = board.snapshot()
        rb = next(c for c in snap["contracts"] if c["symbol"] == "RB0")
        assert rb["price"] is None and rb["pct"] is None
        assert snap["quotes_ready"] is False

    def test_fetch_failure_is_swallowed(self):
        def boom(symbols):
            raise RuntimeError("network down")

        board = _board(quote_fetcher=boom)
        board._refresh_trends()
        snap = board.snapshot()
        assert snap["quotes_ready"] is False
        assert len(snap["contracts"]) == len(FUTURES_UNIVERSE)

    def test_quote_ttl_gates_refetch(self):
        calls = []

        def counting(symbols):
            calls.append(1)
            return _fake_quote_fetcher(symbols)

        board = _board(quote_fetcher=counting, quote_ttl=3600)
        board.snapshot()
        board.snapshot()
        assert len(calls) == 1


class _NoopRunner:
    def __init__(self, settings):
        pass


@pytest.mark.unit
class TestFuturesEndpoint:
    def test_get_futures(self, tmp_path):
        db = Database(tmp_path / "fut.db")
        app = create_app(db=db, queue=TaskQueue(db, runner_cls=_NoopRunner),
                         start_spot=False, start_futures=False)
        board = _board()
        board._refresh_trends()
        app.state.futures = board
        client = TestClient(app)
        resp = client.get("/api/futures")
        assert resp.status_code == 200
        data = resp.json()
        assert data["quotes_ready"] is True
        symbols = {c["symbol"] for c in data["contracts"]}
        assert {"RB0", "IF0", "CL"} <= symbols
        rb = next(c for c in data["contracts"] if c["symbol"] == "RB0")
        assert rb["pct"] is not None and rb["trend"]
