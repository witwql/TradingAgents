"""Configurable LLM SDK retry budget (#1090/#1091).

A single transient 429 burst used to kill an otherwise-healthy multi-agent run
because each provider SDK's max_retries (default 2) was not exposed. This adds an
opt-in llm_max_retries knob forwarded to every provider chat client.
"""
from __future__ import annotations

import importlib

import pytest

import tradingagents.default_config as default_config_module
from tradingagents.graph.trading_graph import TradingAgentsGraph, _coerce_max_retries

# --- coercion / validation -------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [(0, 0), (2, 2), (10, 10), ("6", 6)])
def test_coerce_accepts_non_negative_ints_and_numeric_strings(value, expected):
    assert _coerce_max_retries(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-1, "-3"])
def test_coerce_rejects_negative(bad):
    with pytest.raises(ValueError, match=">= 0"):
        _coerce_max_retries(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [True, False])
def test_coerce_rejects_booleans(bad):
    with pytest.raises(ValueError, match="boolean"):
        _coerce_max_retries(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["abc", "1.5", None])
def test_coerce_rejects_non_integers(bad):
    with pytest.raises(ValueError, match="integer"):
        _coerce_max_retries(bad)


# --- forwarding into provider kwargs --------------------------------------

def _bare_graph(config):
    g = object.__new__(TradingAgentsGraph)
    g.config = config
    return g


@pytest.mark.unit
def test_not_forwarded_when_unset():
    kwargs = _bare_graph({"llm_provider": "openai", "llm_max_retries": None})._get_provider_kwargs()
    assert "max_retries" not in kwargs


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_forwarded_across_providers(provider):
    kwargs = _bare_graph({"llm_provider": provider, "llm_max_retries": 6})._get_provider_kwargs()
    assert kwargs["max_retries"] == 6


@pytest.mark.unit
def test_forwarded_env_string_is_coerced():
    # env vars arrive as strings; the consumer coerces (like temperature)
    kwargs = _bare_graph({"llm_provider": "openai", "llm_max_retries": "4"})._get_provider_kwargs()
    assert kwargs["max_retries"] == 4


@pytest.mark.unit
def test_invalid_config_value_fails_loudly():
    with pytest.raises(ValueError):
        _bare_graph({"llm_provider": "openai", "llm_max_retries": -1})._get_provider_kwargs()


# --- env overlay -----------------------------------------------------------

def _reload_with_env(monkeypatch, **overrides):
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


@pytest.mark.unit
def test_default_is_one_retry(monkeypatch):
    # 1（而非 SDK 默认的 2）：SDK 重试完全静默，每多一次重试就多一倍
    # 节点无事件静默期（幽灵挂死事故：600s×3 ≈ 30 分钟无任何反馈）。
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["llm_max_retries"] == 1


@pytest.mark.unit
def test_env_override_sets_config(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_LLM_MAX_RETRIES="8")
    # int-default key: env coercion yields an int, still passed through
    # _coerce_max_retries downstream before reaching the client.
    assert dc.DEFAULT_CONFIG["llm_max_retries"] == 8
    assert _coerce_max_retries(dc.DEFAULT_CONFIG["llm_max_retries"]) == 8


# --- llm_timeout: per-request cap so a wedged connection raises (then the
# --- langgraph retry policy gets its chance) instead of stalling forever -----

@pytest.mark.unit
@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_timeout_forwarded_across_providers(provider):
    kwargs = _bare_graph({"llm_provider": provider, "llm_timeout": 300})._get_provider_kwargs()
    assert kwargs["timeout"] == 300.0


@pytest.mark.unit
@pytest.mark.parametrize("disabled", [None, 0, ""])
def test_timeout_not_forwarded_when_disabled(disabled):
    kwargs = _bare_graph({"llm_provider": "openai", "llm_timeout": disabled})._get_provider_kwargs()
    assert "timeout" not in kwargs


@pytest.mark.unit
def test_timeout_default_is_bounded(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    # 300s ≈ 3.5× 实测 glm-5.3-flash 分析级调用最慢耗时（~84s）；
    # 与 llm_max_retries=1 组合，单次调用的最坏静默期 ≈ 10 分钟。
    assert dc.DEFAULT_CONFIG["llm_timeout"] == 300


@pytest.mark.unit
def test_timeout_env_override(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_LLM_TIMEOUT="300")
    # int-default key: env coercion yields an int.
    assert dc.DEFAULT_CONFIG["llm_timeout"] == 300


# --- llm_streaming: OpenAI-compatible thinking models stream by default so a
# --- slow-but-healthy generation keeps bytes flowing and the timeout applies
# --- per chunk gap, not to a single silent multi-minute request -------------

@pytest.mark.unit
@pytest.mark.parametrize("provider", ["glm-cn", "glm", "openai", "deepseek"])
def test_streaming_default_on_for_openai_compatible(provider):
    kwargs = _bare_graph({"llm_provider": provider})._get_provider_kwargs()
    assert kwargs["streaming"] is True


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["anthropic", "google"])
def test_streaming_not_forced_on_native_clients(provider):
    kwargs = _bare_graph({"llm_provider": provider})._get_provider_kwargs()
    assert "streaming" not in kwargs


@pytest.mark.unit
def test_streaming_env_override_off(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_LLM_STREAMING="off")
    assert dc.DEFAULT_CONFIG["llm_streaming"] is False
    kwargs = _bare_graph({**dc.DEFAULT_CONFIG, "llm_provider": "glm-cn"})._get_provider_kwargs()
    assert "streaming" not in kwargs


# --- streaming total-duration cap: a degenerate reasoning loop keeps chunks
# --- flowing forever, so the per-chunk idle timeout never fires (observed:
# --- one risk-debate call streamed 33+ min). Bound the whole call. ----------

def _make_streaming_llm(request_timeout):
    from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI

    return NormalizedChatOpenAI(
        model="gpt-4o", api_key="sk-test", request_timeout=request_timeout
    )


@pytest.mark.unit
def test_stream_total_is_multiple_of_idle_timeout():
    llm = _make_streaming_llm(300)
    assert llm._stream_total_seconds() == 1200.0
    llm = _make_streaming_llm(600)
    assert llm._stream_total_seconds() == 2400.0


@pytest.mark.unit
def test_stream_total_falls_back_to_floor_without_timeout():
    llm = _make_streaming_llm(None)
    assert llm._stream_total_seconds() == 1200.0
    llm = _make_streaming_llm(0)
    assert llm._stream_total_seconds() == 1200.0


@pytest.mark.unit
def test_bounded_stream_passes_healthy_chunks_through():
    from tradingagents.llm_clients.openai_client import _bound_stream_seconds

    deadline = __import__("time").monotonic() + 60
    out = list(_bound_stream_seconds(iter(["a", "b", "c"]), deadline, 1200.0))
    assert out == ["a", "b", "c"]


@pytest.mark.unit
def test_bounded_stream_raises_past_deadline():
    import time as _time

    from tradingagents.llm_clients.openai_client import _bound_stream_seconds

    deadline = _time.monotonic() - 1  # already expired
    gen = _bound_stream_seconds(iter(["a"]), deadline, 1200.0)
    with pytest.raises(RuntimeError, match="total time budget"):
        next(gen)
