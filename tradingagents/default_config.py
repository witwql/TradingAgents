import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
    "TRADINGAGENTS_LLM_MAX_RETRIES":      "llm_max_retries",
    "TRADINGAGENTS_LLM_TIMEOUT":          "llm_timeout",
    "TRADINGAGENTS_LLM_STREAMING":        "llm_streaming",
    "TRADINGAGENTS_DISABLED_TOOLS":       "disabled_tools",
    "TRADINGAGENTS_ANALYST_MAX_TOOL_ROUNDS": "analyst_max_tool_rounds",
    # Provider-specific reasoning/thinking knobs (None = each provider's own
    # default). Settable here for non-interactive runs; the CLI also offers an
    # interactive choice, which is skipped when the matching var is set.
    "TRADINGAGENTS_GOOGLE_THINKING_LEVEL":   "google_thinking_level",
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "TRADINGAGENTS_ANTHROPIC_EFFORT":        "anthropic_effort",
}


_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value.

    Invalid values raise ``ValueError`` rather than silently falling back to a
    default — a misspelled boolean (e.g. ``treu``) or non-numeric int should fail
    loudly at startup, not quietly misconfigure an unattended run.
    """
    if isinstance(reference, bool):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE:
            return True
        if normalized in _BOOL_FALSE:
            return False
        raise ValueError(
            f"expected a boolean ({'/'.join(_BOOL_TRUE + _BOOL_FALSE)}), got {value!r}"
        )
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        try:
            config[key] = _coerce(raw, config.get(key))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_var}: {exc}") from exc
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings. GLM via Zhipu BigModel (open.bigmodel.cn, ZHIPU_CN_API_KEY)
    # is the project default; the international Z.AI endpoint is provider "glm"
    # with ZHIPU_API_KEY. Switch via TRADINGAGENTS_LLM_PROVIDER /
    # TRADINGAGENTS_DEEP_THINK_LLM / TRADINGAGENTS_QUICK_THINK_LLM.
    "llm_provider": "glm-cn",
    "deep_think_llm": "glm-5.3-flash",
    "quick_think_llm": "glm-5.3-flash",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Sampling temperature, forwarded to every provider when set. None leaves
    # each provider at its own default. Lower values reduce run-to-run
    # variation on models that honor it; reasoning models largely ignore it
    # and no setting makes LLM output bit-identical across runs (see README).
    "temperature": None,
    # SDK retry budget forwarded to every provider chat client. Pinned to 1:
    # SDK retries are silent (no callback, no event), so each retry multiplies
    # how long a wedged connection stalls a node with zero UI feedback. 600s
    # timeout × provider default 2 retries ≈ 30 min of silence per call — the
    # "analysis stuck for half an hour" incident. 1 retry still rides out a
    # bursty 429/5xx; raise via TRADINGAGENTS_LLM_MAX_RETRIES if your
    # deployment is heavily rate-limited (#1091).
    "llm_max_retries": 1,
    # Per-request LLM timeout in seconds, forwarded to every provider chat
    # client. Without it a wedged provider connection (request accepted,
    # response never sent) blocks the calling node forever: langgraph's node
    # retry only fires on exceptions, and with no timeout none ever comes.
    # With llm_streaming on (the default) this is a per-chunk idle timeout:
    # observed legitimate chunk gaps are ≤1.4s, so 180s leaves >100× headroom
    # while cutting wedge-recovery from 5min to 3min per attempt. 0/None
    # disables the cap.
    "llm_timeout": 180,
    # Stream LLM calls by default on the OpenAI-compatible family (GLM,
    # DeepSeek, Qwen, ...). Thinking models legitimately generate for 5-8 min;
    # non-streaming requests hold one silent connection the whole time and
    # gateways tend to drop exactly those, wedging the node (the "analysis
    # stuck for half an hour" incident). With streaming the llm_timeout above
    # applies per chunk gap instead of per request, so slow-but-healthy calls
    # pass while dead connections fail visibly. Set False to restore
    # buffered/non-streaming requests.
    "llm_streaming": True,
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Cap on tool-call rounds per analyst node (see
    # tradingagents/agents/utils/tool_availability.py). The analyst makes at
    # most this many LLM calls that carry tool_calls; the next call is made
    # without bound tools so the model must emit its final report. Bounding
    # what was previously limited only by recursion_limit=100 keeps a
    # data-thirsty analyst (observed: get_indicators ×8) from stretching the
    # analyst phase into tens of minutes. 0 disables the cap.
    "analyst_max_tool_rounds": 3,
    # Comma-separated tool names to unbind at graph-build time (e.g.
    # "get_prediction_markets,get_macro_indicators"). Gating at bind time —
    # instead of relying on the vendor layer to fail at call time — removes
    # the deterministic failure mode where an unconfigured vendor (no
    # FRED_API_KEY, polymarket host unreachable) burns a tool round-trip plus
    # an LLM digestion round per call. The analyst prompt is segmented so it
    # never advertises a bound tool.
    "disabled_tools": "",
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 20,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 10,      # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # Data vendor configuration
    # Category-level configuration (default for all tools in category).
    # The configured value is the exact vendor chain — requests are NOT silently
    # routed to vendors you didn't choose. For ordered fallback, list several,
    # e.g. "yfinance,alpha_vantage". "default" uses all available vendors.
    # For China A-shares the dashboard preset uses only unthrottled hosts for
    # prices/fundamentals (Sina first, Yahoo fallback); EastMoney is kept ONLY
    # for news (its search-api host is on different infra from the throttled
    # quote endpoints). Money flow degrades to a THS snapshot when EM is down.
    #   {"core_stock_apis": "sina,yfinance",
    #    "technical_indicators": "sina,yfinance",
    #    "fundamental_data": "sina,yfinance",
    #    "news_data": "akshare,yfinance"}
    # (akshare serves A-shares via EastMoney/THS, pip install
    # "tradingagents[akshare]", no API key)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: yfinance, alpha_vantage, akshare (A-share codes)
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance, akshare (A-share codes)
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance, akshare (A-share codes)
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance, akshare (A-share codes)
        "macro_data": "fred",                # Options: fred (needs FRED_API_KEY)
        "prediction_markets": "polymarket",  # Options: polymarket (keyless)
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Benchmark for alpha calculation in the reflection layer.
    # ``benchmark_ticker`` (when set) overrides the suffix map for all
    # tickers; leave it None to use ``benchmark_map`` for auto-detection
    # based on the ticker's exchange suffix. SPY remains the US default
    # so the reflection label keeps reading "Alpha vs SPY" for US tickers
    # while non-US tickers get their regional index automatically.
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS":  "^NSEI",       # NSE India (Nifty 50)
        ".BO":  "^BSESN",      # BSE India (Sensex)
        ".T":   "^N225",       # Tokyo (Nikkei 225)
        ".HK":  "^HSI",        # Hong Kong (Hang Seng)
        ".L":   "^FTSE",       # London (FTSE 100)
        ".TO":  "^GSPTSE",     # Toronto (TSX Composite)
        ".AX":  "^AXJO",       # Australia (ASX 200)
        ".SS":  "000001.SS",   # Shanghai (SSE Composite)
        ".SZ":  "399001.SZ",   # Shenzhen (SZSE Component)
        "":     "SPY",         # default for US-listed tickers (no suffix)
    },
})
