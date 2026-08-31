import contextvars
from copy import deepcopy

import tradingagents.default_config as default_config

# Use default config but allow it to be overridden
_config: dict | None = None

# Per-run scope: set_config mutates process-global state, which is unsafe the
# moment anything else (screener thread, spot cache, a future second worker)
# reads config concurrently. Scoped callers get an isolated merged view for
# the duration of the context; everything else keeps seeing the global.
_scope: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "tradingagents_config_scope", default=None
)


class config_scope:
    """with config_scope(cfg): ... — vendor calls inside see exactly ``cfg``.

    ``cfg`` must be a COMPLETE configuration (callers pass the same dict they
    used to hand to set_config). Nested scopes replace the outer one; on exit
    the previous scope (or global) is restored.
    """

    def __init__(self, config: dict):
        self._config = deepcopy(config)
        self._token = None

    def __enter__(self):
        self._token = _scope.set(self._config)
        return self

    def __exit__(self, *exc):
        _scope.reset(self._token)
        return False


def initialize_config():
    """Initialize the configuration with default values."""
    global _config
    if _config is None:
        _config = deepcopy(default_config.DEFAULT_CONFIG)


def set_config(config: dict):
    """Update the configuration with custom values.

    Scope-aware: when called inside a ``config_scope`` (the dashboard's
    per-run path), updates apply to the scoped view only and the process
    global stays untouched — this is what makes parallel task workers safe.
    Outside any scope (CLI, programmatic use) the global is updated, exactly
    as before.

    Dict-valued keys (e.g. ``data_vendors``) are merged one level deep so a
    partial update like ``{"data_vendors": {"core_stock_apis": "alpha_vantage"}}``
    keeps the other nested keys from the default; scalar keys are replaced.
    """
    incoming = deepcopy(config)
    scoped = _scope.get()
    if scoped is not None:
        _merge_into(scoped, incoming)
        return
    global _config
    initialize_config()
    _merge_into(_config, incoming)


def _merge_into(target: dict, incoming: dict) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key].update(value)
        else:
            target[key] = value


def get_config() -> dict:
    """Get the current configuration (scoped view wins over global)."""
    scoped = _scope.get()
    if scoped is not None:
        return deepcopy(scoped)
    if _config is None:
        initialize_config()
    return deepcopy(_config)


# Initialize with default config
initialize_config()
