"""Shared fixtures: keep module-level caches from leaking between tests."""
import pytest


@pytest.fixture(autouse=True)
def _reset_money_flow_cooldown():
    try:
        from tradingagents.dataflows import money_flow

        money_flow.reset_em_cooldown()
    except Exception:
        pass
    yield
    try:
        money_flow.reset_em_cooldown()
    except Exception:
        pass
