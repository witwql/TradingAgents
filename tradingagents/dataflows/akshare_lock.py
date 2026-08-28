"""Process-wide lock around AKShare invocations.

AKShare bundles ``py-mini-racer`` (a V8 engine) for a few endpoints; V8's
address pool is initialized once per process and is NOT thread-safe —
concurrent first-uses from the dashboard's worker thread and the spot-quote
daemon hard-abort the whole interpreter (``libmini_racer.dylib ... Check
failed: !pool->IsInitialized()``). Serializing every AKShare call through one
lock removes the race; the calls are I/O-bound so the cost is negligible.
"""

import threading

AKSHARE_LOCK = threading.RLock()


__all__ = ["AKSHARE_LOCK"]
