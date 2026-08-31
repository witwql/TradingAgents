"""In-process event bus for SSE: replaces per-connection DB polling.

Producers (worker threads) call :meth:`EventBus.publish` after persisting an
event; async consumers (SSE handlers) subscribe per topic and await their own
queue. Thread→loop handoff uses ``call_soon_threadsafe`` so the asyncio loop
is never blocked from a worker thread.

The database remains the source of truth: subscribers replay persisted rows
first and deduplicate bus events by id, so a reconnect can never miss or
double-deliver.
"""

import asyncio
import logging
import threading

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self):
        self._lock = threading.Lock()
        # topic -> list[(queue, loop)]
        self._subs: dict[str, list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]]] = {}

    def publish(self, topic: str, event: dict) -> None:
        """Thread-safe fan-out. Safe to call before any subscriber exists."""
        with self._lock:
            subs = list(self._subs.get(topic, ()))
        for queue, loop in subs:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                # Subscriber's loop died between registration and publish;
                # drop silently — DB replay covers reconnects.
                self.unsubscribe(topic, queue)

    async def subscribe(self, topic: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subs.setdefault(topic, []).append((queue, loop))
        return queue

    def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subs.get(topic)
            if subs:
                self._subs[topic] = [(q, lp) for q, lp in subs if q is not queue]
                if not self._subs[topic]:
                    self._subs.pop(topic, None)

    def subscriber_count(self, topic: str) -> int:
        with self._lock:
            return len(self._subs.get(topic, ()))


__all__ = ["EventBus"]
