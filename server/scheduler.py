"""Trading-day scheduler for automatic screening runs.

A daemon thread wakes every minute and fires the screener once per trading
day when the wall clock passes the configured time (default 15:30, i.e.
shortly after A-share close). Weekday-only: a full CN holiday calendar would
add a dependency, and running on a holiday is harmless (the model just
re-evaluates the latest session). Settings live in the dashboard's settings
table (`auto_screen_time`), read fresh each cycle so UI changes apply
without restart; "off" disables the schedule entirely.
"""

import datetime as dt
import logging
import threading

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 60


class ScreeningScheduler:
    def __init__(self, db, now_fn=None, trigger=None):
        import datetime as dt

        self.db = db
        self._now = now_fn or dt.datetime.now
        self._trigger = trigger
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_fired_date: str | None = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="screen-scheduler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        from .screener import run_screening

        trigger = self._trigger or run_screening
        while not self._stop.is_set():
            try:
                self._tick(trigger)
            except Exception:
                logger.exception("screening scheduler tick failed")
            self._stop.wait(CHECK_INTERVAL_SECONDS)

    def _tick(self, trigger=None):
        from .screener import run_screening

        trigger = trigger or self._trigger or run_screening
        setting = (self.db.get_settings().get("auto_screen_time") or "15:30").strip()
        if setting.lower() in ("", "off", "disabled"):
            return
        try:
            hh, mm = setting.split(":")
            target = dt.time(int(hh), int(mm))
        except (ValueError, AttributeError):
            logger.warning("invalid auto_screen_time %r; expected HH:MM or off", setting)
            return

        now = self._now()
        today = now.date()
        self._last_fired_date = self._last_fired_date  # keep explicit
        if now.weekday() >= 5:  # Sat/Sun
            return
        if now.time() < target:
            return
        if self._last_fired_date == today.isoformat():
            return

        # DB-level double-guard: skip when a run for today's date already
        # exists (covers restarts, where _last_fired_date resets).
        existing = self.db.fetchone(
            "SELECT id FROM screen_runs WHERE trade_date=? AND created_at > ? LIMIT 1",
            (today.isoformat(),
             dt.datetime.combine(today, target).timestamp()),
        )
        if existing:
            self._last_fired_date = today.isoformat()
            return

        run_id, already = trigger(self.db, today.isoformat())
        self._last_fired_date = today.isoformat()
        logger.info("scheduler: auto screening %s (%s)", run_id,
                    "reused" if already else "started")


__all__ = ["ScreeningScheduler"]
