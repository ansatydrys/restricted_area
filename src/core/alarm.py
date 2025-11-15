from __future__ import annotations

import time


class AlarmController:
    """Manages alarm activation and cooldown."""

    def __init__(self, cooldown_seconds: float = 3.0) -> None:
        self._cooldown = cooldown_seconds
        self._active = False
        self._last_trigger = 0.0

    @property
    def active(self) -> bool:
        return self._active

    def update(self, intrusion_detected: bool) -> bool:
        now = time.monotonic()
        if intrusion_detected:
            self._active = True
            self._last_trigger = now
            return True

        if self._active and now - self._last_trigger >= self._cooldown:
            self._active = False
        return self._active



