"""Precision wall-clock timer."""

import time
from typing import Optional


class PrecisionTimer:
    def __init__(self):
        self._start: float = -1.0
        self._stop: float = 0.0

    def start(self) -> None:
        self._start = time.monotonic()

    def stop(self) -> None:
        self._stop = time.monotonic()

    def elapsed(self) -> float:
        """Return the elapsed wall-clock time in seconds.

        If start() has not been called, returns 0.0.
        If stop() has been called, returns the duration between start and stop.
        Otherwise returns the time since start().
        """
        if self._start < 0:
            return 0.0
        if self._stop > self._start:
            return self._stop - self._start
        return time.monotonic() - self._start

    def start_time(self) -> float:
        """Return the wall-clock time when start() was called.

        Raises RuntimeError if start() has not been called yet.
        """
        if self._start < 0:
            raise RuntimeError("start() must be called before start_time()")
        return self._start
