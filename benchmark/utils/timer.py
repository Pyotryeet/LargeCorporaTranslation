"""Precision wall-clock timer."""

import time


class PrecisionTimer:
    def __init__(self):
        self._start: float | None = None
        self._stop: float = 0.0

    def start(self) -> None:
        self._start = time.monotonic()

    def stop(self) -> None:
        self._stop = time.monotonic()

    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        if self._stop > self._start:
            return self._stop - self._start
        return time.monotonic() - self._start

    def start_time(self) -> float:
        return self._start
