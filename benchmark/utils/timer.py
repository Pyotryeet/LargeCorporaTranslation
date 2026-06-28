"""Precision wall-clock timer using ``time.monotonic()``.

Provides the ``PrecisionTimer`` class for measuring elapsed wall-clock time
during benchmarking. Because it is backed by ``time.monotonic()``, measurements
are immune to system clock adjustments (NTP, DST, manual changes).

Typical usage::

    from benchmark.utils.timer import PrecisionTimer

    t = PrecisionTimer()
    t.start()
    ...  # work to measure
    t.stop()
    print(f"Took {t.elapsed():.3f}s")
"""

import time
from typing import Optional


class PrecisionTimer:
    """High-resolution wall-clock timer backed by ``time.monotonic()``.

    Measures elapsed time between a single ``start()`` and ``stop()`` call.
    While running (after ``start()`` but before ``stop()``), ``elapsed()`` returns
    the live duration. Once stopped, ``elapsed()`` freezes at the captured interval.

    ``time.monotonic()`` is immune to system clock adjustments (NTP, DST), so
    this timer is suitable for benchmarking even on machines with drifting clocks.

    Typical usage::

        t = PrecisionTimer()
        t.start()
        ...  # work to measure
        t.stop()
        print(f"Took {t.elapsed():.3f}s")
    """

    def __init__(self):
        """Initialize a stopped timer.

        Sets ``_start`` to ``-1.0`` (sentinel for "not started") and ``_stop`` to
        ``0.0``. The timer does not begin ticking until ``start()`` is called.
        """
        self._start: float = -1.0
        self._stop: float = 0.0

    def start(self) -> None:
        """Record the current wall-clock time as the start point.

        Overwrites any previous start/stop state. After this call, ``elapsed()``
        begins returning the live duration since this moment. Has no return value;
        the caller should use ``elapsed()`` or ``start_time()`` to retrieve the
        recorded time.
        """
        self._start = time.monotonic()

    def stop(self) -> None:
        """Record the current wall-clock time as the stop point.

        After this call, ``elapsed()`` returns the fixed duration between the most
        recent ``start()`` and this ``stop()``, rather than a live reading. Calling
        ``stop()`` without a prior ``start()`` records a timestamp but has no visible
        effect because ``elapsed()`` returns ``0.0`` when the timer was never started.
        """
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
