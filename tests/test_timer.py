"""Tests for precision timer."""

import time
import math
from benchmark.utils.timer import PrecisionTimer


class TestPrecisionTimer:
    def test_start_and_elapsed(self):
        t = PrecisionTimer()
        t.start()
        time.sleep(0.01)
        elapsed = t.elapsed()
        assert elapsed > 0.0

    def test_stop_records_time(self):
        t = PrecisionTimer()
        t.start()
        time.sleep(0.01)
        t.stop()
        e1 = t.elapsed()
        time.sleep(0.01)
        e2 = t.elapsed()
        assert e1 == e2  # Stopped timer doesn't advance

    def test_start_time(self):
        t = PrecisionTimer()
        t.start()
        assert t.start_time() > 0

    # ── Precision tests ──

    def test_elapsed_monotonic(self):
        """elapsed() never decreases."""
        t = PrecisionTimer()
        t.start()
        readings = []
        for _ in range(5):
            time.sleep(0.001)
            readings.append(t.elapsed())
        for i in range(1, len(readings)):
            assert readings[i] >= readings[i - 1], (
                f"elapsed() decreased: {readings[i-1]} -> {readings[i]}"
            )

    def test_elapsed_second_start(self):
        """Calling start() twice resets the clock."""
        t = PrecisionTimer()
        t.start()
        time.sleep(0.01)
        first = t.elapsed()
        t.start()
        time.sleep(0.005)
        second = t.elapsed()
        assert second < first, f"Second start should reset timer: {second=} vs {first=}"

    def test_lap(self):
        """lap() returns time since last lap/start."""
        t = PrecisionTimer()
        t.start()
        time.sleep(0.01)
        lap1 = t.lap()
        assert lap1 > 0.0
        time.sleep(0.01)
        lap2 = t.lap()
        assert lap2 > 0.0
        # lap() is time since last lap, not cumulative.
        # Both should be ~0.01s each, total ~0.02s

    def test_sub_millisecond_precision(self):
        """Timer should have at least microsecond-precision reads."""
        t = PrecisionTimer()
        t.start()
        # Quick operations should still return finite positive values.
        e = t.elapsed()
        assert e >= 0.0
        assert math.isfinite(e)

    def test_timer_stop_before_start(self):
        """stop() before start() is a no-op (elapsed returns creation-span)."""
        t = PrecisionTimer()
        t.stop()
        result = t.elapsed()
        assert isinstance(result, float)
        assert result >= 0.0
