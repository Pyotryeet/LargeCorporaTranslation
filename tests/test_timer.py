"""Tests for precision timer."""

import time
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
