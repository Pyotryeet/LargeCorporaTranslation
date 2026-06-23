"""Tests for rolling throughput calculator."""

import time
from benchmark.metrics.throughput import ThroughputTracker


class TestThroughputTracker:
    def test_empty_returns_zero(self):
        t = ThroughputTracker()
        assert t.current() == 0.0

    def test_single_event(self):
        t = ThroughputTracker(window_seconds=60)
        t.add(100, 1000)  # 100 tokens in 1000ms
        assert t._total_tokens == 100
        assert len(t._events) == 1

    def test_multiple_events(self):
        t = ThroughputTracker(window_seconds=60)
        t.add(50, 500)
        t.add(50, 500)
        assert t._total_tokens == 100
        assert len(t._events) == 2

    def test_summary(self):
        t = ThroughputTracker(window_seconds=60)
        t.add(100, 1000)
        s = t.summary()
        assert "rolling_tokens_per_second" in s
        assert s["total_tokens_produced"] == 100
        assert s["window_seconds"] == 60

    def test_snapshot(self):
        t = ThroughputTracker(window_seconds=60)
        t.add(100, 1000)
        snap = t.snapshot()
        assert snap.total_tokens == 100
        assert snap.window_seconds == 60

    # ── New tests for fix B9 (single-event no longer returns 0.0) ──

    def test_single_event_returns_nonzero_tps(self):
        """Fix P0-3: single event now returns tokens / effective_span, not 0.0."""
        t = ThroughputTracker(window_seconds=60)
        t.add(100, 1000)  # 100 tokens in ~1s
        tps = t.current()
        assert tps > 0.0, f"Single event tps should be > 0, got {tps}"

    def test_multiple_events_correct_rate(self):
        """Multiple recent events produce correct per-token throughput."""
        t = ThroughputTracker(window_seconds=60)
        # Add events with small sleeps between them to get realistic timestamps.
        t.add(50, 500)
        time.sleep(0.05)
        t.add(50, 500)
        time.sleep(0.05)
        t.add(100, 500)
        tps = t.current()
        assert tps > 0.0
        # 200 tokens over ~0.1s → ~2000 tok/s. Wide range to accommodate timing.
        assert 100 < tps < 10000, f"Expected 500-2000 tps, got {tps:.1f}"

    def test_single_event_clamped_to_window(self):
        """Single aged event uses its age as span (clamped to window)."""
        t = ThroughputTracker(window_seconds=60)
        t.add(100, 1000)
        # NOTE: This test directly mutates t._events[0] to simulate an aged event,
        # bypassing add(). It relies on the internal deque representation of events
        # as (timestamp, tokens) tuples. If the internal storage format changes
        # (e.g., namedtuple, dataclass, different field order), this test must be
        # updated accordingly.
        old_now = time.monotonic() - 10
        t._events[0] = (old_now, 100)
        tps = t.current()
        # 100 tokens / 10s ≈ 10 tps (not 100/0=inf or 0).
        assert 5 < tps < 20, f"Expected ~10 tok/s, got {tps:.1f}"

    def test_empty_tracker_summary(self):
        t = ThroughputTracker()
        s = t.summary()
        assert s["total_tokens_produced"] == 0
        assert s["events_in_window"] == 0
        assert s["rolling_tokens_per_second"] == 0.0
