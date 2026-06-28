"""Rolling throughput calculator with O(1) queries (P0-10).

Maintains a running token sum so ``current()`` is constant-time regardless
of window size.  Previously used ``sum(t for _, t in self._events)`` which
was O(n) per query and recomputed on every heartbeat.

Also tracks per-batch latency (wall-clock, not token-generation latency)
in a parallel deque so callers can retrieve latency percentiles without
adding a separate tracker.
"""

import time
import threading
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ThroughputSnapshot:
    """Immutable snapshot of throughput and latency at a point in time.

    Captured by ``ThroughputTracker.snapshot()`` for reporting or logging
    without holding the tracker lock.  All fields are populated by the
    tracker; consumers should treat this as read-only.

    Attributes:
        timestamp: Monotonic clock reading (seconds) when the snapshot was taken.
        tokens_per_second: Rolling throughput over the configured window, rounded
            to one decimal place.
        total_tokens: Cumulative tokens recorded since tracker creation.
        window_seconds: The tracker's configured time window for throughput
            calculation.
        p50_latency_ms: Median batch latency in milliseconds, or ``None`` if no
            latency data has been recorded.
        p99_latency_ms: 99th-percentile batch latency in milliseconds, or
            ``None`` if fewer than two data points exist.
    """
    timestamp: float
    tokens_per_second: float
    total_tokens: int
    window_seconds: float
    p50_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None


class ThroughputTracker:
    """Rolling-window throughput calculator with O(1) queries.

    Maintains a running token sum and a deque of ``(timestamp, token_count)``
    events.  Evicts events older than the configured window on every mutation
    so that ``current()`` is constant-time regardless of window depth.

    Also records per-batch wall-clock latency (from batch acceptance to
    completion) in a ring buffer for percentile retrieval.

    Thread-safe — all public methods acquire an internal lock.  The internal
    methods ``_prune``, ``_current_locked``, and ``_latency_percentiles``
    expect the caller to already hold ``self._lock``.

    Args:
        window_seconds: Size of the rolling time window in seconds.
            Must be positive.  Defaults to 60.0.

    Raises:
        ValueError: If ``window_seconds`` is not positive.
    """
    def __init__(self, window_seconds: float = 60.0):
        """Initialize a rolling throughput tracker.

        Args:
            window_seconds: Size of the rolling time window in seconds.
                Must be a positive float.  Events older than this are evicted
                from the window.  Defaults to 60.0.

        Raises:
            ValueError: If ``window_seconds <= 0``.
        """
        if window_seconds <= 0:
            raise ValueError(
                f"window_seconds must be positive, got {window_seconds}"
            )
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._events: deque[tuple[float, int]] = deque()
        self._total_tokens: int = 0
        self._window_sum: int = 0  # O(1) running sum of tokens in window
        # Per-batch latency ring buffer (wall-clock ms from accept to add).
        # Capped at 10 000 entries to bound memory; oldest are evicted.
        self._latencies: deque[float] = deque(maxlen=10_000)

    def add(self, tokens: int, latency_ms: Optional[float] = None) -> None:
        """Record a batch completion.

        Computes throughput from wall-clock timestamps rather than
        per-batch latency, but accepts an optional *latency_ms* for
        per-batch latency statistics (p50 / p99 in summary).
        """
        now = time.monotonic()
        with self._lock:
            self._events.append((now, tokens))
            self._total_tokens += tokens
            self._window_sum += tokens
            self._prune(now)
            if latency_ms is not None:
                self._latencies.append(latency_ms)

    def current(self) -> float:
        """O(1) throughput query — uses running sum, not recomputation.

        Returns tokens/second over the window (or event span for partially-full windows).
        Single-event case uses the event's age clamped to window_seconds rather
        than returning 0.0.
        """
        with self._lock:
            return self._current_locked(time.monotonic())

    def _current_locked(self, now: float) -> float:
        """Caller must hold ``self._lock``."""
        self._prune(now)
        assert self._window_sum >= 0, (
            f"Invariant violated: _window_sum={self._window_sum} "
            f"after _prune with now={now}, window={self.window_seconds}"
        )
        if not self._events:
            return 0.0

        # Single event: compute throughput as tokens / age.
        # DO NOT clamp age to window_seconds — that inflates TPS when
        # the single event is older than the window (e.g., 100 tokens at
        # t=200s would report 100/60=1.67 tok/s when actual is 0.5).
        if len(self._events) == 1:
            ts, tokens = self._events[0]
            age = now - ts
            if age <= 0:
                return 0.0
            return tokens / age

        # Multiple events: use the span covered by events (clamped to window).
        span = self._events[-1][0] - self._events[0][0]
        # Guard against events recorded in rapid succession (same acquire time).
        # A span of zero would divide by zero; clamp to 0.1s minimum.
        duration = span if span > 0.1 else 0.1
        if duration <= 0:
            return 0.0
        # When the span exceeds the window, use the window size (pruning ensures
        # events outside the window are removed, but span may be < window when
        # not yet fully populated).
        effective_duration = min(duration, self.window_seconds)
        return self._window_sum / effective_duration

    def snapshot(self) -> ThroughputSnapshot:
        """Capture a throughput and latency snapshot.

        Creates an immutable ``ThroughputSnapshot`` with the current rolling
        throughput, cumulative token count, window size, and latency percentiles
        (p50 / p99).

        Thread-safe — acquires ``self._lock``.

        Returns:
            ThroughputSnapshot: A dataclass instance populated with:
            - ``timestamp`` — monotonic time of capture.
            - ``tokens_per_second`` — current rolling throughput, rounded to 1 dp.
            - ``total_tokens`` — cumulative tokens across all batches.
            - ``window_seconds`` — the configured window size.
            - ``p50_latency_ms`` — median batch latency (or ``None``).
            - ``p99_latency_ms`` — 99th-percentile batch latency (or ``None``).
        """
        now = time.monotonic()
        with self._lock:
            lat_stats = self._latency_percentiles()
            return ThroughputSnapshot(
                timestamp=now,
                tokens_per_second=round(self._current_locked(now), 1),
                total_tokens=self._total_tokens,
                window_seconds=self.window_seconds,
                p50_latency_ms=lat_stats[0],
                p99_latency_ms=lat_stats[1],
            )

    def summary(self) -> dict:
        """Return a dictionary summary of current tracker state.

        Includes rolling throughput, cumulative token count, window size,
        event count, and latency percentiles (if available).  Suitable for
        JSON serialization in logs or monitoring systems.

        Thread-safe — acquires ``self._lock``.

        Returns:
            dict: Always contains:
            - ``"rolling_tokens_per_second"`` (float, rounded to 1 dp).
            - ``"total_tokens_produced"`` (int).
            - ``"window_seconds"`` (float).
            - ``"events_in_window"`` (int).

            Conditionally contains (only when latency data exists):
            - ``"p50_latency_ms"`` (float, rounded to 1 dp).
            - ``"p99_latency_ms"`` (float, rounded to 1 dp).
        """
        with self._lock:
            lat_stats = self._latency_percentiles()
            result = {
                "rolling_tokens_per_second": round(self._current_locked(time.monotonic()), 1),
                "total_tokens_produced": self._total_tokens,
                "window_seconds": self.window_seconds,
                "events_in_window": len(self._events),
            }
            if lat_stats[0] is not None:
                result["p50_latency_ms"] = lat_stats[0]
            if lat_stats[1] is not None:
                result["p99_latency_ms"] = lat_stats[1]
            return result

    def _latency_percentiles(self) -> tuple[Optional[float], Optional[float]]:
        """Return (p50_ms, p99_ms) or (None, None) if no latency data.

        Caller must hold ``self._lock``.
        """
        if not self._latencies:
            return None, None
        sorted_lat = sorted(self._latencies)
        n = len(sorted_lat)
        p50 = sorted_lat[n // 2] if n > 0 else None
        # Use ceil-based index for p99 to avoid int(100*0.99)=99 returning
        # the maximum (index 99 of 100) instead of the true 99th percentile.
        p99_idx = max(0, int(n * 0.99 + 0.5) - 1) if n > 1 else 0
        p99 = sorted_lat[p99_idx] if n > 1 else sorted_lat[0]
        return (
            round(p50, 1) if p50 is not None else None,
            round(p99, 1) if p99 is not None else None,
        )

    def _prune(self, now: float) -> None:
        """Evict events outside the window, updating the running sum.

        Each evicted token count is subtracted from the running sum so
        ``current()`` remains O(1).

        Caller must hold ``self._lock``.

        Amortized O(1); worst-case O(n) if the entire window expires at once.
        """
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            _, tokens = self._events.popleft()
            self._window_sum -= tokens
        # Guard against NTP drift and other clock anomalies pushing window_sum negative.
        if self._window_sum < 0:
            logger.warning(
                "Clock skew detected — _window_sum=%d after pruning with now=%.3f, window=%.1f. "
                "Clamping to 0 to prevent negative throughput.",
                self._window_sum, now, self.window_seconds,
            )
            self._window_sum = 0
