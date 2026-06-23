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
    timestamp: float
    tokens_per_second: float
    total_tokens: int
    window_seconds: float
    p50_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None


class ThroughputTracker:
    def __init__(self, window_seconds: float = 60.0):
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

        # Single event: compute throughput as tokens / effective_span.
        if len(self._events) == 1:
            ts, tokens = self._events[0]
            age = now - ts
            effective_span = age if 0 < age < self.window_seconds else self.window_seconds
            if effective_span <= 0:
                return 0.0
            return tokens / effective_span

        # Multiple events: use the span covered by events (clamped to window).
        span = self._events[-1][0] - self._events[0][0]
        duration = span if span > 0 else self.window_seconds
        if duration <= 0:
            return 0.0
        # When the span exceeds the window, use the window size (pruning ensures
        # events outside the window are removed, but span may be < window when
        # not yet fully populated).
        effective_duration = min(duration, self.window_seconds)
        return self._window_sum / effective_duration

    def snapshot(self) -> ThroughputSnapshot:
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
        p50 = sorted_lat[int(n * 0.50)] if n > 0 else None
        p99 = sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[0]
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
