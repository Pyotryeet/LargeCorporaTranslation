"""Rolling throughput calculator with O(1) queries (P0-10).

Maintains a running token sum so ``current()`` is constant-time regardless
of window size.  Previously used ``sum(t for _, t in self._events)`` which
was O(n) per query and recomputed on every heartbeat.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass


@dataclass
class ThroughputSnapshot:
    timestamp: float
    tokens_per_second: float
    total_tokens: int
    window_seconds: float


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

    def add(self, tokens: int, latency_ms: float) -> None:
        """Record a batch completion.

        ``latency_ms`` is accepted for interface compatibility with
        alternative throughput implementations (e.g., latency-based),
        but this tracker computes throughput from wall-clock timestamps
        rather than per-batch latency, so the argument is unused.
        """
        now = time.monotonic()
        with self._lock:
            self._events.append((now, tokens))
            self._total_tokens += tokens
            self._window_sum += tokens
            self._prune(now)

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
            return ThroughputSnapshot(
                timestamp=now,
                tokens_per_second=round(self._current_locked(now), 1),
                total_tokens=self._total_tokens,
                window_seconds=self.window_seconds,
            )

    def summary(self) -> dict:
        with self._lock:
            return {
                "rolling_tokens_per_second": round(self._current_locked(time.monotonic()), 1),
                "total_tokens_produced": self._total_tokens,
                "window_seconds": self.window_seconds,
                "events_in_window": len(self._events),
            }

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
            self._window_sum = 0
