"""Terminal-based real-time monitoring dashboard (Phase 7).

A curses-based TUI that displays live throughput, GPU utilization, memory
usage, pipeline queue depth, quality scores, and extrapolation estimates
in a compact single-screen view.  Refreshes at 1 Hz.

Usage
-----
>>> from benchmark.observability.dashboard import LiveDashboard
>>> dash = LiveDashboard(prometheus_exporter)
>>> dash.start()  # runs in a background thread
>>> # ... benchmark runs ...
>>> dash.update()
>>> dash.stop()

Or as a standalone command:
    python -m benchmark.observability.dashboard --port 9090
"""

from __future__ import annotations

import curses
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import the Prometheus exporter for live data.
# If unavailable, the dashboard shows a static "no data" view.
try:
    from benchmark.observability.prometheus_metrics import PrometheusExporter
    HAS_EXPORTER = True
except ImportError:
    HAS_EXPORTER = False


class LiveDashboard:
    """Curses-based TUI for real-time benchmark monitoring.

    Displays a compact dashboard with:
      - Throughput (current rolling + average)
      - GPU utilization (per device, with temperature + power)
      - Memory (GPU + system)
      - Pipeline queue depth + data starvation
      - Batch statistics (completed, tokens, latency distribution)
      - Quality scores (if available)
      - Extrapolation estimate (ETA to completion)
    """

    _COLS_PER_DEVICE = 45  # character width per GPU column

    def __init__(
        self,
        exporter: PrometheusExporter | None = None,
        refresh_interval: float = 1.0,
        history_seconds: int = 120,
    ):
        self.exporter = exporter
        self.refresh_interval = refresh_interval
        self.history_seconds = history_seconds

        # Rolling history for sparkline-like visualization.
        self._throughput_history: list[float] = []
        self._util_history: list[list[float]] = []  # per-device
        self._max_history_points = history_seconds

        # Threading.
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._screen = None

        # Track last seen values for delta calculation.
        self._last_batches: float = 0.0
        self._last_tokens: float = 0.0
        self._last_update: float = time.monotonic()

    def start(self) -> None:
        """Start the dashboard in a background thread."""
        self._running.set()
        self._thread = threading.Thread(
            target=self._render_loop, name="dashboard", daemon=True,
        )
        self._thread.start()
        logger.info("Live dashboard started (refresh=%ss)", self.refresh_interval)

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Live dashboard stopped")

    def update(self) -> None:
        """Trigger a manual refresh (no-op for TUI; used in headless mode)."""
        if self.exporter is None:
            return
        snap = self.exporter.snapshot()
        self._update_history(snap)

    def render_text(self) -> str:
        """Render the dashboard as a plain-text string (for headless/CI logs).

        Returns
        -------
        str
            Multi-line dashboard text (ASCII, no curses).
        """
        if self.exporter is None:
            return "Dashboard: no Prometheus exporter connected.\n"

        snap = self.exporter.snapshot()
        self._update_history(snap)

        now = time.monotonic()
        dt = now - self._last_update
        self._last_update = now

        # Compute instant TPS from delta.
        tokens_total = snap.get("tokens_total", 0)
        delta_tokens = tokens_total - self._last_tokens
        instant_tps = delta_tokens / dt if dt > 0 else 0.0
        self._last_batches = snap.get("batches_total", 0)
        self._last_tokens = tokens_total

        lines = []
        lines.append("═" * 78)
        lines.append("  🇹🇷  Turkish Corpus Translation Benchmark — Live Monitor")
        lines.append("═" * 78)

        # Throughput row
        tps_bar = self._sparkline(self._throughput_history, width=40)
        throughput_tps = snap.get("throughput_tps", 0)
        lines.append(
            f"  Throughput: {throughput_tps:8.0f} tok/s "
            f"(instant: {instant_tps:7.0f})  {tps_bar}"
        )

        # Progress
        tokens_total = snap.get("tokens_total", 0)
        tokens_fmt = self._fmt_count(tokens_total)
        lines.append(
            f"  Batches: {snap.get('batches_total', 0):>8.0f}  |  "
            f"Tokens: {tokens_fmt:>12s}  |  "
            f"Queue depth: {snap.get('queue_depth', 0):>4.0f}"
        )

        lines.append("─" * 78)

        # System
        cpu_pct = snap.get("cpu_util_pct", 0)
        ram_bytes = snap.get("ram_used_bytes", 0)
        swap_bytes = snap.get("swap_used_bytes", 0)
        starvation_pct = snap.get("data_starvation_pct", 0)
        lines.append(
            f"  CPU: {cpu_pct:5.1f}%  |  "
            f"RAM: {ram_bytes/(1024**3):6.1f} GiB |  "
            f"Swap: {swap_bytes/(1024**3):5.1f} GiB |  "
            f"Data starvation: {starvation_pct:5.1f}%"
        )

        lines.append("─" * 78)

        # Quality
        bleu = snap.get("quality_bleu", 0)
        chrf = snap.get("quality_chrf", 0)
        comet = snap.get("quality_comet", 0)
        bleu_str = f"{bleu:.1f}" if bleu > 0 else "—"
        chrf_str = f"{chrf:.1f}" if chrf > 0 else "—"
        comet_str = f"{comet:.4f}" if comet > 0 else "—"
        lines.append(
            f"  Quality:  BLEU {bleu_str:>6s}  |  "
            f"chrF++ {chrf_str:>6s}  |  "
            f"COMET {comet_str:>10s}"
        )

        lines.append("═" * 78)
        return "\n".join(lines) + "\n"

    def _render_loop(self) -> None:
        """Main curses rendering loop (runs in background thread)."""
        try:
            self._screen = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.curs_set(0)
            self._screen.nodelay(True)

            while self._running.is_set():
                try:
                    text = self.render_text()
                    self._screen.clear()
                    y = 0
                    for line in text.split("\n"):
                        try:
                            self._screen.addstr(y, 0, line[:curses.COLS - 1])
                            y += 1
                        except curses.error:
                            pass
                    self._screen.refresh()
                except Exception:
                    pass
                time.sleep(self.refresh_interval)
        except Exception as e:
            logger.debug("Curses dashboard failed (terminal may not support it): %s", e)
        finally:
            if self._screen is not None:
                try:
                    curses.nocbreak()
                    curses.echo()
                    curses.endwin()
                except Exception:
                    pass

    def _update_history(self, snap: dict) -> None:
        """Update rolling history buffers."""
        self._throughput_history.append(snap.get("throughput_tps", 0))
        if len(self._throughput_history) > self._max_history_points:
            self._throughput_history = self._throughput_history[-self._max_history_points:]

    @staticmethod
    def _sparkline(values: list[float], width: int = 20, max_val: float | None = None) -> str:
        """Render a unicode sparkline bar from a list of values."""
        if not values:
            return "▁" * width

        if max_val is None:
            m = max(values)
            max_val = m if m > 0 else 1.0

        chars = " ▁▂▃▄▅▆▇█"
        n = len(chars) - 1

        # Downsample to fit width.
        if len(values) <= width:
            return "".join(
                chars[min(int(v / max_val * n), n)] for v in values
            )
        else:
            step = len(values) / width
            result = []
            for i in range(width):
                start = int(i * step)
                end = int((i + 1) * step)
                chunk = values[start:end]
                avg = sum(chunk) / len(chunk) if chunk else 0.0
                result.append(chars[min(int(avg / max_val * n), n)])
            return "".join(result)

    @staticmethod
    def _fmt_count(n: int) -> str:
        """Format a large integer with SI suffixes."""
        if n >= 1e12:
            return f"{n/1e12:.1f}T"
        elif n >= 1e9:
            return f"{n/1e9:.1f}B"
        elif n >= 1e6:
            return f"{n/1e6:.1f}M"
        elif n >= 1e3:
            return f"{n/1e3:.1f}K"
        return str(int(n))
