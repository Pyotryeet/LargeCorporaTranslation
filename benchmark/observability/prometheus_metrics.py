"""Prometheus metrics exporter for real-time observability (Phase 7).

Exposes all benchmark metrics as Prometheus gauges, counters, and histograms.
Integrates with the MetricsCollector to push samples on every batch and
device poll interval.  A lightweight HTTP server on localhost:9090 serves
the ``/metrics`` endpoint for Prometheus scraping.

Metrics exposed
---------------
Counters (monotonically increasing):
  - tr_benchmark_batches_total
  - tr_benchmark_tokens_translated_total
  - tr_benchmark_sequences_completed_total
  - tr_benchmark_errors_total

Gauges (instantaneous value):
  - tr_benchmark_throughput_tokens_per_second
  - tr_benchmark_gpu_utilization_percent (per device)
  - tr_benchmark_gpu_memory_used_bytes (per device)
  - tr_benchmark_gpu_temperature_celsius (per device)
  - tr_benchmark_gpu_power_watts (per device)
  - tr_benchmark_queue_depth
  - tr_benchmark_data_starvation_percent
  - tr_benchmark_cpu_utilization_percent
  - tr_benchmark_ram_used_bytes
  - tr_benchmark_quality_bleu
  - tr_benchmark_quality_chrf
  - tr_benchmark_quality_comet
  - tr_benchmark_quality_bertscore
  - tr_benchmark_quality_comet_kiwi

Histograms:
  - tr_benchmark_batch_latency_seconds
  - tr_benchmark_decode_time_seconds
  - tr_benchmark_prefill_time_seconds

Usage
-----
>>> from benchmark.observability.prometheus_metrics import PrometheusExporter
>>> exporter = PrometheusExporter(port=9090)
>>> exporter.start()
>>> # ... benchmark runs ...
>>> exporter.record_batch(batch_result)
>>> exporter.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight Prometheus client (no external dependency)
# ---------------------------------------------------------------------------
# We implement the Prometheus text format directly to avoid adding
# ``prometheus_client`` as a dependency.  The format is simple enough
# that ~150 lines is sufficient for our use case.


class _Metric:
    """Base metric with atomic updates via lock."""

    def __init__(self, name: str, help_text: str, labels: dict[str, str] | None = None):
        self.name = name
        self.help = help_text
        self.labels = labels or {}
        self._lock = threading.Lock()
        self._created_ts = time.time()

    def _label_str(self) -> str:
        if not self.labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(parts) + "}"

    def render(self) -> str:
        raise NotImplementedError


class Counter(_Metric):
    """Monotonically increasing counter."""

    def __init__(self, name: str, help_text: str, labels: dict[str, str] | None = None):
        super().__init__(name, help_text, labels)
        self._value: float = 0.0

    def inc(self, delta: float = 1.0) -> None:
        with self._lock:
            self._value += delta

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def get(self) -> float:
        with self._lock:
            return self._value

    def render(self) -> str:
        label_str = self._label_str()
        return (
            f"# HELP {self.name} {self.help}\n"
            f"# TYPE {self.name} counter\n"
            f"{self.name}{label_str} {self.get()}\n"
        )


class Gauge(_Metric):
    """Instantaneous value that can go up and down."""

    def __init__(self, name: str, help_text: str, labels: dict[str, str] | None = None):
        super().__init__(name, help_text, labels)
        self._value: float = 0.0

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def get(self) -> float:
        with self._lock:
            return self._value

    def render(self) -> str:
        label_str = self._label_str()
        return (
            f"# HELP {self.name} {self.help}\n"
            f"# TYPE {self.name} gauge\n"
            f"{self.name}{label_str} {self.get()}\n"
        )


class Histogram(_Metric):
    """Pre-configured bucket histogram for latency measurements.

    Buckets (seconds) are tuned for GPU inference latencies ranging from
    sub-millisecond prefill/decode on small models to multi-second batch
    latencies on large models: 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
    0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, +Inf
    """

    _DEFAULT_BUCKETS = [
        0.001, 0.0025, 0.005,
        0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
        1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
    ]

    def __init__(
        self,
        name: str,
        help_text: str,
        labels: dict[str, str] | None = None,
        buckets: list[float] | None = None,
    ):
        super().__init__(name, help_text, labels)
        self.buckets = buckets or self._DEFAULT_BUCKETS
        self._bucket_counts: dict[float, int] = {b: 0 for b in self.buckets}
        self._sum: float = 0.0
        self._count: int = 0
        self._inf_count: int = 0

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            for boundary in self.buckets:
                if value <= boundary:
                    self._bucket_counts[boundary] += 1
                    return
            self._inf_count += 1

    def render(self) -> str:
        label_str = self._label_str()
        lines = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            for boundary in self.buckets:
                lines.append(
                    f"{self.name}_bucket{label_str}"
                    f'{{le="{boundary}"}} {self._bucket_counts[boundary]}'
                )
            lines.append(
                f"{self.name}_bucket{label_str}"
                f'{{le="+Inf"}} {self._count + self._inf_count}'
            )
            lines.append(f"{self.name}_sum{label_str} {self._sum}")
            lines.append(f"{self.name}_count{label_str} {self._count}")
        lines.append("")
        return "\n".join(lines)


class PrometheusRegistry:
    """Collects all metrics and renders the /metrics endpoint."""

    def __init__(self):
        self._metrics: dict[str, _Metric] = {}

    def register(self, metric: _Metric) -> _Metric:
        key = (metric.name, tuple(sorted(metric.labels.items())))
        self._metrics[str(key)] = metric
        return metric

    def counter(self, name: str, help_text: str, labels: dict[str, str] | None = None) -> Counter:
        m = Counter(name, help_text, labels)
        self.register(m)
        return m

    def gauge(self, name: str, help_text: str, labels: dict[str, str] | None = None) -> Gauge:
        m = Gauge(name, help_text, labels)
        self.register(m)
        return m

    def histogram(self, name: str, help_text: str, labels: dict[str, str] | None = None) -> Histogram:
        m = Histogram(name, help_text, labels)
        self.register(m)
        return m

    def render_all(self) -> str:
        parts = []
        for m in self._metrics.values():
            parts.append(m.render())
        parts.append("# EOF\n")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTTP handler for /metrics — implemented as _PerExportHandler closure
# inside PrometheusExporter.start() (sole implementation).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main exporter class
# ---------------------------------------------------------------------------


class PrometheusExporter:
    """Prometheus metrics exporter for the translation benchmark.

    Usage
    -----
    >>> exporter = PrometheusExporter(port=9090)  # each node needs a unique port
    >>> exporter.start()
    >>> # During benchmark:
    >>> exporter.record_batch(tokens=128, latency_ms=450, prefill_ms=80, decode_ms=370)
    >>> exporter.record_device(gpu_id=0, util_pct=85.0, mem_bytes=42e9, temp_c=65.0, power_w=300.0)
    >>> exporter.record_quality(bleu=32.5, chrf=58.2, comet=0.785)
    >>> exporter.stop()

    .. note::

        Each node in a multi-node deployment **must** use a unique port.
        The default ``9090`` conflicts when multiple exporter instances run
        on the same host (e.g., one per GPU process in a multi-GPU or
        multi-node setup).  Assign distinct ports per process, for example::

            exporter = PrometheusExporter(port=9090 + local_rank)
    """

    def __init__(
        self,
        port: int | None = None,
        host: str = "localhost",
        backend: str = "unknown",
        num_gpus: int = 1,
    ):
        # WARNING: port 9090 is hardcoded. Multi-node deployments must pass
        # unique ports via TR_PROMETHEUS_PORT or equivalent.
        import os
        self.port = (
            port if port is not None
            else int(os.environ.get("TR_PROMETHEUS_PORT", 9090))
        )
        self.host = host
        self.registry = PrometheusRegistry()

        # ── Counter metrics ──
        self.batches_total = self.registry.counter(
            "tr_benchmark_batches_total",
            "Total number of batches processed.",
        )
        self.tokens_total = self.registry.counter(
            "tr_benchmark_tokens_translated_total",
            "Total number of output tokens translated.",
        )
        self.sequences_completed = self.registry.counter(
            "tr_benchmark_sequences_completed_total",
            "Total number of sequences completed.",
        )
        self.errors_total = self.registry.counter(
            "tr_benchmark_errors_total",
            "Total number of errors encountered.",
        )

        # ── Gauge metrics ──
        self.throughput = self.registry.gauge(
            "tr_benchmark_throughput_tokens_per_second",
            "Rolling throughput in tokens per second.",
        )
        self.queue_depth = self.registry.gauge(
            "tr_benchmark_queue_depth",
            "Current pipeline queue depth (tokenised chunks waiting).",
        )
        self.data_starvation = self.registry.gauge(
            "tr_benchmark_data_starvation_percent",
            "Percentage of time GPU was idle waiting for data.",
        )
        self.cpu_util = self.registry.gauge(
            "tr_benchmark_cpu_utilization_percent",
            "CPU utilization percentage.",
        )
        self.ram_used = self.registry.gauge(
            "tr_benchmark_ram_used_bytes",
            "RAM used in bytes.",
        )
        self.swap_used = self.registry.gauge(
            "tr_benchmark_swap_used_bytes",
            "Swap used in bytes.",
        )

        # Per-device gauges (one instance per GPU).
        self._device_gauges: list[dict[str, Gauge]] = []
        for i in range(max(num_gpus, 1)):
            dev_labels = {"device": str(i), "backend": backend}
            self._device_gauges.append({
                "util": self.registry.gauge(
                    "tr_benchmark_gpu_utilization_percent",
                    f"GPU {i} utilization percentage.",
                    dev_labels,
                ),
                "mem_used": self.registry.gauge(
                    "tr_benchmark_gpu_memory_used_bytes",
                    f"GPU {i} memory used in bytes.",
                    dev_labels,
                ),
                "mem_total": self.registry.gauge(
                    "tr_benchmark_gpu_memory_total_bytes",
                    f"GPU {i} total memory in bytes.",
                    dev_labels,
                ),
                "temp": self.registry.gauge(
                    "tr_benchmark_gpu_temperature_celsius",
                    f"GPU {i} temperature in Celsius.",
                    dev_labels,
                ),
                "power": self.registry.gauge(
                    "tr_benchmark_gpu_power_watts",
                    f"GPU {i} power draw in watts.",
                    dev_labels,
                ),
            })

        # Quality gauges
        self.quality_bleu = self.registry.gauge(
            "tr_benchmark_quality_bleu",
            "BLEU score from the most recent quality benchmark.",
        )
        self.quality_chrf = self.registry.gauge(
            "tr_benchmark_quality_chrf",
            "chrF++ score from the most recent quality benchmark.",
        )
        self.quality_comet = self.registry.gauge(
            "tr_benchmark_quality_comet",
            "COMET-22 system score from the most recent quality benchmark.",
        )
        self.quality_bertscore = self.registry.gauge(
            "tr_benchmark_quality_bertscore",
            "BERTScore F1 system score from the most recent quality benchmark.",
        )
        self.quality_comet_kiwi = self.registry.gauge(
            "tr_benchmark_quality_comet_kiwi",
            "COMET-Kiwi (reference-free) system score from the most recent quality benchmark.",
        )

        # ── Histogram metrics ──
        self.latency_hist = self.registry.histogram(
            "tr_benchmark_batch_latency_seconds",
            "Total batch latency in seconds.",
        )
        self.decode_hist = self.registry.histogram(
            "tr_benchmark_decode_time_seconds",
            "Decode-phase latency in seconds.",
        )
        self.prefill_hist = self.registry.histogram(
            "tr_benchmark_prefill_time_seconds",
            "Prefill-phase latency in seconds.",
        )

        # HTTP server
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    def _index_html(self) -> str:
        return """<!DOCTYPE html>
<html><head><title>TR Benchmark Metrics</title></head>
<body>
<h1>TR Benchmark v3.6 — Metrics</h1>
<ul>
<li><a href="/metrics">/metrics</a> — Prometheus text format</li>
<li><a href="/health">/health</a> — Health check</li>
</ul>
</body></html>"""

    def start(self) -> None:
        """Start the Prometheus HTTP metrics server on a background thread."""
        _exporter = self
        class _PerExportHandler(BaseHTTPRequestHandler):
            """Sole implementation of the /metrics HTTP handler.

            Defined as a closure inside start() so ``registry`` is captured
            from the enclosing PrometheusExporter instance.
            """
            registry = _exporter.registry

            def do_GET(handler_self) -> None:
                if handler_self.path == "/metrics":
                    if handler_self.registry is None:
                        handler_self.send_error(503, "Registry not initialized")
                        return
                    data = handler_self.registry.render_all()
                    handler_self.send_response(200)
                    handler_self.send_header("Content-Type", "text/plain; charset=utf-8")
                    handler_self.send_header("Content-Length", str(len(data)))
                    handler_self.end_headers()
                    handler_self.wfile.write(data.encode("utf-8"))
                elif handler_self.path == "/health":
                    handler_self.send_response(200)
                    handler_self.send_header("Content-Type", "text/plain")
                    handler_self.end_headers()
                    handler_self.wfile.write(b"OK\n")
                elif handler_self.path == "/":
                    handler_self.send_response(200)
                    handler_self.send_header("Content-Type", "text/html; charset=utf-8")
                    handler_self.end_headers()
                    handler_self.wfile.write(_exporter._index_html().encode("utf-8"))
                else:
                    handler_self.send_error(404)

            def log_message(handler_self, format, *args) -> None:
                """Suppress HTTP access logs (too noisy for metrics endpoint)."""
                pass

        self._server = HTTPServer((self.host, self.port), _PerExportHandler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="prometheus-server",
            daemon=True,
        )
        self._server_thread.start()
        logger.info(
            "Prometheus metrics server started on http://%s:%d/metrics",
            self.host, self.port,
        )

    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
            self._server_thread = None
        logger.info("Prometheus metrics server stopped")

    # ── Record methods ──────────────────────────────────────────────────

    def record_batch(
        self,
        tokens: int,
        latency_ms: float,
        prefill_ms: float = 0.0,
        decode_ms: float = 0.0,
        batch_size: int = 1,
    ) -> None:
        self.batches_total.inc()
        self.tokens_total.inc(tokens)
        self.sequences_completed.inc(batch_size)
        self.throughput.set(
            (tokens / latency_ms) * 1000.0 if latency_ms > 0 else 0.0
        )
        self.latency_hist.observe(latency_ms / 1000.0)
        if prefill_ms > 0:
            self.prefill_hist.observe(prefill_ms / 1000.0)
        if decode_ms > 0:
            self.decode_hist.observe(decode_ms / 1000.0)

    def record_device(
        self,
        device_id: int,
        util_pct: float | None = None,
        mem_used_mib: float | None = None,
        mem_total_mib: float | None = None,
        temp_c: float | None = None,
        power_w: float | None = None,
    ) -> None:
        """Record device-level metrics for one GPU."""
        if device_id >= len(self._device_gauges):
            return
        g = self._device_gauges[device_id]
        if util_pct is not None:
            g["util"].set(util_pct)
        if mem_used_mib is not None:
            g["mem_used"].set(mem_used_mib * 1024 * 1024)
        if mem_total_mib is not None:
            g["mem_total"].set(mem_total_mib * 1024 * 1024)
        if temp_c is not None:
            g["temp"].set(temp_c)
        if power_w is not None:
            g["power"].set(power_w)

    def record_system(
        self,
        cpu_pct: float,
        ram_used_mib: float,
        swap_used_mib: float = 0.0,
    ) -> None:
        """Record system-level metrics."""
        self.cpu_util.set(cpu_pct)
        self.ram_used.set(ram_used_mib * 1024 * 1024)
        self.swap_used.set(swap_used_mib * 1024 * 1024)

    def record_pipeline(self, queue_depth: int, starvation_pct: float = 0.0) -> None:
        """Record pipeline-level metrics."""
        self.queue_depth.set(queue_depth)
        self.data_starvation.set(starvation_pct)

    def record_quality(
        self,
        bleu: float | None = None,
        chrf: float | None = None,
        comet: float | None = None,
        bertscore: float | None = None,
        comet_kiwi: float | None = None,
    ) -> None:
        """Record quality benchmark scores."""
        if bleu is not None:
            self.quality_bleu.set(bleu)
        if chrf is not None:
            self.quality_chrf.set(chrf)
        if comet is not None:
            self.quality_comet.set(comet)
        if bertscore is not None:
            self.quality_bertscore.set(bertscore)
        if comet_kiwi is not None:
            self.quality_comet_kiwi.set(comet_kiwi)

    def record_error(self) -> None:
        """Increment the error counter."""
        self.errors_total.inc()

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a JSON-serializable summary of all current metric values."""
        return {
            "batches_total": self.batches_total.get(),
            "tokens_total": self.tokens_total.get(),
            "sequences_completed": self.sequences_completed.get(),
            "errors_total": self.errors_total.get(),
            "throughput_tps": self.throughput.get(),
            "queue_depth": self.queue_depth.get(),
            "data_starvation_pct": self.data_starvation.get(),
            "cpu_util_pct": self.cpu_util.get(),
            "ram_used_bytes": self.ram_used.get(),
            "swap_used_bytes": self.swap_used.get(),
            "quality_bleu": self.quality_bleu.get(),
            "quality_chrf": self.quality_chrf.get(),
            "quality_comet": self.quality_comet.get(),
            "quality_bertscore": self.quality_bertscore.get(),
            "quality_comet_kiwi": self.quality_comet_kiwi.get(),
        }
