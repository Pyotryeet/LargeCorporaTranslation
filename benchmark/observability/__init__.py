"""Observability package — Prometheus metrics, Nsight profiler integration,
real-time dashboards, and performance regression tracking.

The full Grafana + Prometheus dashboard stack requires Docker
(``make dashboard``), not in-process Python.

Phase 7 modules
---------------
- prometheus_metrics : Prometheus endpoint for real-time metrics scraping.
- nsight_profiler    : NVIDIA Nsight Systems integration helpers.
- dashboard          : Terminal-based live monitoring dashboard.
- perf_regression    : Performance baseline + regression detection.
"""

from benchmark.observability.prometheus_metrics import PrometheusExporter

__all__ = [
    "PrometheusExporter",
]
