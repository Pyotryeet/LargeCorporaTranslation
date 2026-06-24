"""Observability package — Prometheus metrics and performance regression tracking.

The full Grafana + Prometheus dashboard stack requires Docker
(``make dashboard``), not in-process Python.

v3.6 modules
------------
- prometheus_metrics : Prometheus endpoint for real-time metrics scraping.
- perf_regression    : Performance baseline + regression detection (not yet wired).
"""

from benchmark.observability.prometheus_metrics import PrometheusExporter

__all__ = [
    "PrometheusExporter",
]
