"""Observability package — Prometheus metrics.

The full Grafana + Prometheus dashboard stack requires Docker
(``make dashboard``), not in-process Python.

v3.8 modules
------------
- prometheus_metrics : Prometheus endpoint for real-time metrics scraping.
"""

from benchmark.observability.prometheus_metrics import PrometheusExporter

__all__ = [
    "PrometheusExporter",
]
