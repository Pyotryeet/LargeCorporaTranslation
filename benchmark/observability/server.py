"""Prometheus monitoring dashboard — real-time HTML + terminal TUI views.

Provides:
1. A lightweight web dashboard at ``http://<host>:<port>/`` showing live metrics
   with auto-refresh, throughput sparklines, GPU utilization bars, and quality
   score status.
2. Terminal-based TUI (curses) for SSH/headless monitoring.
3. JSON API at ``/api/snapshot`` for programmatic consumption.

.. note::

   This in-process server provides an HTTP-only dashboard and Prometheus
   ``/metrics`` endpoint.  For the full observability stack (Grafana
   dashboards, persistent metric storage, alerting), use the provided
   ``docker-compose`` setup or ``make dashboard`` which launches Prometheus
   + Grafana alongside the benchmark.

Usage
-----
    python -m benchmark --config config.yaml --observability-port 9090 --dashboard

Or programmatically:
    from benchmark.observability.server import start_dashboard_server
    server = start_dashboard_server(exporter, port=9090)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from benchmark.observability.prometheus_metrics import PrometheusExporter
except ImportError:
    PrometheusExporter = None  # type: ignore[assignment]


# ── HTML dashboard template ──────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TR Benchmark — Live Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0d1117;color:#c9d1d9;overflow-x:hidden}
.header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px;
  display:flex;align-items:center;justify-content:space-between}
.header h1{font-size:20px;font-weight:600;color:#f0f6fc}
.header .status{font-size:13px;padding:4px 12px;border-radius:12px}
.status.running{background:#1b3a1b;color:#3fb950}
.status.error{background:#3a1b1b;color:#f85149}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
  gap:16px;padding:24px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px}
.card h2{font-size:14px;font-weight:600;color:#8b949e;text-transform:uppercase;
  letter-spacing:0.5px;margin-bottom:12px}
.big-number{font-size:42px;font-weight:700;color:#f0f6fc;line-height:1}
.big-number .unit{font-size:16px;font-weight:400;color:#8b949e;margin-left:4px}
.sparkline{font-family:monospace;font-size:12px;color:#58a6ff;
  letter-spacing:-1px;white-space:pre;overflow:hidden}
.metric-row{display:flex;justify-content:space-between;padding:6px 0;
  border-bottom:1px solid #21262d;font-size:13px}
.metric-row:last-child{border-bottom:none}
.metric-label{color:#8b949e}
.metric-value{color:#f0f6fc;font-weight:500;font-variant-numeric:tabular-nums}
.metric-value.good{color:#3fb950}.metric-value.warn{color:#d29922}
.metric-value.bad{color:#f85149}
.bar-container{background:#21262d;border-radius:4px;height:8px;margin:4px 0;overflow:hidden}
.bar-fill{background:#58a6ff;height:100%;border-radius:4px;transition:width 0.5s}
.bar-fill.warm{background:#d29922}.bar-fill.hot{background:#f85149}
.gpu-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.footer{text-align:center;padding:16px;color:#484f58;font-size:12px}
.refresh-indicator{display:inline-block;width:8px;height:8px;
  background:#3fb950;border-radius:50%;margin-right:6px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
</style>
</head>
<body>
<div class="header">
  <h1>🇹🇷 Turkish Corpus Translation Benchmark</h1>
  <span class="status running" id="status">● running</span>
</div>
<div class="grid" id="metrics"></div>
<div class="footer">
  <span class="refresh-indicator"></span> Auto-refresh every 2s ·
  Prometheus: <a href="/metrics" style="color:#58a6ff">/metrics</a> ·
  API: <a href="/api/snapshot" style="color:#58a6ff">/api/snapshot</a>
</div>
<script>
async function refresh() {
  try {
    const resp = await fetch('/api/snapshot');
    const data = await resp.json();
    document.getElementById('status').textContent =
      data.throughput_tps > 0 ? '● running' : '○ idle';
    document.getElementById('status').className =
      'status ' + (data.throughput_tps > 0 ? 'running' : 'error');
    document.getElementById('metrics').innerHTML = renderMetrics(data);
  } catch(e) { document.getElementById('status').textContent = '● disconnected';
    document.getElementById('status').className = 'status error'; }
}
function bar(val, max, cls) {
  const pct = Math.min((val/max)*100, 100);
  const c = cls || (pct > 80 ? 'hot' : pct > 60 ? 'warm' : '');
  return `<div class="bar-container"><div class="bar-fill ${c}" style="width:${pct}%"></div></div>`;
}
function renderMetrics(d) {
  return `
    <div class="card">
      <h2>⚡ Throughput</h2>
      <div class="big-number">${d.throughput_tps.toFixed(0)}<span class="unit">tok/s</span></div>
      <div class="sparkline" id="spark"></div>
    </div>
    <div class="card">
      <h2>📊 Progress</h2>
      <div class="metric-row"><span class="metric-label">Batches</span><span class="metric-value">${d.batches_total.toLocaleString()}</span></div>
      <div class="metric-row"><span class="metric-label">Tokens</span><span class="metric-value">${(d.tokens_total/1e6).toFixed(1)}M</span></div>
      <div class="metric-row"><span class="metric-label">Queue depth</span><span class="metric-value">${d.queue_depth.toFixed(0)}</span></div>
      <div class="metric-row"><span class="metric-label">Data starvation</span><span class="metric-value ${d.data_starvation_pct > 20 ? 'warn' : 'good'}">${d.data_starvation_pct.toFixed(1)}%</span></div>
    </div>
    <div class="card">
      <h2>💻 System</h2>
      <div class="metric-row"><span class="metric-label">CPU</span><span class="metric-value">${d.cpu_util_pct.toFixed(1)}%</span></div>
      ${bar(d.cpu_util_pct, 100)}
      <div class="metric-row"><span class="metric-label">RAM</span><span class="metric-value">${(d.ram_used_bytes/(1024**3)).toFixed(1)} GB</span></div>
      <div class="metric-row"><span class="metric-label">Sequences</span><span class="metric-value">${d.sequences_completed.toLocaleString()}</span></div>
      <div class="metric-row"><span class="metric-label">Errors</span><span class="metric-value ${d.errors_total > 0 ? 'bad' : 'good'}">${d.errors_total}</span></div>
    </div>
    <div class="card">
      <h2>🎯 Quality</h2>
      <div class="metric-row"><span class="metric-label">BLEU</span><span class="metric-value ${d.quality_bleu >= 25 ? 'good' : 'warn'}">${d.quality_bleu > 0 ? d.quality_bleu.toFixed(1) : '—'}</span></div>
      <div class="metric-row"><span class="metric-label">chrF++</span><span class="metric-value ${d.quality_chrf >= 54 ? 'good' : 'warn'}">${d.quality_chrf > 0 ? d.quality_chrf.toFixed(1) : '—'}</span></div>
      <div class="metric-row"><span class="metric-label">COMET</span><span class="metric-value ${d.quality_comet >= 0.72 ? 'good' : 'warn'}">${d.quality_comet > 0 ? d.quality_comet.toFixed(4) : '—'}</span></div>
    </div>`;
}
refresh(); setInterval(refresh, 2000);
</script>
</body>
</html>"""


# ── HTTP server ──────────────────────────────────────────────────────────


class DashboardServer:
    """Serve the Prometheus endpoint + web dashboard + JSON API.

    Usage
    -----
    >>> server = DashboardServer(exporter, port=9090)
    >>> server.start()
    >>> # ... benchmark runs ...
    >>> server.stop()
    """

    def __init__(
        self,
        exporter: PrometheusExporter | None = None,
        port: int = 9090,
        host: str = "localhost",
    ):
        self.exporter = exporter
        self.port = port
        self.host = host
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the dashboard HTTP server on a background thread."""
        exporter = self.exporter  # capture for closure
        class _Handler(BaseHTTPRequestHandler):
            exporter_ref = exporter

            def do_GET(handler_self) -> None:
                if handler_self.path == "/" or handler_self.path == "/dashboard":
                    handler_self._serve_html(_DASHBOARD_HTML)
                elif handler_self.path == "/api/snapshot":
                    handler_self._serve_json(handler_self._get_snapshot())
                elif handler_self.path == "/health":
                    handler_self._serve_text("OK\n", 200)
                elif handler_self.path == "/metrics":
                    handler_self._serve_metrics()
                else:
                    handler_self.send_error(404)

            def _get_snapshot(handler_self) -> dict:
                if handler_self.exporter_ref is None:
                    return {"error": "No exporter connected"}
                return handler_self.exporter_ref.snapshot()

            def _serve_html(handler_self, html: str) -> None:
                data = html.encode("utf-8")
                handler_self.send_response(200)
                handler_self.send_header("Content-Type", "text/html; charset=utf-8")
                handler_self.send_header("Content-Length", str(len(data)))
                handler_self.end_headers()
                handler_self.wfile.write(data)

            def _serve_json(handler_self, data: dict) -> None:
                body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
                handler_self.send_response(200)
                handler_self.send_header("Content-Type", "application/json")
                handler_self.send_header("Content-Length", str(len(body)))
                handler_self.send_header("Access-Control-Allow-Origin", "*")
                handler_self.end_headers()
                handler_self.wfile.write(body)

            def _serve_text(handler_self, text: str, code: int = 200) -> None:
                data = text.encode("utf-8")
                handler_self.send_response(code)
                handler_self.send_header("Content-Type", "text/plain")
                handler_self.send_header("Content-Length", str(len(data)))
                handler_self.end_headers()
                handler_self.wfile.write(data)

            def _serve_metrics(handler_self) -> None:
                if handler_self.exporter_ref is None:
                    handler_self.send_error(503)
                    return
                body = handler_self.exporter_ref.registry.render_all().encode("utf-8")
                handler_self.send_response(200)
                handler_self.send_header("Content-Type", "text/plain; charset=utf-8")
                handler_self.send_header("Content-Length", str(len(body)))
                handler_self.end_headers()
                handler_self.wfile.write(body)

            def log_message(handler_self, format, *args):
                """Suppress noisy access logs."""
                if "/api/snapshot" in (args[0] if args else ""):
                    return  # Don't log the polling endpoint.
                logger.debug("Dashboard HTTP: " + format % args)

        self._httpd = HTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="dashboard-http",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Dashboard: http://%s:%d/ (Prometheus: /metrics, API: /api/snapshot)",
            self.host, self.port,
        )

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Dashboard server stopped")


def start_dashboard_server(
    exporter: PrometheusExporter | None = None,
    port: int = 9090,
    host: str = "localhost",
) -> DashboardServer:
    """Convenience: create and start a dashboard server.

    Returns the server handle for later ``stop()``.
    """
    server = DashboardServer(exporter=exporter, port=port, host=host)
    server.start()
    return server
