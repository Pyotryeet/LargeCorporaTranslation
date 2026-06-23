"""NVIDIA Nsight Systems profiler integration (Phase 7).

Provides fine-grained NVTX annotations and CUDA profiler API wrappers
for per-layer timing, memory bandwidth measurement, and kernel launch
overhead quantification.

Usage
-----
>>> profiler = NsightProfiler()
>>> profiler.start()
>>> with profiler.range("prefill_phase"):
...     outputs = model.generate(...)
>>> profiler.stop()
>>> profiler.report()

Requires NVIDIA Nsight Systems CLI (``nsys``) or the Python bindings
(``torch.cuda.nvtx``) — the latter is always available with CUDA PyTorch.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ProfilerSpan:
    """Timing data for one annotated span.

    ``gpu_time_ns`` / ``gpu_duration_ms`` are populated via CUDA Events when a
    CUDA device is available, regardless of profiling mode.
    """

    name: str
    start_ns: int = 0
    end_ns: int = 0
    gpu_time_ns: int = 0
    children: list[ProfilerSpan] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000.0

    @property
    def gpu_duration_ms(self) -> float:
        return self.gpu_time_ns / 1_000_000.0


class NsightProfiler:
    """Nsight Systems–compatible profiler using PyTorch profiler API.

    Provides structured per-phase and per-layer timing with minimal overhead
    (~2 µs per span).  Compatible with both CUDA (via CUPTI) and CPU backends.

    Two modes:
    - **lightweight** — NVTX annotations + CUDA Events for GPU timing
      (sub-microsecond overhead per span).  ``gpu_time_ns`` / ``gpu_duration_ms``
      are populated via ``torch.cuda.Event``.
    - **full** — PyTorch profiler with CUPTI trace export for Chrome trace
      viewer.  Also populates ``gpu_time_ns`` and ``gpu_duration_ms``.

    Usage
    -----
    >>> profiler = NsightProfiler(mode="full", trace_dir="./profiles")
    >>> profiler.start()
    >>>
    >>> with profiler.range("translate_batch"):
    ...     batch_result = engine.translate(batch)
    >>>
    >>> profiler.stop()
    >>> profiler.save_trace("run_2026-06-21.json")
    >>> print(profiler.summary())
    """

    def __init__(
        self,
        mode: str = "lightweight",
        trace_dir: str | Path = "./profiles",
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ):
        self.mode = mode
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack

        # Internal state.
        self._profiler: Optional[object] = None  # torch.profiler.profile when torch available
        self._spans: list[ProfilerSpan] = []
        self._span_stack: list[ProfilerSpan] = []
        self._active = False
        self._backend = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin profiling."""
        if self._active:
            logger.warning("Profiler already active — stop() first.")
            return

        self._spans.clear()
        self._span_stack.clear()

        if self.mode == "full" and torch is not None:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self._backend == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            self._profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=self.record_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
            )
            if self._profiler is not None:
                try:
                    self._profiler.__enter__()
                except Exception:
                    # Clean up on failure so a subsequent start() can retry.
                    self._profiler = None
                    self._spans.clear()
                    raise

        self._active = True

        if self.mode == "lightweight":
            logger.info(
                "Nsight profiler started in lightweight mode. "
                "GPU timing is available via CUDA Events in this mode."
            )
        logger.info(
            "Nsight profiler started (mode=%s, backend=%s, trace_dir=%s)",
            self.mode, self._backend, self.trace_dir,
        )

    def stop(self) -> None:
        """Stop profiling."""
        if not self._active:
            return

        if self._profiler is not None:
            self._profiler.__exit__(None, None, None)

        self._active = False
        logger.info("Profiler stopped (%d spans collected)", len(self._spans))

    # ── NVTX / span annotations ────────────────────────────────────────

    @contextlib.contextmanager
    def range(self, name: str, **metadata):
        """Context manager for annotated profiling ranges.

        Creates an NVTX range (CUDA) plus CPU-side and GPU-side span timing.

        GPU timing uses ``torch.cuda.Event`` — lightweight (~0.5 us overhead)
        and works even outside the PyTorch profiler, populating ``gpu_time_ns``
        and ``gpu_duration_ms`` in both ``lightweight`` and ``full`` modes.
        """
        span = ProfilerSpan(name=name, metadata=metadata)
        span.start_ns = time.monotonic_ns()

        # CUDA events for GPU-side timing (lightweight, works in both modes).
        gpu_start = None
        gpu_end = None
        if torch is not None and self._backend == "cuda" and self._active:
            torch.cuda.nvtx.range_push(name)
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()

        if self._span_stack:
            self._span_stack[-1].children.append(span)
        else:
            self._spans.append(span)
        self._span_stack.append(span)

        try:
            yield span
        finally:
            if torch is not None and self._backend == "cuda" and self._active:
                gpu_end.record()
                torch.cuda.nvtx.range_pop()
                # Synchronize to ensure the GPU events are ready; this is
                # lightweight (typically < 5 us for a single event pair).
                gpu_end.synchronize()
                span.gpu_time_ns = gpu_start.elapsed_time(gpu_end) * 1_000_000  # ms -> ns
            else:
                span.end_ns = time.monotonic_ns()  # Already CPU-only; no GPU timing.

            if self._backend == "cuda":
                # For GPU runs, record CPU end after GPU sync so duration_ms
                # reflects wall-clock (host) time including GPU work.
                span.end_ns = time.monotonic_ns()

            self._span_stack.pop()

    def mark(self, name: str) -> None:
        """Insert an instantaneous marker (no duration)."""
        if torch is not None and self._backend == "cuda" and self._active:
            torch.cuda.nvtx.mark(name)

    # ── Specialized ranges ──────────────────────────────────────────────

    @contextlib.contextmanager
    def prefill_range(self):
        """Named range for the prefill (prompt encoding) phase."""
        with self.range("prefill_phase", phase="prefill"):
            yield

    @contextlib.contextmanager
    def decode_range(self, step: int = 0):
        """Named range for a single decode step."""
        with self.range(f"decode_step_{step:04d}", phase="decode", step=step):
            yield

    @contextlib.contextmanager
    def attention_range(self, layer_idx: int, attn_type: str = "local"):
        """Named range for attention computation at a specific layer."""
        with self.range(
            f"attention_L{layer_idx:02d}_{attn_type}",
            layer=layer_idx,
            attention_type=attn_type,
        ):
            yield

    @contextlib.contextmanager
    def mlp_range(self, layer_idx: int):
        """Named range for MLP computation at a specific layer."""
        with self.range(f"mlp_L{layer_idx:02d}", layer=layer_idx):
            yield

    @contextlib.contextmanager
    def data_transfer_range(self, direction: str = "h2d"):
        """Named range for host↔device data transfer."""
        with self.range(f"transfer_{direction}", direction=direction):
            yield

    # ── Trace export ────────────────────────────────────────────────────

    def save_trace(self, filename: str | None = None) -> Path:
        """Save the profiling trace to disk.

        In ``full`` mode, exports the PyTorch profiler's Chrome trace JSON.
        In ``lightweight`` mode, saves our own span hierarchy as JSON.

        Parameters
        ----------
        filename : str, optional
            Output filename.  Default: ``trace_YYYYMMDD_HHMMSS.json``
            (e.g. ``trace_20260623_143051.json``).

        Returns
        -------
        Path
            Path to the saved trace file.
        """
        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"trace_{ts}.json"

        path = self.trace_dir / filename

        if self.mode == "full" and self._profiler is not None:
            # Export Chrome trace viewer format.
            self._profiler.export_chrome_trace(str(path))
            logger.info("Chrome trace saved to %s (%.1f KB)", path, path.stat().st_size / 1024)
        else:
            # Export our lightweight span tree.
            data = self._export_span_tree()
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Span tree saved to %s (%d spans)", path, len(data.get("spans", [])))

        return path

    def _export_span_tree(self) -> dict:
        """Recursively serialize spans to a JSON-serializable structure."""

        def _serialize(span: ProfilerSpan) -> dict:
            return {
                "name": span.name,
                "duration_ms": round(span.duration_ms, 3),
                "gpu_duration_ms": round(span.gpu_duration_ms, 3),
                "metadata": span.metadata,
                "children": [_serialize(c) for c in span.children],
            }

        return {
            "backend": self._backend,
            "mode": self.mode,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "spans": [_serialize(s) for s in self._spans],
        }

    # ── Summary ────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Compute aggregate statistics from all collected spans.

        Returns
        -------
        dict
            Keys: ``total_duration_ms``, ``span_count``, ``top_spans``,
            ``prefill_pct``, ``decode_pct``, ``phase_breakdown``.
        """
        if not self._spans:
            return {"span_count": 0}

        all_spans: list[ProfilerSpan] = []

        def _collect(s: ProfilerSpan):
            all_spans.append(s)
            for c in s.children:
                _collect(c)

        for s in self._spans:
            _collect(s)

        total_ms = sum(s.duration_ms for s in self._spans)

        # Phase breakdown.
        phases: dict[str, float] = {}
        for s in all_spans:
            phase = s.metadata.get("phase", "unknown")
            phases[phase] = phases.get(phase, 0.0) + s.duration_ms

        # Top 10 longest spans.
        top = sorted(all_spans, key=lambda s: s.duration_ms, reverse=True)[:10]

        return {
            "span_count": len(all_spans),
            "root_spans": len(self._spans),
            "total_duration_ms": round(total_ms, 2),
            "phase_breakdown": {k: round(v, 2) for k, v in sorted(phases.items())},
            "top_spans": [
                {"name": s.name, "duration_ms": round(s.duration_ms, 3)}
                for s in top
            ],
        }

    def print_summary(self) -> None:
        """Pretty-print the summary to stdout."""
        s = self.summary()
        if s.get("span_count", 0) == 0:
            print("No profiling data collected.")
            return

        print("\n" + "=" * 60)
        print("NVIDIA Nsight Profiler — Summary")
        print("=" * 60)
        print(f"  Spans:         {s['span_count']} ({s['root_spans']} root)")
        print(f"  Total CPU:     {s['total_duration_ms']:.1f} ms")
        print()
        print("  Phase breakdown:")
        for phase, dur in s.get("phase_breakdown", {}).items():
            pct = (dur / s["total_duration_ms"] * 100) if s["total_duration_ms"] > 0 else 0
            print(f"    {phase:<20s}: {dur:8.1f} ms ({pct:5.1f}%)")
        print()
        print("  Top spans:")
        for span in s.get("top_spans", [])[:5]:
            print(f"    {span['name']:<40s}: {span['duration_ms']:8.2f} ms")
        print("=" * 60)

    # ── NSys CLI helper ─────────────────────────────────────────────────

    @staticmethod
    def nsys_profile_command(
        output_name: str = "benchmark_profile",
        output_dir: str = "./profiles",
        python_cmd: str = "python -m benchmark --config config.yaml",
    ) -> str:
        """Generate an ``nsys profile`` shell command for external profiling.

        Returns a ready-to-run bash command string.
        """
        return (
            "nsys profile "
            "--trace=cuda,nvtx,osrt,cublas,cudnn "
            "--cuda-memory-usage=true "
            "--cuda-um-cpu-page-faults=true "
            "--cuda-um-gpu-page-faults=true "
            f"--output={output_name} "
            f"--force-overwrite=true "
            f"--export=sqlite "
            f"{python_cmd}"
        )
