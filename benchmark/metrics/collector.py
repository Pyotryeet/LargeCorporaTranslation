"""Metrics coordinator — starts/stops samplers, aggregates results.

Coordinates device (GPU) and system (CPU/RAM) sampling across separate
background threads with deadline-based scheduling to prevent drift.
Buffers samples and flushes them to disk in dedicated output directories
(gpu/, system/, batch/).  Also tracks rolling token throughput and can
push live metrics to a Prometheus exporter.
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from benchmark.hardware.backend import DeviceInfo
from benchmark.metrics.gpu_sampler import DeviceSampler
from benchmark.metrics.system_sampler import SystemSampler
from benchmark.metrics.batch_logger import BatchLogger
from benchmark.metrics.throughput import ThroughputTracker
from benchmark.config.constants import MAX_METRICS_BUFFER_SIZE

if TYPE_CHECKING:
    from benchmark.observability.prometheus_metrics import PrometheusExporter

logger = logging.getLogger(__name__)

# Maximum number of buffered samples before oldest are dropped on persistent flush failure.
# After a failed flush, if the buffer grows past this threshold, the oldest entries are
# discarded with a warning to prevent unbounded memory growth.
MAX_BUFFER_SIZE: int = MAX_METRICS_BUFFER_SIZE


class MetricsCollector:
    """Coordinates device and system metric sampling, batch logging, and throughput tracking.

    Manages background threads for periodic GPU and system health sampling,
    with deadline-based scheduling to prevent clock drift.  Aggregates throughput
    statistics and can optionally push live samples to a Prometheus exporter.

    Parameters:
        output_dir: Path -- root directory where gpu/, system/, and batch/
            subdirectories will be created for sample output files.
        device_info: DeviceInfo -- hardware backend descriptor specifying
            whether to sample GPU, MPS, or other device metrics.
        sample_rate_hz: int -- sampling frequency in Hertz (samples per second).
            Default 1 (one sample per second).

    Side effects:
        Creates gpu/, system/, and batch/ subdirectories under output_dir
        on construction.
    """

    def __init__(self, output_dir: Path, device_info: DeviceInfo, sample_rate_hz: int = 1):
        """Initialise the metrics collector with output directory, device info, and sample rate.

        Parameters:
            output_dir: Path -- root directory for metric output subdirectories.
            device_info: DeviceInfo -- hardware backend descriptor (CUDA, MPS, CPU).
            sample_rate_hz: int -- sampling frequency in Hz (default 1).

        Side effects:
            Creates gpu/, system/, and batch/ subdirectories under output_dir.
        """
        self.output_dir = output_dir
        self.device_info = device_info
        self.sample_rate_hz = sample_rate_hz
        self.gpu_dir = output_dir / "gpu"
        self.batch_dir = output_dir / "batch"
        self.system_dir = output_dir / "system"
        for d in [self.gpu_dir, self.batch_dir, self.system_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.device_sampler = DeviceSampler(device_info, self.gpu_dir, sample_rate_hz)
        self.system_sampler = SystemSampler(self.system_dir, sample_rate_hz)
        self.batch_logger = BatchLogger(self.batch_dir)
        self.throughput_tracker = ThroughputTracker(window_seconds=60)
        self._device_thread: Optional[threading.Thread] = None
        self._system_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._prometheus: Optional['PrometheusExporter'] = None  # type: ignore[valid-type]

    def set_prometheus_exporter(self, exporter: 'PrometheusExporter') -> None:  # type: ignore[valid-type]
        """Attach a PrometheusExporter so device/system samples are pushed live."""
        self._prometheus = exporter

    def start(self, run_start_time: float) -> None:
        self._running.set()
        self.device_sampler.start(run_start_time)
        self.system_sampler.start(run_start_time)
        self.batch_logger.start()
        self._device_thread = threading.Thread(target=self._device_loop, name="metrics-device", daemon=True)
        self._device_thread.start()
        self._system_thread = threading.Thread(target=self._system_loop, name="metrics-system", daemon=True)
        self._system_thread.start()
        logger.info(f"Metrics started at {self.sample_rate_hz} Hz (backend={self.device_info.backend})")

    def stop(self) -> None:
        self._running.clear()
        for t in [self._device_thread, self._system_thread]:
            if t and t.is_alive():
                t.join(timeout=10)
        # Flush after stopping threads — catches any samples buffered between
        # the thread's last interval check and actual thread exit.  A second
        # flush after join() ensures no samples are left orphaned in-memory.
        # (Each sampler's flush() is idempotent on an empty buffer.)
        self.device_sampler.flush()
        self.system_sampler.flush()
        self.batch_logger.flush()
        # Double-flush is intentional: a sampler thread may have been
        # mid-sample when _running was cleared, adding one last entry to the
        # buffer after the first flush call above released the lock.  The
        # join() ensures the thread has exited by this point, so re-flushing
        # guarantees every sample is persisted.
        self.device_sampler.flush()
        self.system_sampler.flush()
        self.batch_logger.flush()
        logger.info("Metrics collection stopped")

    def log_batch(self, batch_result) -> None:
        """Record a completed batch result for throughput tracking and batch logging.

        Parameters:
            batch_result: BatchResult -- the completed batch containing
                output_tokens_total and total_latency_ms fields used for
                throughput calculation.

        Side effects:
            Appends to the batch log on disk (via BatchLogger.log).
            Updates the rolling throughput window (via ThroughputTracker.add).
        """
        self.batch_logger.log(batch_result)
        self.throughput_tracker.add(
            batch_result.output_tokens_total,
            latency_ms=batch_result.total_latency_ms,
        )

    def get_rolling_throughput(self) -> float:
        """Return the current rolling throughput in tokens per second.

        Returns:
            float -- the current throughput averaged over the configured window
            (default 60 seconds), or 0.0 if no batches have been logged yet.
        """
        return self.throughput_tracker.current()

    def get_summary(self) -> dict:
        """Return a dictionary summarising all collected metrics.

        Returns:
            dict with keys:
                throughput -- dict: per-second and per-batch throughput statistics
                device_samples -- int: number of device (GPU) samples collected
                system_samples -- int: number of system (CPU/RAM) samples collected
                batches_logged -- int: number of batch results logged
        """
        return {"throughput": self.throughput_tracker.summary(),
                "device_samples": self.device_sampler.sample_count,
                "system_samples": self.system_sampler.sample_count,
                "batches_logged": self.batch_logger.batch_count}

    def _device_loop(self) -> None:
        """Background thread target that samples device metrics at the configured rate.

        Uses deadline-based scheduling via ``time.monotonic()`` to prevent clock
        accumulation drift over long runs.  Sleep duration is shortened (or skipped)
        when a sample takes longer than the interval, keeping the overall schedule
        stable.

        If a Prometheus exporter is attached, each sample is pushed live via
        ``record_device``.

        Parameters:
            None (reads instance state).

        Side effects:
            Writes device sample files to disk via ``DeviceSampler.sample()``.
            Pushes live metrics to Prometheus if an exporter is configured.

        Exceptions:
            Logged as warnings -- sampling errors do not crash the thread.
        """
        interval = 1.0 / self.sample_rate_hz
        next_deadline = time.monotonic() + interval
        while self._running.is_set():
            try:
                sample = self.device_sampler.sample()
                if sample is not None and self._prometheus is not None:
                    for d in sample.devices:
                        self._prometheus.record_device(
                            device_id=d.get('id', 0),
                            util_pct=d.get('util_pct'),
                            mem_used_mib=d.get('mem_used_mib'),
                            mem_total_mib=d.get('mem_total_mib'),
                            temp_c=d.get('temp_c'),
                            power_w=d.get('power_w'),
                        )
            except Exception as e:
                logger.warning(f"Device sample error: {e}")
            now = time.monotonic()
            sleep_time = next_deadline - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_deadline += interval
            # Prevent deadline runaway if we're very far behind.
            if next_deadline < now:
                next_deadline = now + interval

    def _system_loop(self) -> None:
        """Background thread target that samples system metrics at the configured rate.

        Uses the same deadline-based scheduling as ``_device_loop`` to prevent
        clock drift.  Pushes CPU utilisation, RAM usage, and swap usage to
        Prometheus if an exporter is attached.

        Parameters:
            None (reads instance state).

        Side effects:
            Writes system sample files to disk via ``SystemSampler.sample()``.
            Pushes live metrics to Prometheus if an exporter is configured.

        Exceptions:
            Logged as warnings -- sampling errors do not crash the thread.
        """
        interval = 1.0 / self.sample_rate_hz
        next_deadline = time.monotonic() + interval
        while self._running.is_set():
            try:
                sample = self.system_sampler.sample()
                if sample is not None and self._prometheus is not None:
                    self._prometheus.record_system(
                        cpu_pct=sample.cpu_util_pct,
                        ram_used_mib=sample.ram_used_mib,
                        swap_used_mib=sample.swap_used_mib,
                    )
            except Exception as e:
                logger.warning(f"System sample error: {e}")
            now = time.monotonic()
            sleep_time = next_deadline - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            next_deadline += interval
            if next_deadline < now:
                next_deadline = now + interval
