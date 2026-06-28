"""System-level metrics — CPU, RAM, disk I/O via psutil (cross-platform)."""

import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import psutil

from benchmark.utils.json_utils import sanitized_dumps
from benchmark.config.constants import MAX_METRICS_BUFFER_SIZE, METRICS_FLUSH_INTERVAL

logger = logging.getLogger(__name__)


@dataclass
class SystemSample:
    """A single snapshot of system-level metrics captured at a point in time.

    A dataclass holding CPU utilization, RAM usage, disk I/O throughput, and
    swap usage for one sampling instant.  Also provides JSON serialization
    via :meth:`to_json`.

    Attributes:
        timestamp: ISO-8601 UTC timestamp with millisecond precision (str).
        elapsed_s: Wall-clock seconds since the sampler was started (float).
        cpu_util_pct: CPU utilization as a percentage (0-100) (float).
        ram_used_mib: RAM currently in use, in mebibytes (int).
        ram_total_mib: Total physical RAM, in mebibytes (int).
        disk_read_mbps: Disk read throughput in mebibytes per second (float).
        disk_write_mbps: Disk write throughput in mebibytes per second (float).
        swap_used_mib: Swap space currently in use, in mebibytes. Defaults to 0 (int).
    """

    timestamp: str
    elapsed_s: float
    cpu_util_pct: float
    ram_used_mib: int
    ram_total_mib: int
    disk_read_mbps: float
    disk_write_mbps: float
    swap_used_mib: int = 0

    def to_json(self) -> str:
        """Serialize this sample to a JSON string.

        Returns:
            str: A JSON object string with all dataclass fields. Values are sanitized
            (non-finite floats become null, non-serializable types are converted to
            strings) via :func:`benchmark.utils.json_utils.sanitized_dumps`.
        """
        return sanitized_dumps(asdict(self), ensure_ascii=False)


class SystemSampler:
    def __init__(self, output_dir: Path, sample_rate_hz: int = 1):
        """Initialize the system sampler.

        Args:
            output_dir: Directory where the system-metrics JSONL log file will be
                written. The file is created on :meth:`start`.
            sample_rate_hz: Target sampling frequency in Hz. Defaults to 1 (once per
                second). This value is stored for reference but the caller is
                responsible for pacing calls to :meth:`sample`.
        """
        self.output_dir = output_dir
        self.sample_rate_hz = sample_rate_hz
        self.sample_count = 0
        self._log_file: Optional[Path] = None
        self._start_time: Optional[float] = None
        self._buffer: list[str] = []
        self._flush_interval = METRICS_FLUSH_INTERVAL
        self._prev_read = 0
        self._prev_write = 0
        self._prev_time = 0.0
        self._buffer_lock = threading.RLock()
        # psutil.cpu_percent(interval=None) returns 0.0 on the first call because it
        # needs a prior snapshot to compute the delta.  We prime it once in start().
        self._cpu_initialized = False

    def start(self, start_time: float) -> None:
        """Begin system metric collection.

        Opens a new timestamped JSONL log file in ``output_dir`` and primes the
        disk I/O baseline counters so that the first call to :meth:`sample`
        produces meaningful throughput deltas.

        CPU priming is deferred to the first :meth:`sample` call (see that
        method for rationale).

        Args:
            start_time: The reference ``time.monotonic()`` value used to compute
                ``elapsed_s`` for every subsequent sample.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_file = self.output_dir / f"system_metrics_{ts}.jsonl"
        self._start_time = start_time
        # Prime disk I/O baseline so the first sample() call produces real deltas.
        disk = psutil.disk_io_counters()
        if disk is not None:
            self._prev_read = disk.read_bytes
            self._prev_write = disk.write_bytes
            self._prev_time = time.monotonic()
        # CPU priming is deferred to the first sample() call so the measurement
        # interval is exactly the sampling interval, not the variable gap between
        # start() and the first sample() thread iteration.
        logger.info(f"System sampler -> {self._log_file}")

    def sample(self) -> Optional[SystemSample]:
        """Capture one system-metrics snapshot.

        Reads CPU utilization, virtual memory, swap, and disk I/O counters via
        psutil and produces a :class:`SystemSample`.  On the very first call it
        primes the CPU measurement so subsequent calls produce non-zero deltas.

        The sample is appended to an in-memory buffer.  When the buffer reaches
        ``_flush_interval`` entries, the buffer is flushed to disk automatically.

        Returns:
            Optional[SystemSample]: The captured sample, or ``None`` if
            :meth:`start` has not been called yet.

        Side effects:
            - Updates internal disk I/O baselines (``_prev_read``, ``_prev_write``,
              ``_prev_time``) for the next call.
            - Increments ``sample_count``.
            - May write to the JSONL log file if the buffer threshold is reached.
        """
        if self._start_time is None:
            return None
        wall_now = time.monotonic()
        elapsed = wall_now - self._start_time
        utc_now = datetime.now(timezone.utc)
        timestamp = utc_now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc_now.microsecond // 1000:03d}Z"
        # Prime the CPU measurement on the first sample call so the interval
        # between samples matches the actual sampling period, not the variable
        # gap between start() and the first thread iteration.
        if not self._cpu_initialized:
            psutil.cpu_percent(interval=None)
            self._cpu_initialized = True
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_io_counters()
        r_mbps = w_mbps = 0.0
        if disk is not None and self._prev_time > 0:
            dt = wall_now - self._prev_time
            if dt > 0:
                # Guard against counter wrap: if the counter has wrapped
                # (e.g. on 32-bit systems), treat the sample as 0 delta.
                read_delta = disk.read_bytes - self._prev_read
                if read_delta < 0:
                    logger.warning(
                        "Disk read counter appears to have wrapped "
                        "(prev=%d, cur=%d, delta=%d). Treating as 0.",
                        self._prev_read, disk.read_bytes, read_delta,
                    )
                    read_delta = 0
                write_delta = disk.write_bytes - self._prev_write
                if write_delta < 0:
                    logger.warning(
                        "Disk write counter appears to have wrapped "
                        "(prev=%d, cur=%d, delta=%d). Treating as 0.",
                        self._prev_write, disk.write_bytes, write_delta,
                    )
                    write_delta = 0
                r_mbps = (read_delta / (1024*1024)) / dt
                w_mbps = (write_delta / (1024*1024)) / dt
        if disk is not None:
            self._prev_read = disk.read_bytes
            self._prev_write = disk.write_bytes
        self._prev_time = wall_now
        s = SystemSample(timestamp=timestamp, elapsed_s=round(elapsed, 3), cpu_util_pct=round(cpu, 1),
                        ram_used_mib=int(mem.used/(1024*1024)), ram_total_mib=int(mem.total/(1024*1024)),
                        disk_read_mbps=round(r_mbps, 2), disk_write_mbps=round(w_mbps, 2),
                        swap_used_mib=int(swap.used/(1024*1024)))
        with self._buffer_lock:
            self._buffer.append(s.to_json())
            self.sample_count += 1
            if len(self._buffer) >= self._flush_interval:
                self._flush_locked()
        return s

    def flush(self) -> None:
        """Public flush — acquires the buffer lock."""
        if not self._buffer or not self._log_file:
            return
        try:
            with self._buffer_lock:
                self._flush_locked()
        except (PermissionError, FileNotFoundError, OSError) as e:
            logger.error(f"Flush failed — keeping buffer for next retry ({len(self._buffer)} entries): {e}")
            # Prevent unbounded buffer growth on persistent flush failures.
            excess = len(self._buffer) - MAX_METRICS_BUFFER_SIZE
            if excess > 0:
                with self._buffer_lock:
                    dropped = self._buffer[:excess]
                    self._buffer = self._buffer[excess:]
                logger.warning(
                    f"Buffer exceeded MAX_METRICS_BUFFER_SIZE ({MAX_METRICS_BUFFER_SIZE}); "
                    f"dropped oldest {len(dropped)} system samples to prevent OOM"
                )

    def _flush_locked(self) -> None:
        """Flush the buffer to disk.  Caller must hold ``self._buffer_lock``."""
        if not self._buffer or not self._log_file:
            return
        with open(self._log_file, "a") as f:
            f.write("\n".join(self._buffer) + "\n")
        self._buffer.clear()
