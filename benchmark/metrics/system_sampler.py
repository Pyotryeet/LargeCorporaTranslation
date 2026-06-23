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
    timestamp: str
    elapsed_s: float
    cpu_util_pct: float
    ram_used_mib: int
    ram_total_mib: int
    disk_read_mbps: float
    disk_write_mbps: float
    swap_used_mib: int = 0

    def to_json(self) -> str:
        return sanitized_dumps(asdict(self), ensure_ascii=False)


class SystemSampler:
    def __init__(self, output_dir: Path, sample_rate_hz: int = 1):
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
        self._buffer_lock = threading.Lock()
        # psutil.cpu_percent(interval=None) returns 0.0 on the first call because it
        # needs a prior snapshot to compute the delta.  We prime it once in start().
        self._cpu_initialized = False

    def start(self, start_time: float) -> None:
        self._start_time = start_time
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_file = self.output_dir / f"system_metrics_{ts}.jsonl"
        # Prime psutil's CPU measurement — the first cpu_percent(interval=None) call
        # always returns 0.0; this dummy call seeds the internal counter.
        psutil.cpu_percent(interval=None)
        self._cpu_initialized = True
        # Prime disk I/O baseline so the first sample() call produces real deltas.
        disk = psutil.disk_io_counters()
        if disk is not None:
            self._prev_read = disk.read_bytes
            self._prev_write = disk.write_bytes
            self._prev_time = time.monotonic()
        logger.info(f"System sampler -> {self._log_file}")

    def sample(self) -> Optional[SystemSample]:
        if self._start_time is None:
            return None
        elapsed = time.monotonic() - self._start_time
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_io_counters()
        now = time.monotonic()
        r_mbps = w_mbps = 0.0
        if disk is not None and self._prev_time > 0:
            dt = now - self._prev_time
            if dt > 0:
                r_mbps = ((disk.read_bytes - self._prev_read) / (1024*1024)) / dt
                w_mbps = ((disk.write_bytes - self._prev_write) / (1024*1024)) / dt
        if disk is not None:
            self._prev_read = disk.read_bytes
            self._prev_write = disk.write_bytes
        self._prev_time = now
        s = SystemSample(timestamp=timestamp, elapsed_s=round(elapsed, 3), cpu_util_pct=round(cpu, 1),
                        ram_used_mib=int(mem.used/(1024*1024)), ram_total_mib=int(mem.total/(1024*1024)),
                        disk_read_mbps=round(r_mbps, 2), disk_write_mbps=round(w_mbps, 2),
                        swap_used_mib=int(swap.used/(1024*1024)))
        with self._buffer_lock:
            self._buffer.append(s.to_json())
            self.sample_count += 1
        if len(self._buffer) >= self._flush_interval:
            self.flush()
        return s

    def flush(self) -> None:
        if not self._buffer or not self._log_file:
            return
        try:
            with self._buffer_lock:
                if not self._buffer:
                    return
                with open(self._log_file, "a") as f:
                    f.write("\n".join(self._buffer) + "\n")
                self._buffer.clear()
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
