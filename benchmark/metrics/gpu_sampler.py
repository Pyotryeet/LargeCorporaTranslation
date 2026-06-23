"""Backend-normalised device sampler — CUDA via pynvml, MPS via IOKit/psutil.

Produces identical JSON schema regardless of backend.

v2.0: MPS sampling uses IOKit via ``subprocess`` with a short timeout and
cached availability flag to avoid 5-second hangs on every sample when
powermetrics requires sudo.  CUDA sampling uses batched NVML calls to
reduce per-sample overhead.
"""

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from benchmark.hardware.backend import DeviceInfo
from benchmark.config.constants import (
    POWERMETRICS_CACHE_TTL,
    POWERMETRICS_TIMEOUT,
    METRICS_FLUSH_INTERVAL,
    DEFAULT_SAMPLE_RATE_HZ,
    MAX_METRICS_BUFFER_SIZE,
)
from benchmark.utils.json_utils import sanitized_dumps

logger = logging.getLogger(__name__)

# pynvml availability is probed lazily — importing nvml at module level
# would load the NVML shared library even when the backend is CPU or MPS.
# _PYNVML_MODULE and HAS_PYNVML are populated on first call to _get_pynvml().
_HAS_PYNVML_CACHED: bool | None = None
_PYNVML_CACHED = None


def _get_pynvml():
    """Lazy NVML import — returns ``(nvml_module, has_nvml)``.

    Prefers ``nvidia_ml_py`` (replacement for deprecated ``pynvml``).
    Falls back to ``pynvml`` for older installations.
    """
    global _PYNVML_CACHED, _HAS_PYNVML_CACHED
    if _HAS_PYNVML_CACHED is not None:
        return _PYNVML_CACHED, _HAS_PYNVML_CACHED
    _pymod = None
    for mod_name in ("nvidia_ml_py", "pynvml"):
        try:
            _pymod = __import__(mod_name)
            break
        except ImportError:
            continue
    _PYNVML_CACHED = _pymod
    _HAS_PYNVML_CACHED = _pymod is not None
    return _PYNVML_CACHED, _HAS_PYNVML_CACHED


# ---------------------------------------------------------------------------
# powermetrics pre-flight (run once, not on every sample)
# ---------------------------------------------------------------------------

# ── Module-level constants (imported from central constants) ──
_POWERCACHE_TTL = POWERMETRICS_CACHE_TTL
_POWERMETRICS_TIMEOUT = POWERMETRICS_TIMEOUT
_FLUSH_INTERVAL = METRICS_FLUSH_INTERVAL


# powermetrics output format has been verified on macOS 13 (Ventura), 14 (Sonoma),
# and 15 (Sequoia).  In Sequoia the keyword "GPU Power" may be replaced with a
# variant; we check both formats below.
_POWERMETRICS_KEYWORDS = ("GPU Active Residency", "GPU Power")


def _probe_powermetrics() -> bool:
    """Return True if ``powermetrics`` is callable and produces GPU data.

    Uses a short 100ms sample window — enough to detect GPU counters
    without blocking the sampling loop for seconds.
    """
    try:
        r = subprocess.run(
            ["powermetrics", "--samplers", "gpu_power", "-n", "1", "-i", "100"],
            capture_output=True, text=True, timeout=_POWERMETRICS_TIMEOUT,
        )
        if any(kw in r.stdout for kw in _POWERMETRICS_KEYWORDS):
            return True
        # Fallback: newer macOS may emit version-incompatibility hints on stderr
        # while still producing GPU counters under renamed headings.
        if any(kw in r.stderr for kw in ["version", "deprecated", "renamed"]):
            logger.warning("powermetrics emitted compatibility hints on stderr; probing stdout anyway")
            if r.stdout and "gpu" in r.stdout.lower():
                return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# Cached powermetrics results to avoid calling every sample (v2.0).
# powermetrics returns GPU data at roughly one sample every 5 s; we cache
# the last value and only refresh when the cache is stale.
# Module-level state: safe for single-process use. Not thread-safe for multi-harness scenarios.
_pm_cache: dict[str, tuple[float, float | None]] = {}
"""Keyword → (timestamp, value).  Values refresh when older than _POWERCACHE_TTL
to avoid spawning a subprocess on every metrics sample."""

import threading
_pm_cache_lock: Optional[threading.Lock] = None
"""Lock protecting _pm_cache against concurrent mutation and rehash during reads.
Allocated lazily on first access via ``_get_pm_cache_lock()`` to avoid creating
a threading primitive at module import time."""


def _get_pm_cache_lock() -> threading.Lock:
    global _pm_cache_lock
    if _pm_cache_lock is None:
        _pm_cache_lock = threading.Lock()
    return _pm_cache_lock


def _refresh_powermetrics_cache() -> None:
    """Run powermetrics ONCE and populate ALL keyword caches from the output.

    Previously, each keyword triggered its own subprocess.run, so two keywords
    (GPU Active Residency + GPU Power) spawned two subprocesses per refresh.
    This function runs one subprocess and parses every known keyword from it.
    Also handles macOS Sequoia output-format drift by inspecting stderr for
    compatibility hints.
    """
    now = time.monotonic()
    with _get_pm_cache_lock():
        try:
            r = subprocess.run(
                ["powermetrics", "--samplers", "gpu_power", "-n", "1", "-i", "100"],
                capture_output=True, text=True, timeout=_POWERMETRICS_TIMEOUT,
            )
            for line in r.stdout.split("\n"):
                for kw, is_pct in [("GPU Active Residency", True), ("GPU Power", False)]:
                    if kw in line:
                        raw = line.split(":")[-1].strip().replace("%", "").replace("mW", "").strip()
                        try:
                            val = float(raw) / 1000.0 if not is_pct else float(raw)
                            _pm_cache[kw] = (now, val)
                        except ValueError:
                            _pm_cache[kw] = (now, None)
            # Fallback: stderr may contain version-incompatibility hints on macOS 15+.
            # If stdout produced nothing but stderr warns about renamed keys, try a
            # best-effort parse of stdout looking for any power/residency line.
            if not _pm_cache or all(_ts == now and v is None for _ts, v in _pm_cache.values()):
                stderr_lower = r.stderr.lower()
                if any(word in stderr_lower for word in ("version", "deprecated", "renamed", "sequoia", "unsupported")):
                    logger.warning("powermetrics stderr suggests version incompatibility: %s", r.stderr[:200])
                    # Attempt a looser parse of stdout as a last resort
                    for line in r.stdout.split("\n"):
                        line_lower = line.lower()
                        if "gpu" in line_lower and (
                            "residency" in line_lower or "power" in line_lower or "active" in line_lower
                        ):
                            raw = line.split(":")[-1].strip().replace("%", "").replace("mW", "").replace("W", "").strip()
                            try:
                                val = float(raw)
                                # Guess: if raw value was multiple digits (likely mW), convert to W
                                if "mW" in line:
                                    val /= 1000.0
                                kw_match = "GPU Active Residency" if "residen" in line_lower else "GPU Power"
                                _pm_cache[kw_match] = (now, val)
                            except ValueError:
                                pass
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            # Mark all keys as stale-but-None so callers don't retry immediately.
            for kw in ["GPU Active Residency", "GPU Power"]:
                _pm_cache[kw] = (now, None)


def _get_powermetrics_value(keyword: str) -> float | None:
    """Return cached powermetrics value, refreshing when stale via ONE subprocess."""
    now = time.monotonic()
    with _get_pm_cache_lock():
        if keyword in _pm_cache:
            ts, val = _pm_cache[keyword]
            if now - ts < _POWERCACHE_TTL:
                return val
    # Any stale keyword triggers a single full refresh for all keywords
    # (acquires the lock internally; re-check after).
    _refresh_powermetrics_cache()
    with _get_pm_cache_lock():
        if keyword in _pm_cache:
            _, val = _pm_cache[keyword]
            return val
    return None


@dataclass
class DeviceSample:
    timestamp: str
    elapsed_s: float
    backend: str
    devices: list[dict]

    def to_json(self) -> str:
        return sanitized_dumps(asdict(self), ensure_ascii=False)


class DeviceSampler:
    # Cache powermetrics availability — probing once at startup avoids
    # 5-second timeouts on every sample when the tool is unavailable.
    _powermetrics_ok: bool | None = None

    def __init__(self, device_info: DeviceInfo, output_dir: Path, sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ):
        self.device_info = device_info
        self.output_dir = output_dir
        self.sample_rate_hz = sample_rate_hz
        self.sample_count = 0
        self._log_file: Optional[Path] = None
        self._start_time: Optional[float] = None
        self._buffer: list[str] = []
        self._flush_interval = _FLUSH_INTERVAL
        self._buffer_lock = threading.Lock()
        # Lazy NVML init — only loaded when the CUDA backend is active.
        self._pynvml: Optional[object] = None
        self._has_pynvml: bool = False
        if device_info.backend == "cuda":
            self._pynvml, self._has_pynvml = _get_pynvml()
            if self._has_pynvml:
                try:
                    self._pynvml.nvmlInit()
                    logger.info("NVML initialised")
                except self._pynvml.NVMLError as e:
                    logger.error(f"NVML init failed: {e}")
                    self._has_pynvml = False
        # Probe powermetrics once on MPS
        if device_info.backend == "mps" and DeviceSampler._powermetrics_ok is None:
            DeviceSampler._powermetrics_ok = _probe_powermetrics()

    def start(self, start_time: float) -> None:
        self._start_time = start_time
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_file = self.output_dir / f"device_metrics_{ts}.jsonl"
        logger.info(f"Device sampler -> {self._log_file}")

    def sample(self) -> Optional[DeviceSample]:
        if self._start_time is None:
            return None
        elapsed = time.monotonic() - self._start_time
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        if self.device_info.backend == "cuda":
            devices = self._sample_cuda()
        elif self.device_info.backend == "mps":
            devices = self._sample_mps()
        else:
            devices = self._sample_cpu()
        s = DeviceSample(timestamp=timestamp, elapsed_s=round(elapsed, 3), backend=self.device_info.backend, devices=devices)
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
                    f"dropped oldest {len(dropped)} device samples to prevent OOM"
                )

    def _sample_cuda(self) -> list[dict]:
        if not self._has_pynvml:
            return [{"id": i, "error": "pynvml not available"} for i in range(self.device_info.num_devices)]
        _pymod = self._pynvml
        devices = []
        for i in range(self.device_info.num_devices):
            try:
                h = _pymod.nvmlDeviceGetHandleByIndex(i)
                util = _pymod.nvmlDeviceGetUtilizationRates(h)
                mem = _pymod.nvmlDeviceGetMemoryInfo(h)
                devices.append({"id": i, "util_pct": float(util.gpu),
                    "mem_used_mib": int(mem.used/(1024*1024)), "mem_total_mib": int(mem.total/(1024*1024)),
                    "temp_c": self._safe(_pymod.nvmlDeviceGetTemperature, h, _pymod.NVML_TEMPERATURE_GPU),
                    "power_w": (self._safe(_pymod.nvmlDeviceGetPowerUsage, h) or 0) / 1000.0,
                    "sm_clock_mhz": self._safe(_pymod.nvmlDeviceGetClockInfo, h, _pymod.NVML_CLOCK_SM),
                    "mem_clock_mhz": self._safe(_pymod.nvmlDeviceGetClockInfo, h, _pymod.NVML_CLOCK_MEM)})
            except (_pymod.NVMLError, RuntimeError, ValueError) as e:
                devices.append({"id": i, "error": str(e)})
        return devices

    def _sample_mps(self) -> list[dict]:
        import psutil
        proc = psutil.Process()
        # On Apple Silicon unified memory, process RSS is the closest
        # approximation of "GPU memory used" — the Metal driver, MPS
        # allocator, and model weights all draw from the same physical
        # pool as the rest of the process.  system-wide virtual_memory()
        # includes UBC, kernel wired memory, and other apps, giving a
        # misleadingly high number.
        mem_total = psutil.virtual_memory().total
        mem_used = proc.memory_info().rss
        return [{"id": 0,
                 "util_pct": self._mps_util(),
                 "mem_used_mib": int(mem_used / (1024 * 1024)),
                 "mem_total_mib": int(mem_total / (1024 * 1024)),
                 "temp_c": self._mps_temp(),
                 "power_w": self._mps_power(),
                 "sm_clock_mhz": None,
                 "mem_clock_mhz": None}]

    def _sample_cpu(self) -> list[dict]:
        import psutil
        mem = psutil.virtual_memory()
        return [{"id": 0, "util_pct": float(psutil.cpu_percent(interval=None)),
                 "mem_used_mib": int(mem.used/(1024*1024)), "mem_total_mib": int(mem.total/(1024*1024)),
                 "temp_c": None, "power_w": None, "sm_clock_mhz": None, "mem_clock_mhz": None}]

    def _mps_util(self):
        if not DeviceSampler._powermetrics_ok:
            return None
        return _get_powermetrics_value("GPU Active Residency")

    def _mps_temp(self):
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                for entries in temps.values():
                    if entries:
                        return float(entries[0].current)
        except (AttributeError, OSError, RuntimeError):
            pass
        return None

    def _mps_power(self):
        if not DeviceSampler._powermetrics_ok:
            return None
        return _get_powermetrics_value("GPU Power")

    @staticmethod
    def _safe(func, *args):
        try:
            return func(*args)
        except Exception as e:
            logger.debug(
                "NVML call %s(%s) failed: %s",
                getattr(func, '__name__', func), args, e,
            )
            return None

    def __del__(self):
        try:
            self.flush()
            if self.device_info.backend == "cuda" and self._has_pynvml and self._pynvml is not None:
                try:
                    self._pynvml.nvmlShutdown()
                except self._pynvml.NVMLError:
                    pass
        except Exception:
            pass
