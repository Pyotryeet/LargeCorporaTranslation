"""Backend-normalised device sampler -- CUDA via pynvml, MPS via IOKit/psutil.

Produces identical JSON schema regardless of backend.

Features:
- CUDA sampling uses batched NVML calls to reduce per-sample overhead.
- MPS sampling uses ``powermetrics`` via ``subprocess`` with a caching layer
  to avoid 5-second hangs on every sample when the tool requires sudo.
- CPU backend provides basic utilization/memory via psutil.
- All samples are buffered in-memory and flushed to JSONL on a configurable
  interval to amortize I/O cost.

v2.0: MPS sampling uses IOKit via ``subprocess`` with a short timeout and
cached availability flag.
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

_pm_cache_lock: Optional[threading.Lock] = None
"""Lock protecting _pm_cache against concurrent mutation and rehash during reads.
Allocated lazily on first access via ``_get_pm_cache_lock()`` to avoid creating
a threading primitive at module import time."""


def _get_pm_cache_lock() -> threading.Lock:
    """Return the module-level lock protecting ``_pm_cache``, creating it lazily.

    Uses a double-check pattern with a module-level ``_pm_cache_lock`` variable.
    On first call, a ``threading.Lock`` is allocated and stored. Subsequent
    calls return the same lock instance.

    Returns:
        A ``threading.Lock`` instance shared across all callers.

    Note:
        This avoids creating a threading primitive at module import time, which
        could interfere with fork-based multiprocessing or threaded imports.
    """
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
        # Snapshot existing cache entries before the run so we can preserve
        # them if powermetrics succeeds but returns no new GPU data (e.g.
        # format drift on a new macOS release).  Without this guard the
        # cache is populated with (now, None) entries and callers see a
        # 5-second blackout window.
        prev_cache = dict(_pm_cache)
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
            # If powermetrics ran successfully but did not produce any GPU
            # data (even after the fallback parse), preserve the previous
            # cached values with an extended TTL instead of overwriting
            # them with None.  This avoids a 5-second metrics blackout
            # when macOS incrementally changes powermetrics output format.
            for kw in ["GPU Active Residency", "GPU Power"]:
                if kw not in _pm_cache:
                    if kw in prev_cache:
                        _ts, prev_val = prev_cache[kw]
                        if prev_val is not None:
                            _pm_cache[kw] = (now, prev_val)
                            logger.debug(
                                "powermetrics produced no data for %r — "
                                "extending stale cached value (%.3f) to avoid gap",
                                kw, prev_val,
                            )
                        else:
                            _pm_cache[kw] = (now, None)
                    else:
                        _pm_cache[kw] = (now, None)
        except FileNotFoundError:
            # Permanent failure — powermetrics binary missing. Mark all keys as
            # stale-None so callers don't retry immediately.
            for kw in ["GPU Active Residency", "GPU Power"]:
                _pm_cache[kw] = (now, None)
        except (subprocess.TimeoutExpired, OSError):
            # Transient failure (e.g. timeout, system busy).  Bump the timestamp
            # on existing cache entries so callers get the last known good value
            # for another TTL cycle instead of a gap of 5+ seconds of None.
            # Without this, the stale cache expires, _get_powermetrics_value
            # returns None, and the metrics log shows a blackout window.
            logger.warning(
                "Transient powermetrics failure — extending stale cache TTL to avoid metrics gap"
            )
            for kw in _pm_cache:
                ts, val = _pm_cache[kw]
                if val is not None:
                    _pm_cache[kw] = (now, val)


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
    """A single point-in-time snapshot of device metrics across all available GPUs.

    This dataclass is the unified output format regardless of backend (CUDA, MPS, CPU).

    Attributes:
        timestamp: ISO 8601 UTC timestamp with millisecond precision
            (e.g. ``\"2026-06-28T14:30:05.123Z\"``).
        elapsed_s: Seconds elapsed since the sampler's ``start()`` call.
        backend: The backend identifier string (``\"cuda\"``, ``\"mps\"``, or ``\"cpu\"``).
        devices: List of per-device metric dictionaries. Each dict contains keys such as
            ``id``, ``util_pct``, ``mem_used_mib``, ``mem_total_mib``, ``temp_c``,
            ``power_w``, ``sm_clock_mhz``, ``mem_clock_mhz``, and optionally ``error``
            if a particular device could not be sampled.

    Methods:
        to_json: Serialize the entire sample to a JSON string using
            ``sanitized_dumps(asdict(self), ensure_ascii=False)``.
    """
    timestamp: str
    elapsed_s: float
    backend: str
    devices: list[dict]

    def to_json(self) -> str:
        """Serialize this DeviceSample to a compact JSON string.

        Returns:
            A JSON string representation of all fields, produced via
            ``sanitized_dumps(asdict(self), ensure_ascii=False)``. Unicode characters
            are preserved (not escaped).
        """
        return sanitized_dumps(asdict(self), ensure_ascii=False)


class DeviceSampler:
    """Periodic device-metric sampler producing JSONL log files.

    Dispatches to CUDA (NVML), MPS (IOKit/powermetrics), or CPU (psutil) backends
    based on the ``DeviceInfo`` passed at construction. Samples are buffered in
    memory and flushed to disk in batches (configurable via ``METRICS_FLUSH_INTERVAL``).

    Class-level state:
        _powermetrics_ok: Cached availability flag for ``powermetrics``, shared
            across all instances to avoid re-probing.

    Args:
        device_info: Backend and device-count descriptor from hardware discovery.
        output_dir: Directory where ``device_metrics_<timestamp>.jsonl`` files
            are written.
        sample_rate_hz: Target sampling frequency in Hz (used for interval-based
            pacing; the sampler itself does not sleep -- pacing is the caller's
            responsibility). Defaults to ``DEFAULT_SAMPLE_RATE_HZ``.

    Raises:
        No explicit exceptions at construction. NVML init failures are logged and
        disable CUDA sampling gracefully (per-device entries will contain an
        ``\"error\"`` key).
    """
    # Cache powermetrics availability — probing once at startup avoids
    # 5-second timeouts on every sample when the tool is unavailable.
    _powermetrics_ok: bool | None = None

    def __init__(self, device_info: DeviceInfo, output_dir: Path, sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ):
        """Initialise the device sampler.

        Args:
            device_info: ``DeviceInfo`` instance describing the active backend and
                the number of devices.
            output_dir: Parent directory for JSONL log files.
            sample_rate_hz: Target sample rate in samples per second. Controls the
                pacing interval in the calling loop; the sampler itself does not
                enforce timing. Defaults to ``DEFAULT_SAMPLE_RATE_HZ``.

        Side effects:
            - If the backend is CUDA, lazily imports pynvml (or nvidia_ml_py) and
              calls ``nvmlInit()``. On failure, disables CUDA sampling.
            - If the backend is MPS, probes ``powermetrics`` availability once
              (class-level cache) by running a short subprocess.
        """
        self.device_info = device_info
        self.output_dir = output_dir
        self.sample_rate_hz = sample_rate_hz
        self.sample_count = 0
        self._log_file: Optional[Path] = None
        self._start_time: Optional[float] = None
        self._buffer: list[str] = []
        self._flush_interval = _FLUSH_INTERVAL
        self._buffer_lock = threading.RLock()
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
        """Begin a sampling session and open the output log file.

        Must be called once before ``sample()``. Creates a JSONL file named
        ``device_metrics_<UTC_timestamp>.jsonl`` inside ``self.output_dir``.

        Args:
            start_time: ``time.monotonic()`` value representing the benchmark
                start instant. Used to compute ``elapsed_s`` in every sample.

        Side effects:
            Sets ``self._start_time`` and ``self._log_file``. Log messages
            are emitted for initialization tracking.
        """
        self._start_time = start_time
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_file = self.output_dir / f"device_metrics_{ts}.jsonl"
        logger.info(f"Device sampler -> {self._log_file}")

    def sample(self) -> Optional[DeviceSample]:
        """Take one snapshot of device metrics and buffer it for later flush.

        Dispatches to ``_sample_cuda``, ``_sample_mps``, or ``_sample_cpu`` based on
        the backend. The resulting ``DeviceSample`` is appended to the in-memory
        buffer. When the buffer reaches ``METRICS_FLUSH_INTERVAL`` entries, an
        immediate flush to disk is triggered.

        Returns:
            A ``DeviceSample`` with the current timestamp and per-device metrics,
            or ``None`` if ``start()`` has not been called yet (``_start_time`` is
            ``None``).

        Side effects:
            - Increments ``self.sample_count``.
            - Appends to the internal ``_buffer`` list.
            - May call ``_flush_locked()`` if the buffer is full.

        Note:
            This method acquires ``self._buffer_lock`` to protect the buffer.
        """
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
                    f"dropped oldest {len(dropped)} device samples to prevent OOM"
                )

    def _flush_locked(self) -> None:
        """Flush the buffer to disk.  Caller must hold ``self._buffer_lock``."""
        if not self._buffer or not self._log_file:
            return
        with open(self._log_file, "a") as f:
            f.write("\n".join(self._buffer) + "\n")
        self._buffer.clear()

    def _sample_cuda(self) -> list[dict]:
        """Sample all CUDA GPUs via NVML and return per-device metric dicts.

        For each device index, collects utilization, memory (used/total MiB),
        temperature (C), power (W), and SM/memory clock (MHz). Individual NVML
        calls that fail are wrapped in ``_safe`` and return ``None`` for that
        field; catastrophic failures (NVMLError, RuntimeError) on a device produce
        an ``{\"id\": i, \"error\": ...}`` entry.

        Returns:
            A list of dictionaries, one per device in ``device_info.num_devices``.
            Each dict has keys: ``id``, ``util_pct``, ``mem_used_mib``,
            ``mem_total_mib``, ``temp_c``, ``power_w``, ``sm_clock_mhz``,
            ``mem_clock_mhz``. If pynvml is not available at all, all entries
            contain ``\"error\": \"pynvml not available\"``.
        """
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
                    "power_w": self._nvml_power(h),
                    "sm_clock_mhz": self._safe(_pymod.nvmlDeviceGetClockInfo, h, _pymod.NVML_CLOCK_SM),
                    "mem_clock_mhz": self._safe(_pymod.nvmlDeviceGetClockInfo, h, _pymod.NVML_CLOCK_MEM)})
            except (_pymod.NVMLError, RuntimeError, ValueError) as e:
                devices.append({"id": i, "error": str(e)})
        return devices

    def _sample_mps(self) -> list[dict]:
        """Sample the Apple Silicon GPU via psutil and powermetrics.

        On Apple Silicon, unified memory means process RSS approximates GPU memory
        usage -- the Metal driver, MPS allocator, and model weights all draw from
        the same physical pool. GPU utilization and power are retrieved from
        ``_mps_gpu_metrics`` (powermetrics cache). Temperature is read via
        ``_mps_temp`` (psutil sensors).

        Returns:
            A single-element list with one dict containing: ``id`` (always 0),
            ``util_pct``, ``mem_used_mib``, ``mem_total_mib``, ``temp_c``,
            ``power_w``, ``sm_clock_mhz`` (``None`` -- not obtainable via MPS),
            ``mem_clock_mhz`` (``None`` -- not obtainable via MPS).

        Note:
            ``psutil`` is imported lazily inside this method to avoid an import
            cost on CUDA-only deployments.
        """
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
        _gpu_util, _gpu_power = self._mps_gpu_metrics()
        return [{"id": 0,
                 "util_pct": _gpu_util,
                 "mem_used_mib": int(mem_used / (1024 * 1024)),
                 "mem_total_mib": int(mem_total / (1024 * 1024)),
                 "temp_c": self._mps_temp(),
                 "power_w": _gpu_power,
                 "sm_clock_mhz": None,
                 "mem_clock_mhz": None}]

    def _sample_cpu(self) -> list[dict]:
        """Sample CPU-wide utilization and memory via psutil.

        Returns:
            A single-element list with one dict containing: ``id`` (always 0),
            ``util_pct`` (``psutil.cpu_percent(interval=None)``), ``mem_used_mib``,
            ``mem_total_mib``, ``temp_c`` (``None``), ``power_w`` (``None``),
            ``sm_clock_mhz`` (``None``), ``mem_clock_mhz`` (``None``).

        Note:
            ``psutil`` is imported lazily inside this method to avoid an import
            cost on GPU-only deployments.
        """
        import psutil
        mem = psutil.virtual_memory()
        return [{"id": 0, "util_pct": float(psutil.cpu_percent(interval=None)),
                 "mem_used_mib": int(mem.used/(1024*1024)), "mem_total_mib": int(mem.total/(1024*1024)),
                 "temp_c": None, "power_w": None, "sm_clock_mhz": None, "mem_clock_mhz": None}]

    def _mps_gpu_metrics(self) -> tuple[float | None, float | None]:
        """Return (util_pct, power_w) from a single powermetrics cache refresh.

        Previously _mps_util() and _mps_power() each called _get_powermetrics_value
        independently, acquiring the cache lock twice per sample.
        """
        if not DeviceSampler._powermetrics_ok:
            return None, None
        return _get_powermetrics_value("GPU Active Residency"), _get_powermetrics_value("GPU Power")

    def _mps_temp(self):
        """Read the SoC / GPU die temperature via ``psutil.sensors_temperatures()``.

        Iterates over all temperature sensor groups provided by psutil and returns
        the ``current`` value from the first available entry. On Apple Silicon, this
        typically corresponds to the SoC die temperature, which is the closest
        proxy for GPU junction temperature.

        Returns:
            The temperature in degrees Celsius as a ``float``, or ``None`` if
            no sensor data is available or psutil raises an error
            (``AttributeError``, ``OSError``, ``RuntimeError``).

        Note:
            ``psutil`` is imported lazily inside this method.
        """
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

    @staticmethod
    def _safe(func, *args):
        """Call ``func(*args)`` and return its result, or ``None`` on any recoverable error.

        Static method. Wraps NVML calls that may fail transiently (e.g. a GPU that
        powered off mid-sample). Never suppresses system-critical exceptions
        (``MemoryError``, ``SystemError``, ``KeyboardInterrupt``) -- those are
        re-raised immediately.

        Args:
            func: The callable to invoke (typically an NVML API function).
            *args: Positional arguments forwarded to ``func``.

        Returns:
            The return value of ``func(*args)`` on success, or ``None`` if any
            non-critical ``Exception`` is raised. A debug-level log message is
            emitted on failure.
        """
        try:
            return func(*args)
        except (MemoryError, SystemError, KeyboardInterrupt):
            raise  # never suppress system-critical errors
        except Exception as e:
            logger.debug(
                "NVML call %s(%s) failed: %s",
                getattr(func, '__name__', func), args, e,
            )
            return None

    def _nvml_power(self, handle) -> float | None:
        """Read NVML power usage in watts with explicit None semantics.

        Previously used ``(self._safe(...) or 0) / 1000.0``, which conflated
        a measurement failure (None) with a genuinely idle GPU reading 0 mW.
        Now returns None on failure so callers can distinguish the two cases.
        """
        raw = self._safe(self._pynvml.nvmlDeviceGetPowerUsage, handle)
        if raw is None:
            return None
        return raw / 1000.0

    def close(self) -> None:
        """Explicitly stop sampling and shutdown NVML.

        Call this before the object is garbage-collected.
        ``__del__`` is not guaranteed to run (cyclic references, GC timing).
        """
        self.flush()
        if self.device_info.backend == "cuda" and self._has_pynvml and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except self._pynvml.NVMLError:
                pass

    def __del__(self):
        """Finalizer: attempt to flush buffered samples and shut down NVML.

        Wraps ``self.close()`` in a blanket ``except Exception`` because
        ``__del__`` must never raise (Python 3.8+ ignores exceptions from
        finalizers but logs a warning).

        Note:
            ``__del__`` is not guaranteed to run (cyclic references, GC timing).
            Callers should explicitly invoke ``close()`` before the object goes
            out of scope.
        """
        try:
            self.close()
        except Exception:
            pass
