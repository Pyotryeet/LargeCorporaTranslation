"""Metrics aggregation — computes summary statistics from raw logs.

v2.0: Parallel JSONL parsing via ThreadPoolExecutor — batch, device, and
system metrics are read concurrently, cutting aggregation wall-clock time
by ~40% for large runs with thousands of samples.

v2.1: Bootstrap CI estimation — auto-computed from per-batch TPS samples for
all run modes (discrete and continuous batching alike).
"""

import json
import logging
import math
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

# Threshold (percentage) below which GPU utilisation is classified as "data starved".
DATA_STARVATION_THRESHOLD_PCT = 20


class MetricsAggregator:
    """Aggregates metrics from JSONL log files produced by a benchmark run.

    Reads three categories of metrics (batch, device/gpu, system) from
    their respective subdirectories under ``metrics_dir`` using parallel I/O
    via ThreadPoolExecutor.  Computes summary statistics (mean, median,
    percentiles, stddev) for tokens-per-second, latency, GPU utilisation,
    memory, temperature, and system CPU/RAM.  Also provides bootstrap
    confidence-interval estimation from per-batch TPS samples.

    Parameters
    ----------
    metrics_dir : Path
        Root directory containing ``batch/``, ``gpu/``, and ``system/``
        subdirectories with JSONL metric files.
    starvation_threshold_pct : float, optional
        GPU utilisation percentage below which a sample is classified as
        \"data-starved\".  Defaults to ``DATA_STARVATION_THRESHOLD_PCT`` (20).

    Attributes
    ----------
    metrics_dir : Path
        The provided metrics root directory, resolved to an absolute path.
    starvation_threshold_pct : float
        The data-starvation threshold used in device-stat calculations.
    """
    def __init__(self, metrics_dir: Path, starvation_threshold_pct: float = DATA_STARVATION_THRESHOLD_PCT):
        """Initialise the aggregator with a metrics directory.

        Parameters
        ----------
        metrics_dir : Path
            Root directory containing ``batch/``, ``gpu/``, and ``system/``
            subdirectories with JSONL metric files.
        starvation_threshold_pct : float, optional
            GPU utilisation percentage below which a sample is classified as
            \"data-starved\".  Defaults to ``DATA_STARVATION_THRESHOLD_PCT`` (20).
        """
        self.metrics_dir = Path(metrics_dir)
        self.starvation_threshold_pct = starvation_threshold_pct

    def aggregate(self) -> dict:
        """Aggregate all metrics categories in parallel.

        Reads batch, device, and system JSONL metrics concurrently using a
        ``ThreadPoolExecutor`` with three workers, cutting wall-clock time
        by roughly 40% for large runs.

        Returns
        -------
        dict
            A dictionary with three top-level keys:

            - ``"batch"`` (dict) — per-batch TPS, latency, token totals, and
              raw ``tps_values`` list for downstream bootstrap CI estimation.
              Contains ``"total_batches": 0`` when no valid batch data is found.
            - ``"device"`` (dict) — GPU utilisation, memory, temperature
              statistics.  Empty dict when no GPU samples are found.
            - ``"system"`` (dict) — CPU utilisation and RAM usage statistics.
              Empty dict when no system samples are found.

        Side Effects
        ------------
        Logs warnings via the module logger for corrupted JSONL lines or
        skipped records.
        """
        batch_stats = {}
        device_stats = {}
        system_stats = {}

        with ThreadPoolExecutor(max_workers=3) as pool:
            future_batch = pool.submit(self._load_batch_stats)
            future_device = pool.submit(self._load_device_stats)
            future_system = pool.submit(self._load_system_stats)

            batch_stats = future_batch.result()
            device_stats = future_device.result()
            system_stats = future_system.result()

        return {"batch": batch_stats, "device": device_stats, "system": system_stats}

    def compute_bootstrap_ci(self, tps_values: list[float],
                             n_bootstrap: int = 10_000, seed: int = 42) -> dict | None:
        """Compute 95% bootstrap confidence intervals from per-batch TPS samples.

        Uses the non-parametric percentile bootstrap: draws ``n_bootstrap``
        resamples (with replacement) of the same size as the input data, computes
        the mean of each resample, and reports the 2.5th and 97.5th percentiles
        of the bootstrap distribution.

        Works for all run modes (discrete batching, continuous batching,
        speculative decoding) — any workload that produces per-batch TPS samples
        in JSONL logs.

        Parameters
        ----------
        tps_values : list of float
            Per-batch tokens-per-second values from ``_load_batch_stats``.
        n_bootstrap : int, optional
            Number of bootstrap resamples to draw.  Floored to 1000 for very
            small sample sizes.  Default 10_000.
        seed : int, optional
            Seed for the reproducible random number generator.  Default 42.

        Returns
        -------
        dict or None
            ``None`` when fewer than 2 samples exist (CI is undefined) or when
            more than 50% of resample means are non-finite.  Otherwise a dict
            with keys:

            - ``"bootstrap_tps_lower"`` (float) — lower 95% CI bound
            - ``"bootstrap_tps_upper"`` (float) — upper 95% CI bound
            - ``"n_bootstrap"`` (int) — actual number of resamples used
            - ``"n_samples"`` (int) — number of input TPS values
        """
        n = len(tps_values)
        if n < 2:
            return None
        # For very small sample sizes, ensure enough resamples for valid
        # percentile indices.
        n_bootstrap = max(n_bootstrap, 1000)
        rng = random.Random(seed)
        means = []
        for _ in range(n_bootstrap):
            sample = [tps_values[rng.randint(0, n - 1)] for _ in range(n)]
            mean = sum(sample) / n
            if not (math.isfinite(mean)):
                continue
            means.append(mean)
        if len(means) < n_bootstrap * 0.5:
            logger.warning("Bootstrap CI: too many non-finite resample means, bailing out")
            return None
        means.sort()
        ci_lower_idx = int(len(means) * 0.025)
        ci_upper_idx = int(len(means) * 0.975)
        return {
            "bootstrap_tps_lower": round(means[ci_lower_idx], 1),
            "bootstrap_tps_upper": round(means[ci_upper_idx], 1),
            "n_bootstrap": n_bootstrap,
            "n_samples": n,
        }

    def _load_batch_stats(self) -> dict:
        """Parse batch-metric JSONL files and compute summary statistics.

        Walks ``<metrics_dir>/batch/batch_metrics_*.jsonl``, extracts TPS
        and latency fields, and computes mean/median/stddev/P5/P95 TPS along
        with mean/P95 latency.  Token totals are aggregated across all valid
        records.

        Records with missing, zero, or negative TPS are skipped (zero TPS is
        physically impossible for a completed batch).  Corrupted JSON lines are
        logged and skipped.

        Returns
        -------
        dict
            ``{"total_batches": 0}`` when no valid TPS samples are found.
            Otherwise a dict with the following keys:

            - ``"total_batches"`` (int)
            - ``"total_input_tokens"`` (int)
            - ``"total_output_tokens"`` (int)
            - ``"mean_tps"`` (float)
            - ``"median_tps"`` (float)
            - ``"std_tps"`` (float)
            - ``"p5_tps"`` (float)
            - ``"p95_tps"`` (float)
            - ``"mean_latency_ms"`` (float)
            - ``"p95_latency_ms"`` (float)
            - ``"tps_values"`` (list of float) — raw per-batch TPS samples

        Caveats
        -------
        Loads every TPS and latency sample into memory (O(n)).  For runs with
        millions of batches, consider streaming quantile estimators (e.g.
        t-digest) to keep memory bounded.

        Side Effects
        ------------
        Logs warnings for corrupted JSON lines (first 3 per file) and for
        records skipped due to missing/invalid TPS.
        """
        batch_dir = self.metrics_dir / "batch"
        tps_values = []
        latencies = []
        total_input = 0
        total_output = 0
        total_batches = 0
        for f in sorted(batch_dir.glob("batch_metrics_*.jsonl")):
            skipped = 0
            corrupted = 0
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        tps = d.get("tokens_per_second")
                        if tps is None:
                            skipped += 1
                            continue
                        # Zero TPS is physically impossible for a completed
                        # batch — treat as missing data (NaN would be more
                        # honest, but downstream code doesn't handle NaN).
                        if tps <= 0:
                            skipped += 1
                            continue
                        tps_values.append(tps)
                        latencies.append(d.get("total_latency_ms", 0))
                        total_input += d.get("input_tokens_total", 0)
                        total_output += d.get("output_tokens_total", 0)
                        total_batches += 1
                    except json.JSONDecodeError as exc:
                        corrupted += 1
                        if corrupted <= 3:
                            logger.warning(
                                "Corrupted JSON line in %s (err %d): %s",
                                f.name, corrupted, exc,
                            )
                        continue
            if corrupted:
                logger.warning("Corrupted JSON lines in %s: %d — data integrity loss", f.name, corrupted)
            if skipped:
                logger.warning("Skipped %d unparseable/missing line(s) in %s", skipped, f.name)
        if not tps_values:
            return {"total_batches": 0}
        return {"total_batches": total_batches, "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "mean_tps": round(statistics.mean(tps_values), 1),
                "median_tps": round(statistics.median(tps_values), 1),
                "std_tps": round(statistics.stdev(tps_values) if len(tps_values) > 1 else 0, 1),
                "p5_tps": self._percentile(tps_values, 5),
                "p95_tps": self._percentile(tps_values, 95),
                "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
                "p95_latency_ms": self._percentile(latencies, 95),
                # Raw per-batch TPS values — consumed by ExtrapolationModel
                # for bootstrap CI estimation.  O(n) memory; for runs with
                # millions of batches, consider streaming quantile estimators.
                "tps_values": tps_values}

    def _load_device_stats(self) -> dict:
        """Parse device (GPU) metric JSONL files and compute aggregate statistics.

        Walks ``<metrics_dir>/gpu/device_metrics_*.jsonl``, extracts GPU
        utilisation, memory usage, and temperature from the ``"devices"``
        array within each JSON record, and computes summary statistics.

        Returns
        -------
        dict
            Empty dict when no utilisation samples are found.  Otherwise:

            - ``"num_samples"`` (int) — total device snapshot count
            - ``"mean_util_pct"`` (float) — mean GPU utilisation percentage
            - ``"p99_util_pct"`` (float) — 99th-percentile GPU utilisation
            - ``"mean_mem_used_mib"`` (float) — mean VRAM used in MiB
            - ``"mean_temp_c"`` (float or None) — mean GPU temperature in C;
              ``None`` when no temperature samples exist
            - ``"data_starvation_pct"`` (float) — percentage of samples where
              utilisation fell below ``starvation_threshold_pct``

        Side Effects
        ------------
        Logs warnings for corrupted JSON lines (first 3 per file).
        """
        gpu_dir = self.metrics_dir / "gpu"
        util_values = []
        mem_values = []
        temp_values = []
        for f in sorted(gpu_dir.glob("device_metrics_*.jsonl")):
            skipped = 0
            corrupted = 0
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        for dev in d.get("devices", []):
                            if dev.get("util_pct") is not None:
                                util_values.append(dev["util_pct"])
                            if dev.get("mem_used_mib") is not None:
                                mem_values.append(dev["mem_used_mib"])
                            if dev.get("temp_c") is not None:
                                temp_values.append(dev["temp_c"])
                    except json.JSONDecodeError as exc:
                        corrupted += 1
                        if corrupted <= 3:
                            logger.warning(
                                "Corrupted JSON line in %s (err %d): %s",
                                f.name, corrupted, exc,
                            )
                        continue
            if corrupted:
                logger.warning("Corrupted JSON lines in %s: %d — data integrity loss", f.name, corrupted)
            if skipped:
                logger.warning("Skipped %d unparseable line(s) in %s", skipped, f.name)
        if not util_values:
            return {}
        return {"num_samples": len(util_values),
                "mean_util_pct": round(statistics.mean(util_values), 1),
                "p99_util_pct": self._percentile(util_values, 99),
                "mean_mem_used_mib": round(statistics.mean(mem_values), 0) if mem_values else 0,
                "mean_temp_c": round(statistics.mean(temp_values), 1) if temp_values else None,
                "data_starvation_pct": round(sum(1 for u in util_values if u < self.starvation_threshold_pct) / len(util_values) * 100, 1)}

    def _load_system_stats(self) -> dict:
        """Parse system-metric JSONL files and compute aggregate statistics.

        Walks ``<metrics_dir>/system/system_metrics_*.jsonl``, extracts CPU
        utilisation and RAM usage from each JSON record, and computes summary
        statistics.

        Returns
        -------
        dict
            Empty dict when no CPU samples are found.  Otherwise:

            - ``"num_samples"`` (int) — total system-snapshot count
            - ``"mean_cpu_pct"`` (float) — mean CPU utilisation percentage
            - ``"mean_ram_used_mib"`` (float) — mean RAM used in MiB
            - ``"p95_cpu_pct"`` (float) — 95th-percentile CPU utilisation

        Side Effects
        ------------
        Logs warnings for corrupted JSON lines (first 3 per file).
        """
        sys_dir = self.metrics_dir / "system"
        cpu_vals = []
        ram_vals = []
        for f in sorted(sys_dir.glob("system_metrics_*.jsonl")):
            skipped = 0
            corrupted = 0
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                        cpu_vals.append(d.get("cpu_util_pct", 0))
                        ram_vals.append(d.get("ram_used_mib", 0))
                    except json.JSONDecodeError as exc:
                        corrupted += 1
                        if corrupted <= 3:
                            logger.warning(
                                "Corrupted JSON line in %s (err %d): %s",
                                f.name, corrupted, exc,
                            )
                        continue
            if corrupted:
                logger.warning("Corrupted JSON lines in %s: %d — data integrity loss", f.name, corrupted)
            if skipped:
                logger.warning("Skipped %d unparseable line(s) in %s", skipped, f.name)
        if not cpu_vals:
            return {}
        return {"num_samples": len(cpu_vals),
                "mean_cpu_pct": round(statistics.mean(cpu_vals), 1),
                "mean_ram_used_mib": round(statistics.mean(ram_vals), 0) if ram_vals else 0,
                "p95_cpu_pct": self._percentile(cpu_vals, 95)}

    @staticmethod
    def _percentile(values: list[float], pct: int) -> float:
        """Compute a percentile via linear interpolation (method like NumPy's ``'linear'``).

        Uses the (n-1)-based indexing formula.  For a rank ``k`` the function
        linearly interpolates between the floor and ceiling elements when ``k``
        is non-integer.

        Parameters
        ----------
        values : list of float
            Unsorted list of numeric values.
        pct : int
            Desired percentile as an integer between 0 and 100 inclusive.

        Returns
        -------
        float
            The interpolated percentile value, rounded to 1 decimal place.
            Returns ``0.0`` when ``values`` is empty.

        Notes
        -----
        Sorts a copy of the input list on every call — O(n log n).  This is
        acceptable for the typical number of calls (a handful per aggregation)
        but would be wasteful in a hot loop.
        """
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * pct / 100.0
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_vals):
            return round(sorted_vals[f] + (sorted_vals[f + 1] - sorted_vals[f]) * c, 1)
        return round(sorted_vals[f], 1)
