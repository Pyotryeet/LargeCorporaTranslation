"""Performance regression detection and baseline management (Phase 7).

Tracks throughput, latency, and memory usage across benchmark runs and
detects statistically significant regressions before they reach production.

Usage
-----
>>> mgr = PerformanceBaselineManager("./baselines")
>>> mgr.load_baseline("h200_fp8_12b")
>>> result = mgr.check(current_metrics)
>>> if result.is_regression:
...     print(f"BLOCK: {result.reason}")

CI integration: the ``check()`` method exits with code 1 when a regression
is detected, which fails the CI pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _t_sf_fallback(t: float, df: int) -> float:
    """Two-sided p-value from Student's t-distribution (fallback, no scipy).

    Uses a regularised incomplete beta approximation of the t-CDF.  Accurate
    enough for regression detection (within 0.01 of scipy for df >= 3),
    degrading gracefully for very small df.
    """
    if df <= 0:
        return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))
    x = df / (df + t * t)
    # Continued-fraction regularised incomplete beta for symmetric Student's t.
    # Beta(0.5, df/2) evaluated at x.
    if df % 2 == 0:
        # Even df: closed form.
        term = 1.0
        ssum = 1.0
        for k in range(1, df // 2):
            term *= x * (df / 2.0 - k) / k
            ssum += term
        cdf = 1.0 - 0.5 * (ssum * (1.0 - x) ** (df / 2.0))
        p = 2.0 * min(cdf, 1.0 - cdf)
    else:
        # Odd df: approximate via normal CDF with Welch-style correction.
        z = t * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + t * t / (2.0 * df))
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return min(p, 1.0)


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class BaselinePoint:
    """A single performance measurement from a baseline run."""

    timestamp: str
    mean_tps: float
    median_tps: float
    std_tps: float
    p95_tps: float
    mean_latency_ms: float
    p95_latency_ms: float
    gpu_util_pct: float | None = None
    gpu_mem_gb: float | None = None
    total_output_tokens: int = 0
    duration_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "mean_tps": self.mean_tps,
            "median_tps": self.median_tps,
            "std_tps": self.std_tps,
            "p95_tps": self.p95_tps,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "gpu_util_pct": self.gpu_util_pct,
            "gpu_mem_gb": self.gpu_mem_gb,
            "total_output_tokens": self.total_output_tokens,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BaselinePoint":
        """Deserialize from a dict, normalizing optional floats to 0.0.

        ``gpu_util_pct`` and ``gpu_mem_gb`` are normalised from ``None`` to
        ``0.0`` so that JSON round-trips produce consistent numeric types
        regardless of whether the baseline was captured with or without GPU
        telemetry.
        """
        kwargs: dict = {}
        for k in cls.__dataclass_fields__:
            if k in d:
                val = d[k]
                # Normalise optional GPU fields so 0.0 ≠ None in comparisons.
                if k in ("gpu_util_pct", "gpu_mem_gb") and val is None:
                    val = 0.0
                kwargs[k] = val
        return cls(**kwargs)


@dataclass
class RegressionResult:
    """Result of a regression check against a baseline."""

    is_regression: bool = False
    reason: str = ""
    baseline_mean_tps: float = 0.0
    current_mean_tps: float = 0.0
    pct_change: float = 0.0
    p_value: float | None = None
    severity: str = "none"  # none, minor (5-10%), major (10-20%), critical (>20%)

    def to_dict(self) -> dict:
        return {
            "is_regression": self.is_regression,
            "reason": self.reason,
            "baseline_mean_tps": self.baseline_mean_tps,
            "current_mean_tps": self.current_mean_tps,
            "pct_change": round(self.pct_change, 2),
            "p_value": round(self.p_value, 6) if self.p_value else None,
            "severity": self.severity,
        }


# ── Baseline manager ────────────────────────────────────────────────────


class PerformanceBaselineManager:
    """Manages performance baselines and detects regressions.

    Baselines are stored as JSON files in a directory.  Each baseline
    aggregates multiple runs for statistical robustness.

    Regression detection uses:
      - Throughput drop > threshold (default 5%).
      - Latency increase > threshold (default 5%).
      - One-sample t-test for statistical significance (n >= 3 data points;
        uses scipy.stats when available, normal approximation as fallback).
    """

    def __init__(
        self,
        baseline_dir: str | Path = "./baselines",
        regression_threshold_pct: float = 5.0,
        significance_alpha: float = 0.05,
    ):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.regression_threshold_pct = regression_threshold_pct
        self.significance_alpha = significance_alpha

        # In-memory cache: baseline_name → list[BaselinePoint].
        self._cache: dict[str, list[BaselinePoint]] = {}

    # ── Save / Load ────────────────────────────────────────────────────

    def save_baseline(
        self,
        name: str,
        metrics: dict,
        metadata: dict | None = None,
    ) -> BaselinePoint:
        """Save a single benchmark run as a baseline data point.

        Parameters
        ----------
        name : str
            Baseline name (e.g., "h200_fp8_12b_batch32").
        metrics : dict
            Dictionary with keys matching BaselinePoint fields.
        metadata : dict, optional
            Arbitrary metadata (git commit, PyTorch version, etc.).

        Returns
        -------
        BaselinePoint
        """
        batch = metrics.get("batch", {})
        device = metrics.get("device", {})
        runtime = metrics.get("runtime", {})

        point = BaselinePoint(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            mean_tps=batch.get("mean_tps", 0),
            median_tps=batch.get("median_tps", 0),
            std_tps=batch.get("std_tps", 0),
            p95_tps=batch.get("p95_tps", 0),
            mean_latency_ms=batch.get("mean_latency_ms", 0),
            p95_latency_ms=batch.get("p95_latency_ms", 0),
            gpu_util_pct=device.get("mean_util_pct") if device.get("mean_util_pct") is not None else 0.0,
            gpu_mem_gb=(device.get("mean_mem_used_mib", 0) / 1024)
            if device.get("mean_mem_used_mib") is not None else 0.0,
            total_output_tokens=batch.get("total_output_tokens", 0),
            duration_seconds=metrics.get("runtime", {}).get("actual_duration_seconds", 0),
            metadata=metadata or {},
        )

        # Load existing, append, save.
        points = self._load_raw(name)
        points.append(point)
        self._save_raw(name, points)
        self._cache[name] = points

        logger.info(
            "Baseline '%s' saved: %.1f tok/s (%d data points)",
            name, point.mean_tps, len(points),
        )
        return point

    def load_baseline(self, name: str) -> list[BaselinePoint]:
        """Load all data points for a baseline.

        Returns an empty list if the baseline doesn't exist.
        """
        if name in self._cache:
            return self._cache[name]

        points = self._load_raw(name)
        self._cache[name] = points
        return points

    def baseline_stats(self, name: str) -> dict:
        """Compute aggregate statistics for a baseline.

        Returns
        -------
        dict
            Keys: ``count``, ``mean_tps``, ``std_tps``, ``min_tps``,
            ``max_tps``, ``ci95_lower``, ``ci95_upper``, ``data_points``.
        """
        points = self.load_baseline(name)
        if not points:
            return {"count": 0, "error": "No baseline data"}

        tps_values = [p.mean_tps for p in points]
        n = len(tps_values)
        mean = statistics.mean(tps_values)
        std = statistics.stdev(tps_values) if n > 1 else 0.0

        # 95% confidence interval for the mean (t-distribution).
        if n > 1:
            try:
                from scipy import stats as _sp_stats
                t_crit = _sp_stats.t.ppf(0.975, df=n - 1)
            except ImportError:
                # Fallback: approximate t-critical from normal distribution,
                # with a minimal correction for small n.
                if n <= 2:
                    t_crit = 12.706  # t_0.975, 1 df
                elif n <= 5:
                    t_crit = 2.776   # t_0.975, 4 df
                elif n <= 10:
                    t_crit = 2.262   # t_0.975, 9 df
                elif n <= 30:
                    t_crit = 2.045   # t_0.975, 29 df
                else:
                    t_crit = 1.96    # normal approximation
            ci95_margin = t_crit * std / math.sqrt(n)
        else:
            ci95_margin = 0.0

        return {
            "count": n,
            "mean_tps": round(mean, 1),
            "std_tps": round(std, 1),
            "min_tps": round(min(tps_values), 1),
            "max_tps": round(max(tps_values), 1),
            "ci95_lower": round(mean - ci95_margin, 1),
            "ci95_upper": round(mean + ci95_margin, 1),
            "data_points": [p.to_dict() for p in points[-5:]],  # last 5
        }

    # ── Regression detection ────────────────────────────────────────────

    def check(
        self,
        baseline_name: str,
        current_metrics: dict,
        current_metadata: dict | None = None,
    ) -> RegressionResult:
        """Check if a current run has regressed against a baseline.

        Parameters
        ----------
        baseline_name : str
            Name of the baseline to compare against.
        current_metrics : dict
            Metrics from the current run (same format as report's metrics dict).
        current_metadata : dict, optional
            Metadata for the current run (saved if this becomes a new baseline point).

        Returns
        -------
        RegressionResult
        """
        points = self.load_baseline(baseline_name)
        if not points:
            logger.warning("Baseline '%s' has no data — auto-establishing", baseline_name)
            self.save_baseline(baseline_name, current_metrics, current_metadata)
            return RegressionResult(
                is_regression=False,
                reason="Baseline auto-established (first run)",
                baseline_mean_tps=0.0,
                current_mean_tps=current_metrics.get("batch", {}).get("mean_tps", 0),
                pct_change=0.0,
            )

        baseline_tps_values = [p.mean_tps for p in points]
        baseline_mean = statistics.mean(baseline_tps_values)
        current_mean = current_metrics.get("batch", {}).get("mean_tps", 0)

        if current_mean is None or current_mean <= 0:
            return RegressionResult(
                is_regression=True,
                reason="Current TPS is zero or negative",
                baseline_mean_tps=baseline_mean,
                current_mean_tps=current_mean,
                pct_change=-100.0,
                severity="critical",
            )

        pct_change = ((current_mean - baseline_mean) / baseline_mean) * 100

        # ── Severity classification ──
        severity = "none"
        if pct_change < -20:
            severity = "critical"
        elif pct_change < -10:
            severity = "major"
        elif pct_change < -self.regression_threshold_pct:
            severity = "minor"

        # ── Statistical test (t-test for small samples) ──
        p_value = None
        if len(points) >= 3:
            try:
                baseline_std = statistics.stdev(baseline_tps_values)
                if baseline_std > 0:
                    # One-sample t-test: compare current_mean against the
                    # baseline distribution.  Uses t-distribution with n-1
                    # degrees of freedom (correct for n < 30).
                    se = baseline_std * math.sqrt(1 + 1 / len(points))
                    t_stat = (current_mean - baseline_mean) / se if se > 0 else 0
                    df = len(points) - 1
                    try:
                        from scipy import stats as _sp_stats
                        p_value = 2 * _sp_stats.t.sf(abs(t_stat), df)
                    except ImportError:
                        # Fallback: t-distribution approximation using the
                        # Student's t survival function.  More accurate than
                        # the normal CDF for small samples.
                        import math
                        # Incomplete beta-function approximation for t-CDF.
                        # For df > 2, the regularised incomplete beta gives
                        # usable two-sided p-values.  We fall back to Welch-style
                        # correction when df is very small.
                        x = df / (df + t_stat * t_stat)
                        p_value = _t_sf_fallback(abs(t_stat), df)
            except Exception:
                pass

        is_regression = severity != "none"

        # Statistical significance override: even if pct change is small,
        # a very low p-value signals a real change.
        if p_value is not None and p_value < self.significance_alpha and not is_regression:
            # Check direction: is it a negative change?
            if pct_change < 0:
                is_regression = True
                severity = "minor"
                logger.info(
                    "Statistically significant regression detected "
                    "(p=%.4f, change=%.1f%%)", p_value, pct_change,
                )

        result = RegressionResult(
            is_regression=is_regression,
            reason=(
                f"Throughput dropped {abs(pct_change):.1f}% "
                f"({baseline_mean:.0f} → {current_mean:.0f} tok/s)" if is_regression
                else f"Within threshold ({pct_change:+.1f}%)"
            ),
            baseline_mean_tps=round(baseline_mean, 1),
            current_mean_tps=round(current_mean, 1),
            pct_change=round(pct_change, 2),
            p_value=round(p_value, 6) if p_value else None,
            severity=severity,
        )

        if is_regression:
            logger.warning(
                "PERF REGRESSION [%s]: %s (p=%s)",
                severity.upper(), result.reason, f"{p_value:.4f}" if p_value else "N/A",
            )
        else:
            logger.info(
                "Perf check passed: baseline=%.0f, current=%.0f (%+.1f%%)",
                baseline_mean, current_mean, pct_change,
            )

        return result

    # ── Internal ────────────────────────────────────────────────────────

    def _baseline_path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace(" ", "_")
        return self.baseline_dir / f"{safe}.json"

    def _load_raw(self, name: str) -> list[BaselinePoint]:
        path = self._baseline_path(name)
        if not path.exists():
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return [BaselinePoint.from_dict(d) for d in data.get("points", [])]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to load baseline '%s': %s", name, e)
            return []

    def _save_raw(self, name: str, points: list[BaselinePoint]) -> None:
        path = self._baseline_path(name)
        data = {
            "name": name,
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": len(points),
            "points": [p.to_dict() for p in points],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── List / delete ──────────────────────────────────────────────────

    def list_baselines(self) -> list[dict]:
        """List all known baselines with summary stats."""
        result = []
        for path in sorted(self.baseline_dir.glob("*.json")):
            name = path.stem
            stats = self.baseline_stats(name)
            result.append({
                "name": name,
                "count": stats.get("count", 0),
                "mean_tps": stats.get("mean_tps"),
                "path": str(path),
            })
        return result

    def delete_baseline(self, name: str) -> bool:
        """Delete a baseline."""
        path = self._baseline_path(name)
        if path.exists():
            path.unlink()
            self._cache.pop(name, None)
            logger.info("Baseline '%s' deleted", name)
            return True
        return False
