"""Conservative degradation-aware extrapolation for H200 TR benchmark.

The existing ExtrapolationModel assumes constant throughput, but long runs
experience thermal throttling and memory fragmentation.  This module adds:

  - DegradationModel: detects throughput degradation over time from per-batch
    TPS samples, using linear regression.  Provides a conservative TPS estimate
    for long-run projection.

  - ExtrapolationV2: a drop-in wrapper that delegates to ExtrapolationModel for
    core CI/statistics math, but substitutes the DegradationModel point estimate
    when degradation is detected.

  - degradation_warning(): a human-readable text warning (or None) for inclusion
    in benchmark reports.

All public classes and functions are callable in the benchmark reporting
pipeline without additional configuration.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Optional

import numpy as np

# Keep the scipy import conditional — the module must work without it.
# linear regression is the core operation, and linregress gives us R² plus
# a p-value out of the box; when scipy isn't available, fall back to
# numpy.polyfit with manual R² computation.
_HAS_SCIPY: bool
try:
    from scipy import stats as _scipy_stats  # noqa: F811 — used for linregress

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger(__name__)

# ── Sentinel / constants ────────────────────────────────────────────────────

_MIN_SAMPLES_FOR_REGRESSION = 10
# Require at least 10 samples before claiming a trend exists — fewer than
# this and the regression is noise.
_DEGRADATION_R2_THRESHOLD = 0.1
# R² must be at least this high for a negative slope to be treated as a
# real degradation signal.  Low bar intentionally: we want to be sensitive
# to *any* explainable downward trend.
_SECONDS_PER_HOUR = 3600.0
_CONSERVATIVE_HORIZON_HOURS = 72.0
# When degradation is detected, project the linear fit out to 72 hours
# to get the conservative TPS lower bound.

# ── Internal helpers ─────────────────────────────────────────────────────────


def _fit_linear_regression(
    t: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, float | None]:
    """Fit y = a*t + b via linear regression.

    Returns
    -------
    (slope, intercept, r_squared, p_value)
      p_value is ``None`` when scipy is unavailable (numpy fallback).
    """
    if _HAS_SCIPY:
        result = _scipy_stats.linregress(t, y)
        return result.slope, result.intercept, result.rvalue**2, result.pvalue

    # NumPy polyfit fallback — no p-value.
    slope, intercept = np.polyfit(t, y, 1)  # type: ignore[call-overload]
    # Manual R²
    y_pred = slope * t + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot > 0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = 0.0
    return float(slope), float(intercept), r_squared, None


def _percentile(data: list[float], pct: float) -> float:
    """Compute the pct-th percentile of *data* using linear interpolation.

    Returns a float even when *data* contains only integers.
    """
    if not data:
        raise ValueError("cannot compute percentile of empty list")
    arr = np.asarray(data, dtype=np.float64)
    return float(np.percentile(arr, pct))  # type: ignore[arg-type]


# ── Public API ───────────────────────────────────────────────────────────────


class DegradationModel:
    """Detect throughput degradation over time from per-batch TPS samples.

    Uses linear regression (TPS = a * t + b) over the collected samples.
    A statistically-significant negative slope indicates throughput is
    declining — the most common cause on H200 hardware is thermal throttling
    after sustained load.

    Parameters
    ----------
    tps_samples : list[float]
        Per-batch tokens-per-second values, in chronological order.
    timestamps : list[float]
        Seconds since the start of the run for each sample.  Must have the
        same length as *tps_samples*.
    """

    def __init__(self, tps_samples: list[float], timestamps: list[float]) -> None:
        """Initialize degradation model from per-batch TPS time-series data.

        Parameters
        ----------
        tps_samples : list[float]
            Per-batch tokens-per-second values, in chronological order.
            Each element is the TPS measured for a single batch.
        timestamps : list[float]
            Seconds elapsed since the start of the run for each TPS sample.
            Must have the same length as *tps_samples* and be monotonically
            non-decreasing.

        Raises
        ------
        ValueError
            If *tps_samples* and *timestamps* have different lengths, or if
            fewer than 2 samples are provided.
        """
        if len(tps_samples) != len(timestamps):
            raise ValueError(
                f"tps_samples and timestamps must have the same length; "
                f"got {len(tps_samples)} vs {len(timestamps)}"
            )
        if len(tps_samples) < 2:
            raise ValueError(
                f"Need at least 2 TPS samples for degradation analysis; "
                f"got {len(tps_samples)}"
            )

        self._tps = list(tps_samples)
        self._timestamps = list(timestamps)
        self._n = len(tps_samples)

        # Fit linear regression lazily — only when first queried.
        self._slope: float | None = None
        self._intercept: float | None = None
        self._r_squared: float | None = None
        self._p_value: float | None = None

    # ── Lazy regression fitting ──────────────────────────────────────────

    def _fit(self) -> None:
        """Fit the linear regression once, caching all outputs."""
        if self._slope is not None:
            return
        t_arr = np.asarray(self._timestamps, dtype=np.float64)
        y_arr = np.asarray(self._tps, dtype=np.float64)
        self._slope, self._intercept, self._r_squared, self._p_value = (
            _fit_linear_regression(t_arr, y_arr)
        )

    @property
    def slope(self) -> float:
        """Regression slope (TPS change per second).  Negative = degradation."""
        self._fit()
        assert self._slope is not None
        return self._slope

    @property
    def r_squared(self) -> float:
        """R² of the TPS-vs-time linear fit (0.0–1.0)."""
        self._fit()
        assert self._r_squared is not None
        return self._r_squared

    @property
    def p_value(self) -> float | None:
        """p-value of the slope (``None`` when scipy is unavailable)."""
        self._fit()
        return self._p_value

    # ── Public query methods ─────────────────────────────────────────────

    def has_degradation(self) -> bool:
        """Return True if throughput shows a statistically significant
        downward trend.

        Criteria: (1) at least *_MIN_SAMPLES_FOR_REGRESSION* samples,
        (2) negative slope, and (3) R² >= *_DEGRADATION_R2_THRESHOLD*.
        """
        if self._n < _MIN_SAMPLES_FOR_REGRESSION:
            return False
        self._fit()
        assert self._slope is not None and self._r_squared is not None
        return self._slope < 0.0 and self._r_squared >= _DEGRADATION_R2_THRESHOLD

    def degradation_rate_pct_per_hour(self) -> float:
        """Estimated throughput loss as % per hour of run time.

        Returns a **negative** number when throughput is declining (e.g.,
        -5.2 means ~5.2% TPS loss per hour).  Returns 0.0 when there are
        too few samples to estimate.
        """
        if self._n < _MIN_SAMPLES_FOR_REGRESSION:
            return 0.0
        self._fit()
        assert self._slope is not None
        if self._intercept is None or self._intercept <= 0.0:
            return 0.0
        hourly_loss = self._slope * _SECONDS_PER_HOUR
        # Express as percentage of initial TPS (at t=0, the intercept).
        return float((hourly_loss / self._intercept) * 100.0)

    def conservative_tps_estimate(self) -> float:
        """Conservative TPS estimate for long-run projection.

        - If **degradation is detected**: use the linear-fit TPS at
          t = 72 hours as the sustained rate (conservative lower bound).

        - If **no degradation detected**: fall back to the 25th percentile
          of TPS samples (conservative choice even without a trend — protects
          against right-skewed TPS distributions).

        - If there are **too few samples for any reasonable estimate**:
          return the median (least-worst fallback).
        """
        was_degraded = self.has_degradation()
        self._fit()
        assert self._slope is not None and self._intercept is not None

        if was_degraded:
            # Project out to 72 hours
            t_projection = _CONSERVATIVE_HORIZON_HOURS * _SECONDS_PER_HOUR
            projected = self._slope * t_projection + self._intercept
            if projected <= 0.0:
                # Degeneration would predict zero or negative TPS — use
                # the 5th percentile instead, which is the most pessimistic
                # we can be without being nonsensical.
                projected = _percentile(self._tps, 5.0)
            logger.info(
                "Degradation detected (slope=%.2f TPS/h, R²=%.3f). "
                "Conservative TPS at %dh horizon: %.1f (start was %.1f).",
                self.degradation_rate_pct_per_hour(),
                self.r_squared,
                _CONSERVATIVE_HORIZON_HOURS,
                projected,
                self._intercept,
            )
            return projected

        # No degradation — use 25th percentile.
        if self._n >= 4:
            return _percentile(self._tps, 25.0)

        # Tiny sample — use median.
        sorted_tps = sorted(self._tps)
        mid = len(sorted_tps) // 2
        if len(sorted_tps) % 2 == 0:
            return float((sorted_tps[mid - 1] + sorted_tps[mid]) / 2.0)
        return float(sorted_tps[mid])

    def thermal_throttling_risk(self) -> str:
        """Qualitative risk label based on the degradation slope.

        Returns one of ``'high'``, ``'medium'``, ``'low'``, or ``'none'``.

        The thresholds are calibrated for H200 GPUs running sustained
        transformer inference — a 3%/h decline over 2 hours is a strong
        thermal signal on this hardware.

        When ``has_degradation()`` is ``False``, always returns ``'none'``
        regardless of the raw rate percentage — small-sample noise should
        not produce spurious medium/high labels.
        """
        if not self.has_degradation():
            return "none"
        rate = self.degradation_rate_pct_per_hour()
        # Use absolute value because rate is negative for degradation.
        abs_rate = abs(rate)
        if abs_rate >= 3.0:
            return "high"
        if abs_rate >= 1.5:
            return "medium"
        if abs_rate >= 0.5:
            return "low"
        return "none"

    # ── For external introspection ──────────────────────────────────────

    def __repr__(self) -> str:
        self._fit()
        return (
            f"DegradationModel(n={self._n}, slope={self.slope:.6f}, "
            f"R²={self.r_squared:.4f}, "
            f"rate={self.degradation_rate_pct_per_hour():.2f}%/h, "
            f"throttle_risk={self.thermal_throttling_risk()!r})"
        )


class ExtrapolationV2:
    """Degradation-aware wrapper around `ExtrapolationModel`.

    Delegate semantics
    ------------------
    - If **no degradation is detected** (or too few samples), delegates
      entirely to `ExtrapolationModel.compute()` — the output is identical
      to the existing extrapolation.

    - If **degradation is detected**, uses `DegradationModel.conservative_tps_estimate()`
      as the point-estimate TPS, passes it as `median_tps` to
      `ExtrapolationModel.compute()`, and injects extra degradation metadata
      into the result dict.

    Parameters
    ----------
    base_model : ExtrapolationModel
        An already-constructed instance from the existing extrapolation module.
    """

    def __init__(self, base_model) -> None:
        """Initialize the degradation-aware extrapolation wrapper.

        Parameters
        ----------
        base_model : ExtrapolationModel
            An already-constructed instance from
            ``benchmark.reporting.extrapolation``.  The wrapper delegates all
            core statistics computation to this model, only adjusting the TPS
            point estimate when degradation is detected.

        Raises
        ------
        TypeError
            If *base_model* is not an instance of ``ExtrapolationModel``.
        """
        # Lazy import to avoid circular dependency at module level.
        from benchmark.reporting.extrapolation import ExtrapolationModel as _Base

        if not isinstance(base_model, _Base):
            raise TypeError(
                f"base_model must be an ExtrapolationModel; got {type(base_model)}"
            )
        self._base = base_model
        self._degradation_model: DegradationModel | None = None
        self._tps_samples: list[float] = []
        self._timestamps: list[float] = []

    # ── Data ingestion ───────────────────────────────────────────────────

    def ingest_samples(
        self, tps_samples: list[float], timestamps: list[float]
    ) -> None:
        """Feed per-batch TPS time-series data for degradation analysis.

        Call this after the translation loop completes, once all per-batch
        TPS values are available.  Must be called before `compute()`.
        """
        if len(tps_samples) <= 1:
            logger.debug(
                "ExtrapolationV2.ingest_samples: only %d sample(s) — "
                "skipping degradation analysis.",
                len(tps_samples),
            )
            return
        self._tps_samples = list(tps_samples)
        self._timestamps = list(timestamps)
        self._degradation_model = DegradationModel(
            self._tps_samples, self._timestamps
        )

    # ── Compute ──────────────────────────────────────────────────────────

    def compute(
        self,
        mean_tps: float,
        std_tps: float = 0.0,
        num_gpus: int = 2,
        n_batches: int = 1,
        median_tps: float | None = None,
    ) -> dict:
        """Compute days-to-completion, degradation-aware.

        Signature is a strict superset of ``ExtrapolationModel.compute()`` —
        callers that pass the same kwargs will receive a compatible result dict.

        Returns
        -------
        dict
            Same keys as ``ExtrapolationModel.compute()`` plus optional
            degradation metadata:
            - ``degradation_pct_per_hour`` (float | None)
            - ``thermal_throttling_risk`` (str | None)
            - ``degradation_r_squared`` (float | None)
            - ``degradation_n_samples`` (int)
            - ``estimation_method`` = ``"degradation_aware"`` or ``"standard"``
        """
        # Decide whether to use the degradation-adjusted point estimate.
        degraded = False
        conservative_tps: float | None = None

        if self._degradation_model is not None and self._degradation_model.has_degradation():
            degraded = True
            conservative_tps = self._degradation_model.conservative_tps_estimate()
            logger.info(
                "ExtrapolationV2: degradation detected; using conservative "
                "TPS estimate %.1f for long-run projection.",
                conservative_tps,
            )

        # Choose the effective median for the base model call.
        if degraded and conservative_tps is not None:
            effective_median = conservative_tps
            # Also adjust mean downward proportionally so the CI isn't inflated.
            if median_tps is not None and median_tps > 0:
                scale = conservative_tps / median_tps
                effective_mean = mean_tps * scale
            elif mean_tps > 0:
                scale = conservative_tps / mean_tps
                effective_mean = conservative_tps
            else:
                effective_mean = conservative_tps
        else:
            effective_median = median_tps
            effective_mean = mean_tps

        # Delegate to the base model for core statistics.
        result = self._base.compute(
            mean_tps=effective_mean,
            std_tps=std_tps,
            num_gpus=num_gpus,
            n_batches=n_batches,
            median_tps=effective_median,
        )

        # ---- Inject degradation metadata ----
        dm = self._degradation_model
        deg_rate = dm.degradation_rate_pct_per_hour() if dm else None
        throttle = dm.thermal_throttling_risk() if dm else None
        deg_r2 = dm.r_squared if dm else None
        deg_n = dm._n if dm else len(self._tps_samples)

        result.update({
            "estimation_method": "degradation_aware" if degraded else "standard",
            "degradation_pct_per_hour": round(deg_rate, 2) if deg_rate else None,
            "thermal_throttling_risk": throttle,
            "degradation_r_squared": round(deg_r2, 4) if deg_r2 is not None else None,
            "degradation_n_samples": deg_n,
        })

        if degraded:
            logger.warning(
                "ExtrapolationV2: throughput degraded %.2f%%/hour (R²=%.4f). "
                "Days estimate adjusted downward.  Thermal risk: %s.",
                deg_rate,
                deg_r2,
                throttle,
            )

        return result

    @property
    def degradation_model(self) -> DegradationModel | None:
        """The fitted degradation model (or None if not yet ingested)."""
        return self._degradation_model

    @property
    def base_model(self):
        """The underlying `ExtrapolationModel` instance."""
        return self._base


def degradation_warning(
    tps_samples: list[float], timestamps: list[float]
) -> str | None:
    """Return a human-readable degradation warning, or *None*.

    Intended for embedding in benchmark reports (Markdown and JSON).

    Example output
    --------------
    "Throughput degraded 5.2%/hour over this run.  The days-to-completion
    estimate has been adjusted downward to account for this."

    Returns *None* when degradation is not statistically significant or
    there are too few samples to evaluate.
    """
    if len(tps_samples) < _MIN_SAMPLES_FOR_REGRESSION:
        return None
    try:
        model = DegradationModel(tps_samples, timestamps)
    except ValueError:
        return None

    if not model.has_degradation():
        return None

    rate = model.degradation_rate_pct_per_hour()
    risk = model.thermal_throttling_risk()

    # Build the message.
    msg = (
        f"Throughput degraded {rate:.1f}%/hour over this run. "
        f"The days-to-completion estimate has been adjusted downward "
        f"to account for this."
    )

    if risk == "high":
        msg += (
            f" Thermal throttling risk is HIGH — consider improving cooling, "
            f"reducing batch size, or scheduling breaks between long runs."
        )
    elif risk == "medium":
        msg += (
            f" Thermal throttling risk is MEDIUM — monitor GPU temperatures "
            f"and consider shorter measurement intervals."
        )

    return msg
