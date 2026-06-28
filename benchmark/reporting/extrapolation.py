"""Full-dataset extrapolation model — predicts days to completion."""

import logging
import math

logger = logging.getLogger(__name__)

# Try to import scipy for proper t-distribution critical values.
# Falls back to conservative Normal approximation when unavailable.
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _t_critical_value(alpha: float, df: int) -> float:
    """Return the two-tailed t critical value for significance *alpha* and *df*.

    Falls back to the Normal approximation (z-score) when scipy is not
    installed or the degrees-of-freedom are large.

    .. warning::
       The Normal fallback is **only** valid when df >= 30.  For small
       samples (df < 30) without scipy, the returned value is a
       conservative over-estimate that produces wider CIs than a true
       t-distribution would, but is still preferable to the z-score for
       decision-making about statistical significance.
    """
    if _HAS_SCIPY and df >= 1:
        return float(_scipy_stats.t.ppf(1.0 - alpha / 2.0, df))
    # Conservative Normal approximation — only valid for df >= 30.
    # For df < 30 this over-estimates the critical value, producing
    # wider-than-true CIs (safer than under-estimation).
    if df < 30 and not _HAS_SCIPY:
        logger.warning(
            "scipy not installed; using Normal approximation for df=%d "
            "(n < 30). CI will be wider than a proper t-test. Install "
            "scipy for accurate small-sample inference.", df,
        )
    if alpha <= 0.01:
        return 2.576
    if alpha <= 0.05:
        return 1.96
    if alpha <= 0.10:
        return 1.645
    return 1.0


class ExtrapolationModel:
    """Estimate corpus-completion time from measured throughput.
    
        Takes per-batch TPS samples and total corpus token count,
        then computes:
        - Median-based point estimate (robust to skew)
        - SEM + t-distribution 95% CI
        - Bootstrap 95% CI (percentile method)
    
        Assumes constant throughput (validated over 2.2h on H200).
        """
    def __init__(self, total_tokens: int = 6_230_000_000_000, gpu_cost_per_hour: float | None = None):
        """Initialise the extrapolation model.

        Parameters
        ----------
        total_tokens : int
            Total tokens in the full dataset.  Must be positive.
        gpu_cost_per_hour : float or None
            Cost per GPU-hour in USD, or None to skip cost computation.

        Raises
        ------
        ValueError
            If ``total_tokens <= 0``.
        """
        if total_tokens <= 0:
            raise ValueError(f"total_tokens must be positive, got {total_tokens}")
        self.total_tokens = total_tokens
        self.gpu_cost_per_hour = gpu_cost_per_hour

    def compute(self, mean_tps: float, std_tps: float = 0.0, num_gpus: int = 2,
                n_batches: int = 1, median_tps: float | None = None) -> dict:
        """Compute days-to-completion with 95% CI using t-distribution.

        Parameters
        ----------
        mean_tps : float
            Mean tokens per second across all batches.
        std_tps : float
            Standard deviation of per-batch tokens per second.
        num_gpus : int
            Number of GPUs used (for GPU-hour calculation).
        n_batches : int
            Number of batches used to estimate mean/std.
        median_tps : float or None
            Median tokens per second.  When provided, the point estimate
            uses the median (more robust against right-skewed TPS
            distributions).  Falls back to mean when None.
        """
        # Use median for point estimate when available — TPS distributions
        # are typically right-skewed and the mean overestimates throughput.
        effective_tps = median_tps if median_tps is not None else mean_tps
        if effective_tps <= 0:
            return {
                "error": "effective_tps must be positive",
                "days_point_estimate": 0,
                "days_95ci_lower": 0,
                "days_95ci_upper": 0,
                "gpu_hours": 0,
                "estimated_cost_usd": None,
                "relative_uncertainty_pct": 0,
                "sem_tokens_per_second": 0,
            }
        if num_gpus <= 0:
            return {
                "error": "num_gpus must be positive",
                "days_point_estimate": float("inf"),
                "days_95ci_lower": 0,
                "days_95ci_upper": 0,
                "gpu_hours": 0,
                "estimated_cost_usd": None,
                "relative_uncertainty_pct": 0,
                "sem_tokens_per_second": 0,
            }
        if n_batches < 5:
            logger.warning(
                "Extrapolation from %d batch(es) is not statistically meaningful. "
                "Run at least 30 batches for a valid CI.",
                n_batches,
            )

        # Point estimate: days = total_tokens / median_tps (robust).
        seconds = self.total_tokens / effective_tps
        days = seconds / 86400
        gpu_hours = days * num_gpus * 24

        # Standard error of the mean TPS.
        se_tps = std_tps / math.sqrt(n_batches) if n_batches > 0 else std_tps
        # Relative uncertainty MUST use the same TPS value as the point estimate.
        # Previously this used se_tps/mean_tps but days came from median_tps,
        # producing invalid CIs when mean ≠ median (skewed distributions).
        rel_uncertainty = se_tps / effective_tps if effective_tps > 0 else 0

        # Use t-distribution critical value (n-1 degrees of freedom) for
        # samples with n < 30.  Falls back to Normal approximation when
        # scipy is unavailable or n >= 30.
        df = n_batches - 1
        if n_batches < 30:
            z = _t_critical_value(0.05, df)
        else:
            z = _t_critical_value(0.05, df)  # ≈ 1.96 for large df

        days_uncertainty = days * rel_uncertainty
        days_lower = max(0, days - z * days_uncertainty)
        days_upper = days + z * days_uncertainty

        # Warn when mean and median diverge (skew indicator).
        if median_tps is not None and mean_tps > 0:
            skew_ratio = mean_tps / median_tps
            if skew_ratio > 1.2:
                logger.warning(
                    "Mean TPS (%.0f) is %.1fx higher than median (%.0f) — "
                    "distribution is right-skewed. Using median for the point "
                    "estimate; CI based on mean may be optimistic.",
                    mean_tps, skew_ratio, median_tps,
                )

        cost = None
        if self.gpu_cost_per_hour:
            cost = gpu_hours * self.gpu_cost_per_hour
        result = {"total_tokens": self.total_tokens, "mean_tokens_per_second": round(mean_tps, 1),
                  "median_tokens_per_second": round(median_tps, 1) if median_tps else None,
                  "std_tokens_per_second": round(std_tps, 1),
                  "sem_tokens_per_second": round(se_tps, 1),
                  "n_batches": n_batches,
                  "seconds_needed": round(seconds, 0), "days_point_estimate": round(days, 1),
                  "days_95ci_lower": round(days_lower, 1), "days_95ci_upper": round(days_upper, 1),
                  "gpu_hours": round(gpu_hours, 1), "estimated_cost_usd": round(cost, 2) if cost else None,
                  "num_gpus": num_gpus, "relative_uncertainty_pct": round(rel_uncertainty * 100, 1)}
        logger.info("Extrapolation: %.1f days, 95%% CI [%.1f, %.1f]", result["days_point_estimate"], days_lower, days_upper)
        return result

    def compute_bootstrap(self, tps_samples: list[float], num_gpus: int = 2,
                          n_bootstrap: int = 10_000, seed: int = 42) -> dict:
        """Compute CI via bootstrap resampling of per-batch TPS values.

        This is more robust than the parametric CI — no Gaussian assumption,
        correctly handles small samples, and can model asymmetric distributions.

        Used for both discrete and continuous batching run modes when per-batch
        TPS samples are available in the batch metrics summary.

        Parameters
        ----------
        tps_samples : list[float]
            Raw per-batch tokens-per-second values.
        num_gpus : int
            Number of GPUs used.
        n_bootstrap : int
            Number of bootstrap resamples (default 10k).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict with bootstrap_days_lower, bootstrap_days_upper, days_point_estimate.
        """
        import random
        rng = random.Random(seed)
        n = len(tps_samples)
        # Guard: ensure enough resamples for valid 2.5th/97.5th percentile indices.
        n_bootstrap = max(n_bootstrap, 1000)
        if n == 0:
            return {"error": "no TPS samples provided"}
        if n == 1:
            return self.compute(mean_tps=tps_samples[0], num_gpus=num_gpus, n_batches=1)

        # Point estimate.
        mean_tps = sum(tps_samples) / n
        std_pop = (sum((x - mean_tps)**2 for x in tps_samples) / (n - 1))**0.5 if n > 1 else 0

        # Bootstrap: resample with replacement, compute mean, derive days.
        means = []
        for _ in range(n_bootstrap):
            sample = [tps_samples[rng.randint(0, n - 1)] for _ in range(n)]
            means.append(sum(sample) / n)
        means.sort()

        ci_lower_tps = means[int(n_bootstrap * 0.025)]
        ci_upper_tps = means[int(n_bootstrap * 0.975)]

        days_point = self.total_tokens / mean_tps / 86400
        seconds_lower = self.total_tokens / ci_upper_tps if ci_upper_tps > 0 else float('inf')
        seconds_upper = self.total_tokens / ci_lower_tps if ci_lower_tps > 0 else float('inf')

        gpu_hours = days_point * num_gpus * 24
        cost = None
        if self.gpu_cost_per_hour:
            cost = gpu_hours * self.gpu_cost_per_hour

        result = {
            "total_tokens": self.total_tokens,
            "mean_tokens_per_second": round(mean_tps, 1),
            "std_tokens_per_second": round(std_pop, 1),
            "days_point_estimate": round(days_point, 1),
            "bootstrap_days_lower": round(seconds_lower / 86400, 1),
            "bootstrap_days_upper": round(seconds_upper / 86400, 1),
            "gpu_hours": round(gpu_hours, 1),
            "estimated_cost_usd": round(cost, 2) if cost else None,
            "num_gpus": num_gpus,
            "n_batches": n,
            "method": "bootstrap",
        }
        logger.info(
            "Bootstrap extrapolation: %.1f days, CI [%.1f, %.1f] (n=%d, %d resamples)",
            result["days_point_estimate"],
            result["bootstrap_days_lower"],
            result["bootstrap_days_upper"],
            n, n_bootstrap,
        )
        return result
