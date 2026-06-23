"""Full-dataset extrapolation model — predicts days to completion."""

import logging
import math

logger = logging.getLogger(__name__)


class ExtrapolationModel:
    def __init__(self, total_tokens: int = 6_230_000_000_000, gpu_cost_per_hour: float | None = None):
        self.total_tokens = total_tokens
        self.gpu_cost_per_hour = gpu_cost_per_hour

    def compute(self, mean_tps: float, std_tps: float = 0.0, num_gpus: int = 2,
                n_batches: int = 1) -> dict:
        """Compute days-to-completion with 95% CI using the standard error of the mean.

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
            The CI width shrinks by 1/sqrt(n_batches) — larger samples
            yield tighter confidence intervals.
        """
        if mean_tps <= 0:
            return {
                "error": "mean_tps must be positive",
                "days_point_estimate": 0,
                "days_95ci_lower": 0,
                "days_95ci_upper": 0,
                "gpu_hours": 0,
                "estimated_cost_usd": None,
                "relative_uncertainty_pct": 0,
            }
        if n_batches < 5:
            logger.warning(
                "Extrapolation from %d batch(es) is not statistically meaningful. "
                "Run at least 5 batches for a useful days estimate.",
                n_batches,
            )
        seconds = self.total_tokens / mean_tps
        days = seconds / 86400
        gpu_hours = days * num_gpus * 24

        # Standard error of the mean (not raw std).
        se_tps = std_tps / math.sqrt(n_batches) if n_batches > 0 else std_tps
        rel_uncertainty = se_tps / mean_tps if mean_tps > 0 else 0
        days_uncertainty = days * rel_uncertainty
        days_lower = max(0, days - 1.96 * days_uncertainty)
        days_upper = days + 1.96 * days_uncertainty

        cost = None
        if self.gpu_cost_per_hour:
            cost = gpu_hours * self.gpu_cost_per_hour
        result = {"total_tokens": self.total_tokens, "mean_tokens_per_second": round(mean_tps, 1),
                  "std_tokens_per_second": round(std_tps, 1),
                  "sem_tokens_per_second": round(se_tps, 1),
                  "n_batches": n_batches,
                  "seconds_needed": seconds, "days_point_estimate": round(days, 1),
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
            "Bootstrap extrapolation: %s days, CI [%s, %s] (n=%d, %d resamples)",
            result["days_point_estimate"],
            result["bootstrap_days_lower"],
            result["bootstrap_days_upper"],
            n, n_bootstrap,
        )
        return result
