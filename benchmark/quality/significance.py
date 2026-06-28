"""Paired bootstrap statistical significance testing for model comparison.

Provides paired bootstrap resampling to determine whether one translation model
is statistically significantly better than another at a given confidence level.

Method
------
1. For each segment, compute the score difference between Model A and Model B.
2. Draw B bootstrap samples (with replacement) from the paired differences.
3. For each bootstrap sample, compute the mean difference.
4. The 95% confidence interval is the 2.5th and 97.5th percentiles of the
   bootstrap distribution.  If the CI excludes zero, the difference is
   statistically significant.

This is the standard paired bootstrap test recommended in the WMT shared
task methodology (Koehn, 2004; Graham et al., 2014).

References
----------
- Koehn, "Statistical Significance Tests for Machine Translation Evaluation", EMNLP 2004.
- Graham et al., "Is Machine Translation Getting Better over Time?", EACL 2014.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default number of bootstrap resamples.
DEFAULT_N_BOOTSTRAP = 10_000
# Default confidence level (95%).
DEFAULT_CONFIDENCE = 0.95


@dataclass
class SignificanceResult:
    """Result of a paired bootstrap significance test.

    Attributes
    ----------
    metric_name : str
        Name of the metric being compared (e.g., "xcomet_lite").
    model_a : str
        Identifier for the baseline/reference model.
    model_b : str
        Identifier for the challenger model.
    mean_diff : float
        Mean paired difference (positive = B is better).
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    significant : bool
        True if the CI excludes zero (i.e., one model is significantly better).
    winner : str | None
        Name of the winning model, or None if not significant.
    p_value_approx : float
        Approximate two-sided p-value from bootstrap distribution.
    n_segments : int
        Number of paired segments.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level used (e.g., 0.95).
    """

    metric_name: str
    model_a: str
    model_b: str
    mean_diff: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    significant: bool = False
    winner: str | None = None
    p_value_approx: float = 1.0
    n_segments: int = 0
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    confidence: float = DEFAULT_CONFIDENCE


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    *,
    metric_name: str = "score",
    model_a: str = "baseline",
    model_b: str = "challenger",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> SignificanceResult:
    """Run a paired bootstrap significance test.

    Parameters
    ----------
    scores_a : list[float]
        Per-segment scores for the baseline model.  Must be the same length
        as *scores_b* and aligned by segment index.
    scores_b : list[float]
        Per-segment scores for the challenger model.
    metric_name : str
        Label for the metric (appears in the result).
    model_a : str
        Identifier for the baseline model.
    model_b : str
        Identifier for the challenger model.
    n_bootstrap : int
        Number of bootstrap resamples (default 10,000).
    confidence : float
        Confidence level (default 0.95 = 95%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    SignificanceResult
    """
    import random

    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score lists must have the same length: "
            f"{len(scores_a)} (A) vs {len(scores_b)} (B)"
        )

    n = len(scores_a)
    if n < 2:
        logger.warning(
            "Paired bootstrap requires ≥2 segments (%d provided). "
            "Returning non-significant result.",
            n,
        )
        return SignificanceResult(
            metric_name=metric_name,
            model_a=model_a,
            model_b=model_b,
            n_segments=n,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
        )

    # Paired differences: d_i = score_b_i - score_a_i
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    mean_diff = sum(diffs) / n

    # Bootstrap the mean of paired differences.
    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()

    # Percentile confidence interval.
    alpha = 1.0 - confidence
    lo_idx = int(n_bootstrap * (alpha / 2.0))
    hi_idx = int(n_bootstrap * (1.0 - alpha / 2.0)) - 1
    ci_lower = bootstrap_means[max(0, lo_idx)]
    ci_upper = bootstrap_means[min(n_bootstrap - 1, hi_idx)]

    # Significance: CI excludes zero.
    significant = (ci_lower > 0) or (ci_upper < 0)

    winner: str | None = None
    if significant:
        winner = model_b if mean_diff > 0 else model_a

    # Approximate two-sided p-value: fraction of bootstrap means whose
    # absolute value exceeds |mean_diff|, doubled for two-sided test.
    abs_mean = abs(mean_diff)
    extreme = sum(1 for m in bootstrap_means if abs(m) >= abs_mean)
    p_value = extreme / n_bootstrap

    return SignificanceResult(
        metric_name=metric_name,
        model_a=model_a,
        model_b=model_b,
        mean_diff=round(mean_diff, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        significant=significant,
        winner=winner,
        p_value_approx=round(p_value, 6),
        n_segments=n,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )


@dataclass
class ModelComparisonReport:
    """Comparison of multiple models against a baseline across multiple metrics.

    Attributes
    ----------
    baseline : str
        Identifier of the baseline model.
    results : list[SignificanceResult]
        Per-metric significance results.
    summary : dict[str, list[str]]
        Metric name → list of models that are significantly better than baseline.
    """

    baseline: str
    results: list[SignificanceResult] = field(default_factory=list)
    summary: dict[str, list[str]] = field(default_factory=dict)


def compare_models(
    baseline_scores: dict[str, list[float]],
    challenger_scores: dict[str, dict[str, list[float]]],
    *,
    baseline_name: str = "baseline",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 42,
) -> ModelComparisonReport:
    """Compare multiple challenger models against a baseline across metrics.

    Parameters
    ----------
    baseline_scores : dict[str, list[float]]
        Per-metric per-segment scores for the baseline model.
        Example: ``{"xcomet_lite": [0.72, 0.65, ...], "bleu": [30.1, 25.4, ...]}``
    challenger_scores : dict[str, dict[str, list[float]]]
        Per-model per-metric per-segment scores.
        Example: ``{"model_x": {"xcomet_lite": [0.75, ...], ...}}``

    Returns
    -------
    ModelComparisonReport
    """
    report = ModelComparisonReport(baseline=baseline_name)
    summary: dict[str, list[str]] = {}

    for metric_name, baseline_seg_scores in baseline_scores.items():
        significantly_better: list[str] = []
        for challenger_name, challenger_metrics in challenger_scores.items():
            challenger_seg_scores = challenger_metrics.get(metric_name)
            if challenger_seg_scores is None:
                logger.warning(
                    "Metric '%s' missing for challenger '%s' — skipping",
                    metric_name, challenger_name,
                )
                continue

            try:
                result = paired_bootstrap_test(
                    baseline_seg_scores,
                    challenger_seg_scores,
                    metric_name=metric_name,
                    model_a=baseline_name,
                    model_b=challenger_name,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    seed=seed,
                )
                report.results.append(result)

                if result.significant and result.winner == challenger_name:
                    significantly_better.append(challenger_name)

            except (ValueError, TypeError) as e:
                logger.warning(
                    "Paired bootstrap failed for %s vs %s on %s: %s",
                    baseline_name, challenger_name, metric_name, e,
                )

        if significantly_better:
            summary[metric_name] = significantly_better

    report.summary = summary
    return report
