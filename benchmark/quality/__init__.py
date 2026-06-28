"""Quality benchmarking — BLEU, chrF++, COMET-22, xCOMET-lite, BERTScore, COMET-Kiwi, MetricX-24.

All seven metrics run in parallel via ThreadPoolExecutor (quality/benchmark.py).
Statistical significance testing between model runs is provided by quality/significance.py.
"""

from benchmark.quality.benchmark import QualityBenchmark, QualityResults
from benchmark.quality.benchmark import BLEU_TARGET_MIN, CHRF_TARGET_MIN, COMET_TARGET_MIN
from benchmark.quality.references import ReferenceLoader
from benchmark.quality.metrics_bleu import compute_bleu
from benchmark.quality.metrics_chrf import compute_chrf
from benchmark.quality.metrics_comet import compute_comet, compute_comet_kiwi
from benchmark.quality.metrics_bertscore import compute_bertscore
from benchmark.quality.metrics_metricx import compute_metricx
from benchmark.quality.metrics_xcomet import compute_xcomet, clear_xcomet_cache
from benchmark.quality.significance import (
    SignificanceResult,
    ModelComparisonReport,
    paired_bootstrap_test,
    compare_models,
)

__all__ = [
    "QualityBenchmark", "QualityResults", "ReferenceLoader",
    "compute_bleu", "compute_chrf", "compute_comet",
    "compute_comet_kiwi", "compute_bertscore", "compute_metricx",
    "compute_xcomet", "clear_xcomet_cache",
    "SignificanceResult", "ModelComparisonReport",
    "paired_bootstrap_test", "compare_models",
    "BLEU_TARGET_MIN", "CHRF_TARGET_MIN", "COMET_TARGET_MIN",
]
