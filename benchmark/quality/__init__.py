"""Quality benchmarking — BLEU, chrF++, COMET-22, BERTScore, COMET-Kiwi evaluation.

All five metrics run in parallel via ThreadPoolExecutor (quality/benchmark.py).
"""

from benchmark.quality.benchmark import QualityBenchmark, QualityResults
from benchmark.quality.benchmark import BLEU_TARGET_MIN, CHRF_TARGET_MIN, COMET_TARGET_MIN
from benchmark.quality.references import ReferenceLoader
from benchmark.quality.metrics_bleu import compute_bleu
from benchmark.quality.metrics_chrf import compute_chrf
from benchmark.quality.metrics_comet import compute_comet, compute_comet_kiwi
from benchmark.quality.metrics_bertscore import compute_bertscore

__all__ = [
    "QualityBenchmark", "QualityResults", "ReferenceLoader",
    "compute_bleu", "compute_chrf", "compute_comet",
    "compute_comet_kiwi", "compute_bertscore",
    "BLEU_TARGET_MIN", "CHRF_TARGET_MIN", "COMET_TARGET_MIN",
]
