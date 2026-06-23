"""Quality benchmarking — BLEU, chrF++, COMET-22 evaluation.

v2.0 additions: ensemble translation, confidence estimation,
back-translation verification, domain classification.
"""

from benchmark.quality.benchmark import QualityBenchmark, QualityResults
from benchmark.quality.benchmark import BLEU_TARGET_MIN, CHRF_TARGET_MIN, COMET_TARGET_MIN
from benchmark.quality.references import ReferenceLoader
from benchmark.quality.metrics_bleu import compute_bleu
from benchmark.quality.metrics_chrf import compute_chrf
from benchmark.quality.metrics_comet import compute_comet

__all__ = [
    "QualityBenchmark", "QualityResults", "ReferenceLoader",
    "compute_bleu", "compute_chrf", "compute_comet",
    "BLEU_TARGET_MIN", "CHRF_TARGET_MIN", "COMET_TARGET_MIN",
]
