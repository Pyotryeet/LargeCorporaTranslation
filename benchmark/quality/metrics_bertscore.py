"""BERTScore — reference-free semantic evaluation for translation quality.

BERTScore uses a pretrained multilingual BERT model (bert-base-multilingual-cased)
to compute cosine similarity between source and hypothesis token embeddings.
It is reference-free (source+hypothesis only), correlates better with human
judgment than BLEU, and handles Turkish morphology naturally.

COMET-Kiwi is the academic standard but its models are gated on HuggingFace.
BERTScore is ungated, widely cited, and gives sensible scores for EN-TR.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from bert_score import BERTScorer

    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

# ── Module-level scorer cache ────────────────────────────────────────────────

_bertscore_scorer: Optional["BERTScorer"] = None
"""BERTScore scorer loaded once per process lifetime (the model is ~700MB)."""

BS_DEFAULT_MODEL = "bert-base-multilingual-cased"
BS_DEFAULT_BATCH_SIZE = 16


def _get_bertscore_scorer() -> Optional["BERTScorer"]:
    """Return a cached BERTScore scorer, loading on first access."""
    global _bertscore_scorer
    if _bertscore_scorer is not None:
        return _bertscore_scorer
    if not HAS_BERTSCORE:
        logger.error("bert-score not installed. Run: pip install bert-score>=0.3.13")
        return None
    logger.info(
        "Loading BERTScore model %s (first use, cached thereafter)", BS_DEFAULT_MODEL
    )
    _bertscore_scorer = BERTScorer(
        model_type=BS_DEFAULT_MODEL,
        num_layers=9,
        batch_size=BS_DEFAULT_BATCH_SIZE,
        idf=False,
        # lang=None + rescale_with_baseline=False: use raw cosine similarity
        # without language-specific IDF rescaling.  Cross-lingual pairs
        # (EN source, TR hypothesis) cannot use monolingual baseline rescaling
        # because the IDF statistics are language-specific.
        lang=None,
        rescale_with_baseline=False,
    )
    logger.info("BERTScore model cached: %s", BS_DEFAULT_MODEL)
    return _bertscore_scorer


def compute_bertscore(
    sources: list[str],
    hypotheses: list[str],
) -> dict:
    """Compute reference-free BERTScore for EN→TR translation quality.

    BERTScore computes the similarity between source and hypothesis
    token embeddings using a multilingual BERT model.  It is reference-free,
    making it suitable when only single references exist or when the
    reference wording differs from legitimate model output.

    Returns
    -------
    dict with system_score (mean F1), segments_scores, precision, recall.
    """
    if not HAS_BERTSCORE:
        logger.error("bert-score not installed")
        return {
            "system_score": None,
            "error": "bert-score not installed",
            "segments_scores": [],
        }
    if not sources or not hypotheses:
        logger.warning("Empty data for BERTScore")
        return {"system_score": 0.0, "segments_scores": []}

    try:
        scorer = _get_bertscore_scorer()
        if scorer is None:
            return {
                "system_score": None,
                "error": "BERTScore model not available",
                "segments_scores": [],
            }

        P, R, F1 = scorer.score(hypotheses, sources)
        scores = [float(f) for f in F1]
        sys_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(
            "BERTScore system score: %.4f (P=%.4f R=%.4f F1=%.4f)",
            sys_score,
            float(P.mean()),
            float(R.mean()),
            float(F1.mean()),
        )
        return {
            "system_score": round(sys_score, 4),
            "segments_scores": [round(s, 4) for s in scores],
            "precision": round(float(P.mean()), 4),
            "recall": round(float(R.mean()), 4),
            "model": BS_DEFAULT_MODEL,
            "method": "bertscore_reference_free",
        }
    except Exception as e:
        logger.error("BERTScore evaluation failed: %s", e)
        return {"system_score": None, "error": str(e), "segments_scores": []}


def clear_bertscore_cache():
    """Destroy cached BERTScore model and release all GPU memory.

    Deletes the model object entirely (not just moving to CPU), then
    forces MPS synchronization and cache clearing to prevent SIGSEGV
    when the next model loads onto the same device.
    """
    global _bertscore_scorer
    if _bertscore_scorer is not None:
        try:
            del _bertscore_scorer
        except Exception:
            pass
        _bertscore_scorer = None
    import torch
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()
    logger.info("BERTScore model cache cleared")
