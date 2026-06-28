"""BERTScore — reference-hypothesis semantic similarity via BERTScore.

BERTScore computes cosine similarity between reference and hypothesis token
embeddings using a pretrained multilingual BERT model
(bert-base-multilingual-cased).  It evaluates how well the hypothesis
aligns with the reference translation semantically, complementing n-gram
metrics by capturing meaning rather than surface form overlap.

BERTScore correlates better with human judgment than BLEU and handles
Turkish morphology naturally.  However, it is not a substitute for
reference-based metrics (COMET-22, BLEU, chrF++) in formal evaluation
settings.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from bert_score import BERTScorer

    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

# ── Module-level scorer cache ────────────────────────────────────────────────

_bertscore_scorer: Optional["BERTScorer"] = None
_bertscore_lock = threading.Lock()

BS_DEFAULT_MODEL = "bert-base-multilingual-cased"
BS_DEFAULT_BATCH_SIZE = 16


def _get_bertscore_scorer() -> Optional["BERTScorer"]:
    """Return a cached BERTScore scorer, loading on first access.

    This function implements double-checked locking to ensure thread-safe lazy
    initialization of a singleton BERTScorer instance. On first call it loads
    the bert-base-multilingual-cased model (layer 9, no IDF, no baseline
    rescaling) and caches it in the module-level ``_bertscore_scorer`` global.

    Parameters
    ----------
    None

    Returns
    -------
    Optional[BERTScorer]
        The cached BERTScorer instance, or ``None`` if the ``bert-score``
        package is not installed.

    Side Effects
    ------------
    Sets the module-level ``_bertscore_scorer`` global on first successful
    load. Emits log messages (info on success, error if bert-score is
    missing).

    Notes
    -----
    Uses ``threading.Lock`` for mutual exclusion. Acquires the lock only
    when the scorer has not been initialized yet, avoiding contention on the
    hot path."""
    global _bertscore_scorer
    if _bertscore_scorer is not None:
        return _bertscore_scorer
    with _bertscore_lock:
        # Double-check after acquiring lock — another thread may have loaded.
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
            lang=None,
            rescale_with_baseline=False,
        )
        logger.info("BERTScore model cached: %s", BS_DEFAULT_MODEL)
        return _bertscore_scorer


def compute_bertscore(
    references: list[str],
    hypotheses: list[str],
) -> dict:
    """Compute BERTScore (F1, precision, recall) for reference-hypothesis pairs.

    Evaluates semantic similarity by encoding references and hypotheses with
    a pretrained multilingual BERT model and computing cosine similarity
    between token embeddings. Returns segment-level F1 scores and
    aggregate system-level metrics.

    Parameters
    ----------
    references : list[str]
        Ground-truth reference translations. Must be the same length as
        ``hypotheses`` and must not be empty.
    hypotheses : list[str]
        Candidate translations produced by the model under evaluation.
        Must be the same length as ``references`` and must not be empty.

    Returns
    -------
    dict
        A dictionary with the following keys:

        system_score : float or None
            Mean F1 across all segments, rounded to 4 decimal places.
            ``None`` if bert-score is not installed or the model failed
            to load.
        segments_scores : list[float]
            Per-segment F1 scores, rounded to 4 decimal places. Empty list
            on error.
        precision : float, optional
            Mean precision across all segments, rounded to 4 decimal places.
        recall : float, optional
            Mean recall across all segments, rounded to 4 decimal places.
        model : str, optional
            The BERT model identifier used for scoring.
        method : str, optional
            Always ``"bertscore_reference_based"``, identifying this metric.
        error : str, optional
            Present only on failure. Describes why scoring could not proceed.

    Raises
    ------
    Nothing explicitly; all exceptions are caught and returned as an error
    dict with ``system_score`` set to ``None``.

    Side Effects
    ------------
    Logs the system-level BERTScore (F1, precision, recall) at INFO level.
    Calls ``_get_bertscore_scorer()`` which may load and cache the BERT
    model on first invocation."""
    if not HAS_BERTSCORE:
        logger.error("bert-score not installed")
        return {
            "system_score": None,
            "error": "bert-score not installed",
            "segments_scores": [],
        }
    if not references or not hypotheses:
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

        P, R, F1 = scorer.score(hypotheses, references)
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
            "method": "bertscore_reference_based",
        }
    except Exception as e:
        logger.error("BERTScore evaluation failed: %s", e)
        return {"system_score": None, "error": str(e), "segments_scores": []}


def clear_bertscore_cache():
    """Destroy the cached BERTScore model and release associated GPU memory.

    Deletes the module-level ``_bertscore_scorer`` singleton, forces Python
    garbage collection, synchronizes the CUDA (or MPS) device, and empties
    the PyTorch CUDA/MPS memory allocator caches.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Side Effects
    ------------
    - Sets the module-level ``_bertscore_scorer`` global to ``None``.
    - Calls ``gc.collect()`` twice (before and after device synchronization).
    - If CUDA is available: calls ``torch.cuda.synchronize()`` and
      ``torch.cuda.empty_cache()``.
    - If MPS is available: calls ``torch.mps.synchronize()`` and
      ``torch.mps.empty_cache()``; exceptions during MPS cleanup are
      silently suppressed (MPS memory APIs are best-effort on macOS).
    - Emits an INFO log message when complete.

    Notes
    -----
    This function does **not** acquire ``_bertscore_lock``. Callers should
    ensure no concurrent ``compute_bertscore()`` calls are in flight (i.e.,
    call this after all scoring work is done) to avoid a race where a
    background thread reloads the scorer immediately after it is cleared."""
    global _bertscore_scorer
    import torch
    import gc

    if _bertscore_scorer is not None:
        try:
            del _bertscore_scorer
        except Exception:
            pass
        _bertscore_scorer = None

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
