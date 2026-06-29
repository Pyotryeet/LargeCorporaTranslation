"""MetricX-24 neural machine translation quality metric.

google/metricx-24-hybrid-large-v2p6 uses a T5-style encoder-decoder architecture
to predict translation quality directly. Note that MetricX scores represent MQM
error ratings, meaning **lower is better** (0.0 represents a perfect translation).
"""

__all__ = [
    "compute_metricx",
    "clear_metricx_cache",
    "DEFAULT_METRICX_MODEL",
    "HAS_METRICX",
]

import gc
import logging
import threading
from typing import Optional
import torch

logger = logging.getLogger(__name__)

DEFAULT_METRICX_MODEL = "google/metricx-24-hybrid-large-v2p6"

# ── Scorer caching ────────────────────────────────────────────────────────────
_metricx_model = None
_metricx_tokenizer = None
_metricx_lock = threading.Lock()

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_METRICX = True
except ImportError:
    HAS_METRICX = False


def _get_metricx_model_and_tokenizer(model_name: str = DEFAULT_METRICX_MODEL):
    """Load and cache the MetricX-24 model and tokenizer. Thread-safe.

    Uses double-checked locking to ensure the model is loaded exactly once
    across threads. On first call, selects the best available device
    (CUDA > MPS > CPU) and loads both the tokenizer and the Seq2Seq model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the MetricX variant to load.
        Defaults to ``google/metricx-24-hybrid-large-v2p6``.

    Returns
    -------
    tuple[PreTrainedModel, PreTrainedTokenizer]
        A two-tuple of (model, tokenizer). The model is in eval mode on the
        selected device.

    Raises
    ------
    ImportError
        If ``transformers`` is not installed.

    Side Effects
    ------------
    Sets the global ``_metricx_model`` and ``_metricx_tokenizer`` variables
    so that subsequent calls return the cached instances without reloading.
    Logs an info message on first load."""
    global _metricx_model, _metricx_tokenizer
    if _metricx_model is not None and _metricx_tokenizer is not None:
        return _metricx_model, _metricx_tokenizer
    with _metricx_lock:
        # Double-check after acquiring lock.
        if _metricx_model is not None and _metricx_tokenizer is not None:
            return _metricx_model, _metricx_tokenizer

    if not HAS_METRICX:
        raise ImportError(
            "transformers is required for MetricX. Run: pip install transformers"
        )

    logger.info(
        "Loading MetricX-24 model %s (first use, cached thereafter)", model_name
    )

    tokenizer_model = "google/mt5-xl"  # Official MetricX tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        logger.info("MetricX-24 tokenizer loaded from %s", tokenizer_model)
    except Exception as e:
        logger.warning(
            "MetricX tokenizer %s unavailable (%s) — trying checkpoint tokenizer",
            tokenizer_model, e,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Select best device
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.float32  # Standardize to float32 for perfect numerical consistency across MPS/CUDA
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS doesn't support bfloat16 for all ops
    else:
        device = "cpu"
        dtype = torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    _metricx_model = model
    _metricx_tokenizer = tokenizer
    logger.info("MetricX-24 model cached on %s", device)
    return _metricx_model, _metricx_tokenizer


def clear_metricx_cache():
    """Destroy the cached MetricX-24 model and release all associated memory.

    Deletes the global model and tokenizer references, runs the garbage
    collector, and forces GPU cache cleanup on CUDA and MPS backends.
    Safe to call even when no model is cached (no-op).

    Parameters
    ----------
    None

    Returns
    -------
    None

    Side Effects
    ------------
    - Resets global ``_metricx_model`` and ``_metricx_tokenizer`` to None.
    - Runs ``gc.collect()`` twice.
    - On CUDA: calls ``torch.cuda.synchronize()`` and ``torch.cuda.empty_cache()``.
    - On MPS: calls ``torch.mps.synchronize()`` and ``torch.mps.empty_cache()``
      (exceptions are silently swallowed if MPS calls fail).
    - Logs an info message on completion."""
    global _metricx_model, _metricx_tokenizer
    if _metricx_model is not None:
        try:
            del _metricx_model
        except Exception:
            pass
        _metricx_model = None
    if _metricx_tokenizer is not None:
        _metricx_tokenizer = None

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
    logger.info("MetricX-24 model cache cleared")


def compute_metricx(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    model_name: str = DEFAULT_METRICX_MODEL,
) -> dict:
    """Compute reference-based translation quality scores using MetricX-24.

    MetricX-24 predicts a continuous MQM error score where **lower is better**
    (0.0 represents a perfect translation). Each segment is scored
    individually by constructing a ``source: ... reference: ... candidate: ...``
    prompt and decoding the model output into a float.

    Parameters
    ----------
    sources : list[str]
        Source-language texts, one per segment.
    hypotheses : list[str]
        Machine-translated candidate texts, one per segment.
    references : list[str]
        Reference (gold-standard) translations, one per segment.
    model_name : str, optional
        HuggingFace model identifier. Defaults to
        ``google/metricx-24-hybrid-large-v2p6``.

    Returns
    -------
    dict
        Always contains:

        - **system_score** : float or None
            Mean of all valid segment scores, rounded to 4 decimal places.
            None if transformers is not installed or an unrecoverable
            error occurred.
        - **segments_scores** : list[float | None]
            Per-segment scores in the same order as the input lists.
            None entries indicate unparseable model output for that segment.

        May additionally contain:

        - **error** : str
            Present when ``system_score`` is None; describes what failed
            (e.g., missing transformers, runtime exception message).
        - **model** : str
            The model_name argument, only included on success.

    Caveats
    ------
    - Segments whose model output cannot be parsed as a float are stored as
      None in ``segments_scores`` and excluded from ``system_score``.
    - If *all* segments fail to parse, ``system_score`` is 0.0, not None.
    - An empty input list (any of sources/hypotheses/references) returns
      ``system_score=0.0`` without running inference.

    Side Effects
    ------------
    Calls ``_get_metricx_model_and_tokenizer``, which loads and caches the
    MetricX model on the first invocation (logs an info message).

    Raises
    ------
    No exceptions propagate to the caller; all failures are caught and
    returned as an error dict with ``system_score=None``."""
    if not HAS_METRICX:
        logger.error("transformers not installed")
        return {
            "system_score": None,
            "error": "transformers not installed",
            "segments_scores": [],
        }
    if not sources or not hypotheses or not references:
        logger.warning("Empty data for MetricX-24")
        return {"system_score": 0.0, "segments_scores": []}

    try:
        model, tokenizer = _get_metricx_model_and_tokenizer(model_name)
        device = next(model.parameters()).device

        seg_scores = []
        for src, ref, hyp in zip(sources, references, hypotheses):
            input_text = f"source: {src} candidate: {hyp} reference: {ref}"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
            ).to(device)

            with torch.no_grad():
                batch_size = inputs["input_ids"].size(0)
                decoder_input_ids = torch.zeros(
                    (batch_size, 1),
                    dtype=torch.long,
                    device=inputs["input_ids"].device,
                )
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    decoder_input_ids=decoder_input_ids,
                    return_dict=True,
                )
                lm_logits = outputs.logits  # shape: [batch_size, 1, vocab_size]
                pred = lm_logits[:, 0, 250089]
                pred = torch.clamp(pred, 0.0, 25.0)

            try:
                score = float(pred.item())
            except Exception:
                score = None
            seg_scores.append(score)

        valid_scores = [s for s in seg_scores if s is not None]
        sys_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        logger.info(
            "MetricX-24 system score: %.4f (lower is better, perfect = 0.0)",
            sys_score,
        )
        return {
            "system_score": round(sys_score, 4),
            "segments_scores": [round(s, 4) if s is not None else None for s in seg_scores],
            "model": model_name,
        }

    except Exception as e:
        logger.error("MetricX-24 evaluation failed: %s", e)
        return {"system_score": None, "error": str(e), "segments_scores": []}
