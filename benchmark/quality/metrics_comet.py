"""COMET-22 reference-based evaluation — neural MT quality metric.

v2.0: Module-level model cache (P0-09) — the COMET model is downloaded and
loaded once per process lifetime, not on every ``compute_comet()`` call.
"""

__all__ = [
    "compute_comet",
    "compute_comet_kiwi",
    "clear_comet_cache",
    "DEFAULT_COMET_MODEL",
    "DEFAULT_COMETKIWI_MODEL",
    "HAS_COMET",
    "HAS_TORCH",
]

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Monkey-patch for transformers 5.x compatibility ────────────────────
# transformers 5.x removed ``build_inputs_with_special_tokens`` from some
# tokenizer subclasses.  COMET 2.2.7 calls this method directly.
# Re-add it as a compatibility shim that delegates to the tokenizer's own
# internal implementation (which still exists — it was just renamed).


def _apply_comet_tokenizer_patch(tokenizer_instance):
    """Ensure ``build_inputs_with_special_tokens`` exists on *tokenizer_instance*.

    SCOPE: Only patches the **instance**, not the base class.  This
    prevents global corruption of tokenization across the process
    (a previous version monkey-patched PreTrainedTokenizerBase at the
    class level, which silently corrupts unrelated tokenizers).
    """
    if hasattr(tokenizer_instance, "build_inputs_with_special_tokens"):
        return  # already present, nothing to do
    try:
        from types import MethodType

        def _build_inputs_compat(self, token_ids_0, token_ids_1=None):
            """Re-add removed method for COMET 2.2.7 compatibility."""
            if token_ids_1 is None:
                return self.build_inputs_with_special_tokens_single_seq(token_ids_0)
            return self.build_inputs_with_special_tokens_pair(token_ids_0, token_ids_1)

        tokenizer_instance.build_inputs_with_special_tokens = MethodType(
            _build_inputs_compat, tokenizer_instance
        )
        logger.info("Applied COMET tokenizer compatibility patch (instance-scoped)")
    except Exception:
        logger.debug("COMET tokenizer patch not needed for this instance")


try:
    import torch
    import torch.cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from comet import download_model, load_from_checkpoint
    HAS_COMET = True
except ImportError:
    HAS_COMET = False

# ── Module-level COMET model cache (P0-09) ─────────────────────────────────

DEFAULT_COMET_MODEL = "Unbabel/wmt22-comet-da"
DEFAULT_COMETKIWI_MODEL = "Unbabel/wmt22-cometkiwi-da"
DEFAULT_COMET_BATCH_SIZE = 8

def _download_comet_model_direct(model_name: str):
    """Download a COMET model checkpoint via huggingface_hub directly.

    COMET v2.2.7's download_model() only works for models in the COMET
    registry (wmt22-comet-da, etc.) and rejects Kiwi models.  This
    bypasses that limitation by downloading the HF repo directly and
    returning the checkpoint path.
    """
    from huggingface_hub import snapshot_download
    from pathlib import Path

    try:
        model_path = snapshot_download(repo_id=model_name)
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            raise RuntimeError(
                f"COMET model '{model_name}' is gated on HuggingFace. "
                f"Request access at https://huggingface.co/{model_name} "
                f"or use a different COMET model via --comet-model."
            ) from e
        raise
    checkpoint = Path(model_path) / "checkpoints" / "model.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoints/model.ckpt found in {model_path}"
        )
    return str(checkpoint)


_comet_model_cache: dict[str, object] = {}
"""Cache: model_name → loaded COMET model.  One model per process lifetime."""


def _get_comet_model(model_name: str):
    """Return a cached COMET model, downloading only on first access (P0-09)."""
    if model_name not in _comet_model_cache:
        if not HAS_COMET:
            return None
        logger.info("Downloading COMET model %s (first use, cached thereafter)", model_name)
        try:
            model_path = download_model(model_name)
        except (KeyError, Exception) as e:
            # COMET v2.2.7 download_model() only supports registered models.
            # Kiwi models and others need a direct HF snapshot download.
            logger.info(
                "COMET download_model failed for %s — trying direct HF download: %s",
                model_name, e,
            )
            model_path = _download_comet_model_direct(model_name)
        model = load_from_checkpoint(model_path)
        # Patch the model's tokenizer instance (NOT the base class) for
        # transformers 5.x compatibility — COMET calls
        # build_inputs_with_special_tokens on its tokenizer.
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            _apply_comet_tokenizer_patch(model.tokenizer)
        _comet_model_cache[model_name] = model
        logger.info("COMET model cached: %s", model_name)
    return _comet_model_cache[model_name]


def clear_comet_cache():
    """Destroy cached COMET models and free all GPU memory.

    On MPS, Moving to CPU (.to("cpu")) is insufficient — the Lightning
    Trainer holds internal MPS state that persists across model boundaries.
    The next model's MPS allocation may reclaim pages still referenced by
    the trainer, causing a SIGSEGV.

    Fix: destroy the trainer FIRST, then move to CPU, then delete the
    model entirely, then force MPS synchronization + cache clearing.
    """
    if not _comet_model_cache:
        return
    for model_name, model in list(_comet_model_cache.items()):
        try:
            # Step 1: destroy Lightning trainer (holds MPS internal state).
            if hasattr(model, 'trainer') and model.trainer is not None:
                try:
                    model.trainer.teardown()
                except Exception:
                    pass
                model.trainer = None
            # Step 2: move to CPU (frees GPU tensors).
            model.to("cpu")
        except Exception:
            pass
        # Step 3: explicitly delete to break any remaining references.
        del model
    _comet_model_cache.clear()
    import gc
    gc.collect()
    if HAS_TORCH:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except Exception:
                pass
    gc.collect()  # second pass catches objects freed by empty_cache
    logger.info("COMET model cache cleared; GPU memory released")


def compute_comet(sources: list[str], hypotheses: list[str], references: list[str],
                  model_name: str = DEFAULT_COMET_MODEL) -> dict:
    if not HAS_COMET:
        logger.error("COMET not installed. Run: pip install unbabel-comet>=2.2.0")
        return {"system_score": None, "error": "COMET not installed", "segments_scores": []}
    if not sources or not hypotheses:
        logger.warning("Empty data for COMET")
        return {"system_score": 0.0, "segments_scores": []}
    try:
        model = _get_comet_model(model_name)
        if model is None:
            return {"system_score": None, "error": "COMET model not available", "segments_scores": []}

        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
        result = model.predict(data, batch_size=DEFAULT_COMET_BATCH_SIZE, progress_bar=False)
        # COMET 2.2.x returns a Prediction (dict subclass).  Tuple unpacking
        # iterates dict KEYS, not values — so seg_scores, system_score =
        # result gives the strings "scores" and "system_score" instead of
        # the actual values.  Use attribute access instead.
        if hasattr(result, 'scores') and hasattr(result, 'system_score'):
            seg_scores = result.scores
            try:
                sys_score = float(result.system_score)
            except (ValueError, TypeError):
                sys_score = 0.0
        else:
            seg_scores = result[0]
            try:
                sys_score = float(result[1])
            except (ValueError, TypeError):
                sys_score = 0.0
        # Convert segment scores defensively (numpy scalars, etc.).
        import numpy as _np
        def _to_float(val):
            try:
                if isinstance(val, (_np.ndarray, list)):
                    return float(_np.mean([float(x) for x in val]))
                return float(val)
            except (ValueError, TypeError):
                return 0.0
        scores = [_to_float(s) for s in seg_scores]
        low_quality = [{"src": s, "mt": h, "ref": r, "comet": sc}
                       for s, h, r, sc in zip(sources, hypotheses, references, scores) if sc < 0.4]
        logger.info("COMET-22 system score: %.4f, low-quality segments: %d", sys_score, len(low_quality))
        return {"system_score": round(sys_score, 4), "segments_scores": [round(s, 4) for s in scores],
                "low_quality_segments": low_quality, "model": model_name}
    except Exception as e:
        logger.error("COMET evaluation failed: %s", e)
        return {"system_score": None, "error": str(e), "segments_scores": []}


def compute_comet_kiwi(sources: list[str], hypotheses: list[str],
                       model_name: str = DEFAULT_COMETKIWI_MODEL) -> dict:
    """Reference-free COMETKiwi evaluation for EN-TR translation quality.

    wmt22-cometkiwi-da correlates better with human judgments for English-Turkish
    than reference-based COMET, because high-quality reference translations are
    scarce for this language pair.
    """
    if not HAS_COMET:
        logger.error("COMETKiwi not available. Run: pip install unbabel-comet>=2.2.0")
        return {"system_score": None, "error": "COMETKiwi not installed", "segments_scores": []}
    if not sources or not hypotheses:
        logger.warning("Empty data for COMETKiwi")
        return {"system_score": 0.0, "segments_scores": []}
    try:
        model = _get_comet_model(model_name)
        if model is None:
            return {"system_score": None, "error": "COMETKiwi model not available", "segments_scores": []}

        data = [{"src": s, "mt": h} for s, h in zip(sources, hypotheses)]
        result = model.predict(data, batch_size=DEFAULT_COMET_BATCH_SIZE, progress_bar=False)
        # COMET 2.2.x Prediction is a dict subclass — tuple unpacking gives KEYS.
        if hasattr(result, 'scores') and hasattr(result, 'system_score'):
            seg_scores = result.scores
            try:
                sys_score = float(result.system_score)
            except (ValueError, TypeError):
                sys_score = 0.0
        else:
            seg_scores = result[0]
            try:
                sys_score = float(result[1])
            except (ValueError, TypeError):
                sys_score = 0.0
        scores = []
        for s in seg_scores:
            try:
                scores.append(float(s))
            except (ValueError, TypeError):
                scores.append(0.0)
        low_quality = [{"src": s, "mt": h, "cometkiwi": sc}
                       for s, h, sc in zip(sources, hypotheses, scores) if sc < 0.4]
        logger.info("COMETKiwi system score: %.4f, low-quality segments: %d", sys_score, len(low_quality))
        return {"system_score": round(sys_score, 4), "segments_scores": [round(s, 4) for s in scores],
                "low_quality_segments": low_quality, "model": model_name}
    except Exception as e:
        logger.error("COMETKiwi evaluation failed: %s", e)
        return {"system_score": None, "error": str(e), "segments_scores": []}
