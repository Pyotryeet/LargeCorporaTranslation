"""COMET-22 reference-based and COMETKiwi reference-free evaluation — neural MT quality metrics.

v2.0: Module-level model cache (P0-09) — the COMET model is downloaded and
loaded once per process lifetime, not on every ``compute_comet()`` call.
Includes a transformers 5.x compatibility shim (``_apply_comet_tokenizer_patch``)
and direct HuggingFace Hub download fallback for non-registry models (e.g., Kiwi).
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
            """Re-add the ``build_inputs_with_special_tokens`` method removed in transformers 5.x.

            This is a compatibility shim for COMET 2.2.7, which calls
            ``build_inputs_with_special_tokens`` directly.  The underlying logic
            still exists in transformers — it was just renamed — so this shim
            delegates to the newer method names.

            Args:
                token_ids_0: First sequence of token IDs (``list[int]``).
                token_ids_1: Optional second sequence of token IDs (``list[int]``
                    or ``None``).  When ``None``, delegates to single-sequence
                    builder; otherwise delegates to pair builder.

            Returns:
                ``list[int]`` — the combined token IDs with special tokens inserted.

            Notes:
                This function is defined inside ``_apply_comet_tokenizer_patch``
                and is bound to the tokenizer instance as a method via ``types.MethodType``.
                It is intentionally instance-scoped to avoid polluting the tokenizer
                base class globally.
            """
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

    COMET v2.2.7's ``download_model()`` only works for models in the COMET
    registry (wmt22-comet-da, etc.) and rejects Kiwi models.  This function
    bypasses that limitation by downloading the HF repo directly and
    returning the checkpoint path.

    Args:
        model_name: HuggingFace repo ID to download (e.g.
            ``"Unbabel/wmt22-cometkiwi-da"``).

    Returns:
        ``str`` — the absolute path to the ``checkpoints/model.ckpt`` file
        inside the downloaded repo snapshot.

    Raises:
        RuntimeError: If the model repo is gated (HTTP 401/403) on
            HuggingFace.  The exception message includes a link to request
            access.
        FileNotFoundError: If the ``checkpoints/model.ckpt`` file is not
            found inside the downloaded snapshot.
        Propagates ``huggingface_hub.snapshot_download`` network/filesystem
        exceptions on non-gating failures.
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
    """Return a cached COMET model, downloading only on first access (P0-09).

    Args:
        model_name: HuggingFace repo ID of the COMET model to load
            (e.g. ``"Unbabel/wmt22-comet-da"``).

    Returns:
        The loaded COMET model object on success, or ``None`` if
        ``HAS_COMET`` is ``False`` (the ``unbabel-comet`` package could not
        be imported).  Once loaded, the model is stored in
        ``_comet_model_cache`` and reused on subsequent calls.

    Side effects:
        On first access for a given ``model_name``:

        1. Downloads the model checkpoint via ``download_model`` (COMET
           registry) or falls back to ``_download_comet_model_direct``
           (direct HF snapshot) for Kiwi and other non-registry models.
        2. Patches the model's tokenizer instance for transformers 5.x
           compatibility via ``_apply_comet_tokenizer_patch``.
        3. Stores the loaded model in ``_comet_model_cache``.

        Logs progress at INFO level.

    Raises:
        FileNotFoundError: If the checkpoint file is missing after download.
        RuntimeError: If the model repo is gated on HuggingFace.
        Propagates any exception from ``load_from_checkpoint`` on failure.
    """
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
    """Evaluate translation quality using a reference-based COMET model.

    Args:
        sources: Source-language sentences as a list of strings.
        hypotheses: Model-generated (machine-translated) sentences as a list
            of strings.  Must be the same length as ``sources``.
        references: Human reference translations as a list of strings.  Must
            be the same length as ``sources``.
        model_name: HuggingFace repo ID of the COMET model to use.
            Defaults to ``DEFAULT_COMET_MODEL`` ("Unbabel/wmt22-comet-da").

    Returns:
        A ``dict`` with keys:

        * ``system_score``: The segment-aggregated COMET score as a ``float``
          rounded to 4 decimal places, or ``None`` on failure.
        * ``segments_scores``: A ``list[float]`` of per-segment COMET scores,
          each rounded to 4 decimal places.
        * ``low_quality_segments``: A ``list[dict]`` of segments whose COMET
          score falls below 0.4.  Each dict contains the original ``src``,
          ``mt``, ``ref``, and the ``comet`` score.
        * ``model``: The HuggingFace repo ID that was used (``str``).
        * ``error``: An error message string (only present when
          ``system_score`` is ``None``).

    Side effects:
        Downloads and caches the COMET model on first call (via
        ``_get_comet_model``).  Logs the system score at INFO level and
        errors at ERROR level.

    Raises:
        Does not raise — all exceptions are caught internally and returned
        as an error dict with ``system_score=None``.

    Notes:
        Requires ``unbabel-comet>=2.2.0`` (``HAS_COMET`` must be ``True``).
        When COMET is not installed, the function returns immediately with
        ``system_score=None`` and an error string.
    """
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
            """Convert a COMET segment score to a plain Python ``float`` defensively.

            COMET segment scores can be numpy scalars, numpy arrays, or plain
            Python floats.  This helper normalizes them to a single ``float``,
            averaging if the value is an array.

            Args:
                val: The raw score value — may be a ``float``, ``int``, numpy
                    scalar, ``numpy.ndarray``, or ``list``.

            Returns:
                ``float`` — the scalar value, or the mean if ``val`` is an
                array/list.  Returns ``0.0`` if conversion fails for any reason.

            Notes:
                This is a nested function inside ``compute_comet``, used to process
                per-segment scores defensively before rounding.
            """
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
    """Evaluate translation quality using a reference-free COMETKiwi model.

    COMETKiwi (wmt22-cometkiwi-da) correlates better with human judgments for
    English-Turkish than reference-based COMET, because high-quality reference
    translations are scarce for this language pair.

    Args:
        sources: Source-language sentences as a list of strings.
        hypotheses: Model-generated (machine-translated) sentences as a list
            of strings.  Must be the same length as ``sources``.
        model_name: HuggingFace repo ID of the COMETKiwi model to use.
            Defaults to ``DEFAULT_COMETKIWI_MODEL``
            ("Unbabel/wmt22-cometkiwi-da").

    Returns:
        A ``dict`` with keys:

        * ``system_score``: The segment-aggregated COMETKiwi score as a
          ``float`` rounded to 4 decimal places, or ``None`` on failure.
        * ``segments_scores``: A ``list[float]`` of per-segment COMETKiwi
          scores, each rounded to 4 decimal places.
        * ``low_quality_segments``: A ``list[dict]`` of segments whose
          COMETKiwi score falls below 0.4.  Each dict contains ``src``,
          ``mt``, and ``cometkiwi``.
        * ``model``: The HuggingFace repo ID that was used (``str``).
        * ``error``: An error message string (only present when
          ``system_score`` is ``None``).

    Side effects:
        Downloads and caches the COMETKiwi model on first call (via
        ``_get_comet_model``).  Logs the system score at INFO level and
        errors at ERROR level.

    Raises:
        Does not raise — all exceptions are caught internally and returned
        as an error dict with ``system_score=None``.

    Notes:
        Unlike ``compute_comet``, this function does **not** require
        reference translations.  It is a quality-estimation (QE) metric that
        operates on source-hypothesis pairs only.
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
