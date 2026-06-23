"""Gemma 4 QAT model loader — works around transformers 5.x incompatibility.

The QAT mobile checkpoints use ``weight_packed`` / ``weight_scale`` instead
of standard ``.weight`` parameters.  Transformers 5.x has multiple call sites
that access ``module.weight`` during loading (device_map, initialize_weights,
accelerate meta-device path).  This module monkey-patches all of them to
handle the packed format.

Usage
-----
>>> from benchmark.fix_qat import load_qat_model
>>> model, tokenizer = load_qat_model("google/gemma-4-E2B-it-qat-mobile-ct")
>>> model.to("mps")
"""

from __future__ import annotations

import logging
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Patch: add .weight to Gemma4TextScaledWordEmbedding ────────────────
# The packed embedding stores weights as `weight_packed` (int32 quantized)
# and `weight_scale` (bfloat16).  Multiple code paths in transformers
# access `module.weight` — we synthesize it from the packed representation.

_patch_applied = False


def _apply_qat_patches() -> None:
    """Apply all monkey-patches needed to load QAT models on transformers 5.x.

    Two patches are applied:

    1. **Gemma4TextScaledWordEmbedding.__init__** — Adds a synthetic
       ``.weight`` parameter (zeros of correct shape/dtype) so that code
       paths in transformers that access ``module.weight`` (device_map,
       initialize_weights, accelerate meta-device path) do not crash.
       The actual QAT weights are stored in ``weight_packed`` (int32) and
       ``weight_scale`` (bfloat16) — the synthetic ``.weight`` is a
       compatibility shim that is overwritten when the real state dict
       loads.

    2. **Gemma4ForConditionalGeneration._init_weights** — Wraps the
       original initializer to skip modules that have ``weight_packed``
       but no ``.weight``.  This prevents a ``KeyError`` / ``AttributeError``
       during HF's weight initialization pass.

    The patches are idempotent — calling this function multiple times has
    no additional effect (guarded by ``_patch_applied``).
    """
    global _patch_applied
    if _patch_applied:
        return

    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextScaledWordEmbedding,
        Gemma4ForConditionalGeneration,
    )

    # ── Patch: Gemma4TextScaledWordEmbedding — add .weight property ────
    _orig_embed_init = Gemma4TextScaledWordEmbedding.__init__

    def _patched_embed_init(self, *args, **kwargs):
        _orig_embed_init(self, *args, **kwargs)
        if hasattr(self, "weight_packed") and not hasattr(self, "weight"):
            shape = getattr(self, "weight_shape", None)
            scale = self.weight_scale
            if shape is not None and shape.numel() == 2:
                v, e = int(shape[0].item()), int(shape[1].item())
            else:
                v, e = scale.shape[0], 1024
            self.register_parameter("weight", nn.Parameter(torch.zeros(v, e, dtype=scale.dtype)))

    Gemma4TextScaledWordEmbedding.__init__ = _patched_embed_init

    # ── Patch: Gemma4ForConditionalGeneration._init_weights ─────────────
    _orig_gemma4_iw = Gemma4ForConditionalGeneration._init_weights

    @classmethod
    def _patched_gemma4_iw(cls, module):
        if not hasattr(module, "weight") and hasattr(module, "weight_packed"):
            return
        _orig_gemma4_iw.__func__(cls, module)

    Gemma4ForConditionalGeneration._init_weights = _patched_gemma4_iw

    _patch_applied = True


def load_qat_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> tuple[torch.nn.Module, "PreTrainedTokenizerBase"]:
    """Load a Gemma 4 QAT model with compatibility patches.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID (e.g. ``google/gemma-4-E2B-it-qat-mobile-ct``).
    device : torch.device, optional
        Target device.  If None, loads to CPU (caller should move afterwards).

    Returns
    -------
    tuple[torch.nn.Module, PreTrainedTokenizerBase]
        A ``(model, tokenizer)`` pair.  The model is in eval mode and on the
        requested device (or CPU if device was None).

    Raises
    ------
    OSError
        If the model checkpoint cannot be found locally (local_files_only=True).
    ImportError
        If ``transformers`` is not installed or the Gemma4 model classes
        are not available in the installed version.
    """
    _apply_qat_patches()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load to CPU first (device_map=None avoids accelerate meta-device path).
    # Then move to target device manually.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        device_map=None,
        low_cpu_mem_usage=False,
        local_files_only=True,
    )
    model.eval()

    if device is not None:
        model = model.to(device)

    n = sum(p.numel() for p in model.parameters())
    logger.info("QAT model loaded: %s → %.2fB params", model_path, n / 1e9)
    return model, tokenizer
