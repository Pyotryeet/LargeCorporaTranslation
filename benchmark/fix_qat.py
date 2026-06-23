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
    """Apply all monkey-patches needed to load QAT models on transformers 5.x."""
    global _patch_applied
    if _patch_applied:
        return

    import transformers.modeling_utils

    # ── Patch 0: PreTrainedModel._init_weights — MUST run before model __init__ ──
    # _init_weights is a @classmethod on PreTrainedModel.
    # Save the underlying function via descriptor protocol.
    _orig_pretrained_init_weights = transformers.modeling_utils.PreTrainedModel.__dict__["_init_weights"].__func__

    @classmethod
    def _patched_pretrained_init_weights(cls, module):
        """Skip weight initialization for modules with packed weights."""
        if not hasattr(module, "weight") and hasattr(module, "weight_packed"):
            return
        _orig_pretrained_init_weights(cls, module)

    transformers.modeling_utils.PreTrainedModel._init_weights = _patched_pretrained_init_weights

    # ── Patch 1: Gemma4TextScaledWordEmbedding ──────────────────────────
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

    _orig_init = Gemma4TextScaledWordEmbedding.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if hasattr(self, "weight_packed") and not hasattr(self, "weight"):
            shape = getattr(self, "weight_shape", None)
            scale = self.weight_scale
            if shape is not None and shape.numel() == 2:
                vocab, emb_dim = int(shape[0].item()), int(shape[1].item())
            else:
                vocab, emb_dim = scale.shape[0], 1024
            dummy = torch.zeros(vocab, emb_dim, dtype=scale.dtype)
            self.register_parameter("weight", nn.Parameter(dummy))

    Gemma4TextScaledWordEmbedding.__init__ = _patched_init

    # ── Patch 2: module.to() passthrough for packed modules ──
    # The packed module's forward handles dequantization internally.
    # .weight is a materialized placeholder — after load_state_dict
    # replaces it, we need to keep it as a regular parameter.

    _patch_applied = True
    logger.debug("QAT patches applied: PreTrainedModel._init_weights + Gemma4TextScaledWordEmbedding")


def load_qat_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> tuple:
    """Load a Gemma 4 QAT model with compatibility patches.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID (e.g. ``google/gemma-4-E2B-it-qat-mobile-ct``).
    device : torch.device, optional
        Target device.  If None, loads to CPU (caller should move).

    Returns
    -------
    (model, tokenizer)
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
