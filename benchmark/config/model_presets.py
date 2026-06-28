"""Model preset registry — single source of truth for supported model configurations.

Each preset carries architecture constants, model paths, quantization settings,
and platform compatibility flags.  All other modules should resolve architecture
defaults through this registry rather than hardcoding constants.

Registered presets (see ``MODEL_PRESETS`` dict for full details):
  translategemma-4b-bf16, translategemma-4b-int8, translategemma-4b-int4,
  ministral-3b-bf16, gemma4-e2b-qat-ct, gemma4-e2b-qat-int4,
  gemma4-e4b-qat-ct, gemma4-e4b-qat-int4, gemma4-e2b-q4_0,
  gemma4-e4b-q4_0, diffusiongemma-26b-a4b

Usage
-----
>>> from benchmark.config.model_presets import get_preset_by_name, resolve_architecture_defaults
>>> preset = get_preset_by_name("translategemma-4b-int8")
>>> arch = resolve_architecture_defaults("google/translategemma-4b-it", quantization="int8")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPreset:
    """Immutable configuration for a supported model.

    Attributes
    ----------
    name : str
        Short machine-readable name (e.g. ``"translategemma-4b-bf16"``).
    display_name : str
        Human-readable label for reports and logs.
    hf_model_id : str
        HuggingFace Hub model ID or local filesystem path.
    num_layers : int
        Number of transformer decoder layers.
    num_kv_heads : int
        Number of key/value attention heads (GQA — may differ from query heads).
    head_dim : int
        Dimension per attention head.
    hidden_size : int
        Hidden dimension of the transformer.
    vocab_size : int
        Vocabulary size (for memory planning, not tokenization).
    quantization : str
        ``"bf16"``, ``"fp16"``, ``"int8"``, or ``"int4"``.
    quantization_method : str
        Loading method: ``"none"``, ``"bitsandbytes-int8"``, ``"bitsandbytes-nf4"``, ``"awq"``.
    eos_token_id : int
        End-of-sequence token ID.
    end_of_turn_token_id : int
        Additional stop token (e.g. ``<end_of_turn>`` for Gemma).  Set to -1 if unused.
    max_seq_len : int
        Maximum sequence length the model was trained for.
    supports_mps : bool
        Whether this preset works on Apple Silicon MPS.
    supports_cuda : bool
        Whether this preset works on NVIDIA CUDA.
    supports_fp8 : bool
        Whether FP8 (native H200) is the recommended compute precision on CUDA.
    recommended_batch_size : int
        Safe starting batch size for auto-tuning.
    """

    name: str
    display_name: str
    hf_model_id: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    quantization: str = "bf16"
    quantization_method: str = "none"
    eos_token_id: int = 1
    end_of_turn_token_id: int = -1
    max_seq_len: int = 2048
    supports_mps: bool = True
    supports_cuda: bool = True
    supports_fp8: bool = True
    recommended_batch_size: int = 1

    @property
    def is_quantized(self) -> bool:
        """Return True if this preset uses a quantization method other than "none".

        Shortcut for checking whether any bitsandbytes (INT8/NF4) or other
        quantization scheme is active.  Equivalent to
        ``preset.quantization_method != "none"``.

        Returns
        -------
        bool
            ``True`` when ``quantization_method`` is not ``"none"``,
            ``False`` otherwise.
        """
        return self.quantization_method != "none"

    @property
    def bytes_per_element(self) -> float:
        """Return the number of bytes consumed by a single weight element.

        Derived from ``self.quantization``:
          - ``"int4"`` returns 0.5 (4-bit packed),
          - ``"int8"`` returns 1,
          - anything else (bf16, fp16) returns 2.

        Useful for memory-footprint estimates of weight matrices and KV caches.

        Returns
        -------
        float
            Byte count per weight element at this preset's quantization level.
        """
        if self.quantization in ("int4",):
            return 0.5  # 4 bits per weight = 0.5 bytes per element (packed)
        if self.quantization in ("int8",):
            return 1
        return 2  # bf16/fp16

    @property
    def kv_cache_bytes_per_layer(self) -> int:
        """Return the estimated KV-cache memory per layer, per sequence position.

        Computed as ``2 * num_kv_heads * head_dim * bytes_per_element``,
        representing the combined size of one key and one value vector across
        all KV heads for a single token position.  Multiply by ``num_layers``
        and ``max_seq_len`` to get the total KV-cache budget.

        Returns
        -------
        int
            Estimated byte count per layer per sequence position.

        Notes
        -----
        The formula assumes the KV cache uses the same precision as the weights
        (as reported by ``bytes_per_element``).  Some backends may store the
        cache in a different dtype; this is a planning estimate, not a
        guarantee.
        """
        return 2 * self.num_kv_heads * self.head_dim * self.bytes_per_element


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PRESETS: dict[str, ModelPreset] = {
    # ── Primary default (TE-FP8 safe) ────────────────────────────────────────
    "4B": ModelPreset(
        name="4B",
        display_name="Ministral 3B (BF16) ★ TE-FP8 SAFE",
        hf_model_id="mistralai/Ministral-3B-Instruct",
        num_layers=24,
        num_kv_heads=8,
        head_dim=128,
        hidden_size=2048,
        vocab_size=131_072,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=2,
        end_of_turn_token_id=-1,
        max_seq_len=4096,
        supports_fp8=True,
        recommended_batch_size=1,
    ),
    # ── TranslateGemma 4B variants (⚠️ TE FP8 crashes on attention) ────────
    "translategemma-4b-bf16": ModelPreset(
        name="translategemma-4b-bf16",
        display_name="TranslateGemma 4B (BF16)",
        hf_model_id="google/translategemma-4b-it",
        num_layers=36,
        num_kv_heads=4,
        head_dim=256,
        hidden_size=2560,    # Gemma 3 4B hidden_size=2560 (12B=3840, 27B=5376)
        vocab_size=262_144,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=1,
        end_of_turn_token_id=106,  # <end_of_turn>
        max_seq_len=2048,
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA
        recommended_batch_size=1,
    ),
    "translategemma-4b-int8": ModelPreset(
        name="translategemma-4b-int8",
        display_name="TranslateGemma 4B (INT8)",
        hf_model_id="google/translategemma-4b-it",
        num_layers=36,
        num_kv_heads=4,
        head_dim=256,
        hidden_size=2560,
        vocab_size=262_144,
        quantization="int8",
        quantization_method="bitsandbytes-int8",
        eos_token_id=1,
        end_of_turn_token_id=106,
        max_seq_len=2048,
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA
        recommended_batch_size=1,
    ),
    "translategemma-4b-int4": ModelPreset(
        name="translategemma-4b-int4",
        display_name="TranslateGemma 4B (INT4 NF4)",
        hf_model_id="google/translategemma-4b-it",
        num_layers=36,
        num_kv_heads=4,
        head_dim=256,
        hidden_size=2560,
        vocab_size=262_144,
        quantization="int4",
        quantization_method="bitsandbytes-nf4",
        eos_token_id=1,
        end_of_turn_token_id=106,
        max_seq_len=2048,
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA
        recommended_batch_size=1,
    ),
    # ── Ministral 3B ────────────────────────────────────────────────────────
    "ministral-3b-bf16": ModelPreset(
        name="ministral-3b-bf16",
        display_name="Ministral 3B (BF16) ★ TE-FP8 SAFE",
        hf_model_id="mistralai/Ministral-3B-Instruct",
        num_layers=24,
        num_kv_heads=8,      # GQA: 8 KV heads, 16 query heads
        head_dim=128,        # 2048 / 16 query heads = 128
        hidden_size=2048,
        vocab_size=131_072,  # Tekken tokenizer
        quantization="bf16",
        quantization_method="none",
        eos_token_id=2,      # Mistral uses </s>
        end_of_turn_token_id=-1,
        max_seq_len=4096,    # Ministral supports longer context
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA
        recommended_batch_size=1,
    ),
    # ── Gemma 4 E2B QAT variants (v3.4) ──────────────────────────────────────
    "gemma4-e2b-qat-ct": ModelPreset(
        name="gemma4-e2b-qat-ct",
        display_name="Gemma 4 E2B QAT (BF16)",
        hf_model_id="google/gemma-4-E2B-it-qat-mobile-ct",
        num_layers=26,       # Gemma 4 E2B — estimated, auto-detected at load
        num_kv_heads=4,
        head_dim=256,
        hidden_size=2560,
        vocab_size=262_144,
        quantization="bf16",
        quantization_method="none",  # QAT-trained but standard BF16 weights
        eos_token_id=1,
        end_of_turn_token_id=106,
        max_seq_len=2048,
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA
        recommended_batch_size=1,
    ),
    "gemma4-e2b-qat-int4": ModelPreset(
        name="gemma4-e2b-qat-int4",
        display_name="Gemma 4 E2B QAT (INT4 NF4)",
        hf_model_id="google/gemma-4-E2B-it-qat-mobile-ct",
        num_layers=26,
        num_kv_heads=4,
        head_dim=256,
        hidden_size=2560,
        vocab_size=262_144,
        quantization="int4",
        quantization_method="bitsandbytes-nf4",
        eos_token_id=1,
        end_of_turn_token_id=106,
        max_seq_len=2048,
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA
        recommended_batch_size=1,
    ),
    # ── Gemma 4 E4B QAT variants (v3.4) ──────────────────────────────────────
    "gemma4-e4b-qat-ct": ModelPreset(
        name="gemma4-e4b-qat-ct",
        display_name="Gemma 4 E4B QAT (BF16)",
        hf_model_id="google/gemma-4-E4B-it-qat-mobile-ct",
        num_layers=34,       # Gemma 4 E4B — estimated, auto-detected at load
        num_kv_heads=8,
        head_dim=256,
        hidden_size=3072,
        vocab_size=262_144,
        quantization="bf16",
        quantization_method="none",  # QAT-trained but standard BF16 weights
        eos_token_id=1,
        end_of_turn_token_id=106,
        max_seq_len=2048,
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA
        recommended_batch_size=1,
    ),
    "gemma4-e4b-qat-int4": ModelPreset(
        name="gemma4-e4b-qat-int4",
        display_name="Gemma 4 E4B QAT (INT4 NF4)",
        hf_model_id="google/gemma-4-E4B-it-qat-mobile-ct",
        num_layers=34,
        num_kv_heads=8,
        head_dim=256,
        hidden_size=3072,
        vocab_size=262_144,
        quantization="int4",
        quantization_method="bitsandbytes-nf4",
        eos_token_id=1,
        end_of_turn_token_id=106,
        max_seq_len=2048,
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA
        recommended_batch_size=1,
    ),
    # ── Gemma 4 E2B Q4_0 quantized (v3.4) ────────────────────────────────────
    "gemma4-e2b-q4_0": ModelPreset(
        name="gemma4-e2b-q4_0",
        display_name="Gemma 4 E2B Q4_0 (4-bit Pre-Quantized)",
        hf_model_id="google/gemma-4-E2B-it-qat-mobile-transformers",
        num_layers=26,
        num_kv_heads=4,
        head_dim=256,
        hidden_size=2560,
        vocab_size=262_144,
        quantization="int4",  # Q4_0 = 4-bit
        quantization_method="bitsandbytes-nf4",  # Load via bnb NF4 on CUDA; BF16 dequant on MPS
        eos_token_id=1,
        end_of_turn_token_id=106,
        max_seq_len=2048,
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA
        recommended_batch_size=1,
    ),
    # ── Gemma 4 E4B Q4_0 quantized (v3.4) ────────────────────────────────────
    "gemma4-e4b-q4_0": ModelPreset(
        name="gemma4-e4b-q4_0",
        display_name="Gemma 4 E4B Q4_0 (4-bit Pre-Quantized)",
        hf_model_id="google/gemma-4-E4B-it-qat-mobile-transformers",
        num_layers=34,
        num_kv_heads=8,
        head_dim=256,
        hidden_size=3072,
        vocab_size=262_144,
        quantization="int4",  # Q4_0 = 4-bit
        quantization_method="bitsandbytes-nf4",  # Load via bnb NF4 on CUDA; BF16 dequant on MPS
        eos_token_id=1,
        end_of_turn_token_id=106,
        max_seq_len=2048,
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA
        recommended_batch_size=1,
    ),
    # ── DiffusionGemma 26B-A4B (v3.4) ────────────────────────────────────────
    "diffusiongemma-26b-a4b": ModelPreset(
        name="diffusiongemma-26b-a4b",
        display_name="DiffusionGemma 26B-A4B (MoE Diffusion)",
        hf_model_id="google/diffusiongemma-26B-A4B-it",
        num_layers=48,       # 26B total, ~4B active MoE — estimated architecture
        num_kv_heads=8,
        head_dim=256,
        hidden_size=4096,
        vocab_size=262_144,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=1,
        end_of_turn_token_id=106,
        max_seq_len=2048,
        # DiffusionGemma is a large diffusion model (26B total, ~4B active MoE)
        # that requires ~48+ GB unified memory for MPS inference.  MPS is
        # disabled by default — enable it explicitly only on machines with
        # sufficient unified memory.
        supports_mps=False,
        supports_cuda=True,
        supports_fp8=True,      # StaticFP8Linear — always works on CUDA (diffusion backend)
        recommended_batch_size=1,
    ),
    "nllb-600m": ModelPreset(
        name="nllb-600m",
        display_name="NLLB 600M (BF16)",
        hf_model_id="facebook/nllb-200-distilled-600M",
        num_layers=12,
        num_kv_heads=16,
        head_dim=64,
        hidden_size=1024,
        vocab_size=256206,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=2,
        end_of_turn_token_id=-1,
        max_seq_len=1024,
        supports_fp8=False,
        recommended_batch_size=1,
    ),
    "nllb-1.3b": ModelPreset(
        name="nllb-1.3b",
        display_name="NLLB 1.3B (BF16)",
        hf_model_id="facebook/nllb-200-distilled-1.3B",
        num_layers=24,
        num_kv_heads=16,
        head_dim=64,
        hidden_size=1024,
        vocab_size=256206,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=2,
        end_of_turn_token_id=-1,
        max_seq_len=1024,
        supports_fp8=False,
        recommended_batch_size=1,
    ),
    "nllb-3b": ModelPreset(
        name="nllb-3b",
        display_name="NLLB 3B (BF16)",
        hf_model_id="facebook/nllb-200-3.3B",
        num_layers=24,
        num_kv_heads=16,
        head_dim=128,
        hidden_size=2048,
        vocab_size=256206,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=2,
        end_of_turn_token_id=-1,
        max_seq_len=1024,
        supports_fp8=False,
        recommended_batch_size=1,
    ),
    "madlad-3b": ModelPreset(
        name="madlad-3b",
        display_name="MADLAD 3B (BF16)",
        hf_model_id="google/madlad400-3b-mt",
        num_layers=32,
        num_kv_heads=16,
        head_dim=64,
        hidden_size=1024,
        vocab_size=256000,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=1,
        end_of_turn_token_id=-1,
        max_seq_len=2048,
        supports_fp8=False,
        recommended_batch_size=1,
    ),
}



# ═══════════════════════════════════════════════════════════════════════════════
# Lookup functions
# ═══════════════════════════════════════════════════════════════════════════════


def get_preset_by_name(name: str) -> Optional[ModelPreset]:
    """Look up a preset by its short name.

    Searches the ``MODEL_PRESETS`` registry for a key matching *name*
    (case-sensitive).  Typical keys are strings like
    ``"translategemma-4b-int8"`` or ``"ministral-3b-bf16"``.

    Parameters
    ----------
    name : str
        Short machine-readable preset name (e.g. ``"translategemma-4b-int8"``).

    Returns
    -------
    ModelPreset or None
        The matching immutable preset dataclass if *name* is registered,
        or ``None`` if the name is not found in the registry.
    """
    return MODEL_PRESETS.get(name)


# ── Dead functions removed (zero callers) ──────────────────────────────
# get_preset_by_model_id — zero external callers
# list_available_presets — zero external callers
# resolve_architecture_defaults — zero external callers
# resolve_preset — zero external callers
# These were replaced by ModelArchitecture.from_model() in
# benchmark/hardware/architecture.py (Flaw #9 fix).

