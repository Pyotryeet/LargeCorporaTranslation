"""Model preset registry — single source of truth for supported model configurations.

Each preset carries architecture constants, model paths, and platform
compatibility flags.  All other modules should resolve architecture
defaults through this registry rather than hardcoding constants.

Registered presets (see ``MODEL_PRESETS`` dict for full details):
  nllb-600m, nllb-1.3b, nllb-3b, madlad-3b, translategemma-4b-bf16
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# Data class
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ModelPreset:
    """Immutable configuration for a single supported model architecture.

    Attributes
    ----------
    name : str
        Short key used for CLI lookup (e.g. ``"translategemma-4b-bf16"``).
    display_name : str
        Human-readable label for reports and tables.
    hf_model_id : str
        HuggingFace Hub model ID or local path.
    num_layers : int
        Number of decoder (or encoder+decoder) layers.
    num_kv_heads : int
        Number of key-value attention heads.
    head_dim : int
        Per-head dimension.
    hidden_size : int
        Model hidden dimension (d_model).
    vocab_size : int
        Vocabulary size.
    quantization : str
        Default quantization level (``"bf16"``, ``"int8"``, ``"int4"``).
    quantization_method : str
        How quantization is applied (``"none"``, ``"bitsandbytes"``, ``"qat"``).
    eos_token_id : int
        End-of-sequence token ID.
    end_of_turn_token_id : int
        End-of-turn token ID.  Set to ``-1`` if unused.
    max_seq_len : int
        Maximum sequence length the model was trained for.
    supports_fp8 : bool
        Whether the model architecture is compatible with FP8 quantization.
    recommended_batch_size : int
        Conservative safe batch size for initial tuning.
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
    eos_token_id: int = 2
    end_of_turn_token_id: int = -1
    max_seq_len: int = 4096
    supports_fp8: bool = False
    recommended_batch_size: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Model presets — 5 supported base architectures
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PRESETS: dict[str, ModelPreset] = {
    # ── NLLB 200 encoder-decoder family ────────────────────────────────────
    "nllb-600m": ModelPreset(
        name="nllb-600m",
        display_name="NLLB 200 600M (BF16)",
        hf_model_id="facebook/nllb-200-distilled-600M",
        num_layers=12,
        num_kv_heads=16,
        head_dim=64,
        hidden_size=1024,
        vocab_size=256_000,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=2,
        end_of_turn_token_id=-1,
        max_seq_len=1024,
        supports_fp8=False,
        recommended_batch_size=64,
    ),
    "nllb-1.3b": ModelPreset(
        name="nllb-1.3b",
        display_name="NLLB 200 1.3B (BF16)",
        hf_model_id="facebook/nllb-200-distilled-1.3B",
        num_layers=12,
        num_kv_heads=16,
        head_dim=64,
        hidden_size=1024,
        vocab_size=256_000,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=2,
        end_of_turn_token_id=-1,
        max_seq_len=1024,
        supports_fp8=False,
        recommended_batch_size=32,
    ),
    "nllb-3b": ModelPreset(
        name="nllb-3b",
        display_name="NLLB 200 3.3B (BF16)",
        hf_model_id="facebook/nllb-200-3.3B",
        num_layers=24,
        num_kv_heads=16,
        head_dim=64,
        hidden_size=2048,
        vocab_size=256_000,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=2,
        end_of_turn_token_id=-1,
        max_seq_len=1024,
        supports_fp8=False,
        recommended_batch_size=8,
    ),
    # ── MADLAD 400 encoder-decoder ─────────────────────────────────────────
    "madlad-3b": ModelPreset(
        name="madlad-3b",
        display_name="MADLAD 400 3B (BF16)",
        hf_model_id="google/madlad400-3b-mt",
        num_layers=24,
        num_kv_heads=16,
        head_dim=64,
        hidden_size=2048,
        vocab_size=256_000,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=2,
        end_of_turn_token_id=-1,
        max_seq_len=1024,
        supports_fp8=False,
        recommended_batch_size=8,
    ),
    # ── TranslateGemma 4B autoregressive ───────────────────────────────────
    "translategemma-4b-bf16": ModelPreset(
        name="translategemma-4b-bf16",
        display_name="TranslateGemma 4B (BF16)",
        hf_model_id="google/translategemma-4b-it",
        num_layers=36,
        num_kv_heads=4,
        head_dim=256,
        hidden_size=2560,
        vocab_size=262_144,
        quantization="bf16",
        quantization_method="none",
        eos_token_id=2,
        end_of_turn_token_id=106,
        max_seq_len=2048,
        supports_fp8=True,
        recommended_batch_size=32,
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Lookup functions
# ═══════════════════════════════════════════════════════════════════════════════


def get_preset_by_name(name: str) -> Optional[ModelPreset]:
    """Look up a preset by its short name.

    Searches the ``MODEL_PRESETS`` registry for a key matching *name*
    (case-sensitive).  Returns ``None`` when the name is not found.

    Args:
        name: Short key such as ``"nllb-600m"`` or ``"translategemma-4b-bf16"``.

    Returns:
        ``ModelPreset`` or ``None``.
    """
    return MODEL_PRESETS.get(name)


def resolve_architecture_defaults(
    preset_name: str,
) -> tuple[int, int, int, int, int]:
    """Return (num_layers, num_kv_heads, head_dim, hidden_size, vocab_size)
    for *preset_name*, falling back to DEFAULT_* constants on miss.

    Args:
        preset_name: Short key or HuggingFace model ID.

    Returns:
        Tuple of five architecture integers.
    """
    from benchmark.config.constants import (
        DEFAULT_NUM_LAYERS, DEFAULT_NUM_KV_HEADS, DEFAULT_HEAD_DIM,
        DEFAULT_HIDDEN_SIZE, DEFAULT_VOCAB_SIZE,
    )

    preset = MODEL_PRESETS.get(preset_name)
    if preset is not None:
        return (
            preset.num_layers, preset.num_kv_heads,
            preset.head_dim, preset.hidden_size, preset.vocab_size,
        )
    return (
        DEFAULT_NUM_LAYERS, DEFAULT_NUM_KV_HEADS,
        DEFAULT_HEAD_DIM, DEFAULT_HIDDEN_SIZE, DEFAULT_VOCAB_SIZE,
    )
