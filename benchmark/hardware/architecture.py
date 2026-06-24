"""Model architecture introspection from HF model config.

Replaces hardcoded DEFAULT_* constants with values read from the loaded
model's ``model.config`` at runtime.  This fixes Flaw #9 (Hardcoded
Architectural Assumptions) and eliminates the 4B-vs-12B constant mismatch.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelArchitecture:
    """Architecture dimensions introspected from a loaded model.

    All fields are populated from model.config with DEFAULT_* constant
    fallbacks only when config is unavailable (should not happen in practice
    since load() runs first).
    """
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelArchitecture":
        """Introspect a loaded HF model for real architecture dimensions.

        Reads from ``model.config``.  Handles the most common config key
        variants across model families (Gemma, LLaMA, Mistral, NLLB, etc.).
        Falls back to DEFAULT_* constants only when a dimension is genuinely
        absent from the config.
        """
        from benchmark.config.constants import (
            DEFAULT_NUM_LAYERS, DEFAULT_NUM_KV_HEADS, DEFAULT_HEAD_DIM,
            DEFAULT_HIDDEN_SIZE, DEFAULT_VOCAB_SIZE,
        )
        cfg = model.config if hasattr(model, "config") else None
        if cfg is None:
            logger.warning("Model has no .config attribute — using DEFAULT_* constants")
            return cls(
                num_layers=DEFAULT_NUM_LAYERS,
                num_attention_heads=DEFAULT_NUM_KV_HEADS,  # conservative
                num_kv_heads=DEFAULT_NUM_KV_HEADS,
                head_dim=DEFAULT_HEAD_DIM,
                hidden_size=DEFAULT_HIDDEN_SIZE,
                intermediate_size=DEFAULT_HIDDEN_SIZE * 4,
                vocab_size=DEFAULT_VOCAB_SIZE,
                max_position_embeddings=2048,
            )

        # Read with fallbacks — try multiple config key names
        num_layers = (
            getattr(cfg, "num_hidden_layers", None)
            or getattr(cfg, "num_layers", None)
            or getattr(cfg, "n_layer", None)
            or DEFAULT_NUM_LAYERS
        )
        num_attention_heads = (
            getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "n_head", None)
            or DEFAULT_NUM_KV_HEADS
        )
        num_kv_heads = (
            getattr(cfg, "num_key_value_heads", None)
            or getattr(cfg, "num_kv_heads", None)
            or num_attention_heads  # MHA (not GQA) → kv_heads == attn_heads
        )
        head_dim = (
            getattr(cfg, "head_dim", None)
            or getattr(cfg, "hidden_size", DEFAULT_HIDDEN_SIZE) // num_attention_heads
        )
        hidden_size = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", None)
            or DEFAULT_HIDDEN_SIZE
        )
        intermediate_size = (
            getattr(cfg, "intermediate_size", None)
            or getattr(cfg, "ffn_dim", None)
            or hidden_size * 4
        )
        vocab_size = (
            getattr(cfg, "vocab_size", None)
            or DEFAULT_VOCAB_SIZE
        )
        max_position_embeddings = (
            getattr(cfg, "max_position_embeddings", None)
            or getattr(cfg, "max_sequence_length", None)
            or getattr(cfg, "n_positions", None)
            or 2048
        )

        return cls(
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
        )

    @property
    def kv_cache_bytes_per_token(self) -> int:
        """Estimate KV-cache memory per token in bytes (BF16)."""
        return 2 * 2 * self.num_layers * self.num_kv_heads * self.head_dim

    @property
    def uses_gqa(self) -> bool:
        """True if the model uses Grouped-Query Attention."""
        return self.num_kv_heads < self.num_attention_heads
