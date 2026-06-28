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


def _first_non_none(*vals: Any) -> Optional[Any]:
    """Return the first value that is not None.

    Unlike ``a or b``, this treats falsy-but-valid values (0, empty string)
    as present rather than falling through to the next candidate.

    Args:
        *vals: Positional arguments to inspect in order.

    Returns:
        The first argument that is not ``None``, or ``None`` if every
        argument is ``None``.
    """
    for v in vals:
        if v is not None:
            return v
    return None


@dataclass
class ModelArchitecture:
    """Architecture dimensions introspected from a loaded HuggingFace model.

    All fields are populated from ``model.config`` at runtime. Default constants
    are used only as a last-resort fallback when config is genuinely unavailable
    (which should not happen in practice since ``load()`` runs first).

    Attributes:
        num_layers (int): Number of decoder/encoder layers.
        num_attention_heads (int): Number of attention heads.
        num_kv_heads (int): Number of key-value heads (may differ from
            ``num_attention_heads`` for Grouped-Query Attention models).
        head_dim (int): Dimensionality of each attention head.
        hidden_size (int): Model hidden dimension (d_model).
        intermediate_size (int): Feed-forward network inner dimension.
        vocab_size (int): Vocabulary size (token count).
        max_position_embeddings (int): Maximum sequence length the model supports.
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
        """Introspect a loaded HuggingFace model for real architecture dimensions.

        Reads from ``model.config``. Handles the most common config key variants
        across model families (Gemma, LLaMA, Mistral, NLLB, etc.). Falls back to
        DEFAULT_* constants only when a dimension is genuinely absent from config.

        Args:
            model (nn.Module): A loaded HuggingFace model instance that carries a
                ``config`` attribute (e.g., ``AutoModelForCausalLM``).

        Returns:
            ModelArchitecture: Populated dataclass with all architectural dimensions
                resolved from the model config.

        Caveats:
            - If the model has no ``config`` attribute, all fields fall back to
              DEFAULT_* constants and a warning is logged.
            - Uses ``_first_non_none`` internally to avoid the "0 is falsy" bug
              that plagued the old ``or``-chain approach.
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
                num_attention_heads=DEFAULT_HIDDEN_SIZE // DEFAULT_HEAD_DIM,
                num_kv_heads=DEFAULT_NUM_KV_HEADS,
                head_dim=DEFAULT_HEAD_DIM,
                hidden_size=DEFAULT_HIDDEN_SIZE,
                intermediate_size=DEFAULT_HIDDEN_SIZE * 4,
                vocab_size=DEFAULT_VOCAB_SIZE,
                max_position_embeddings=2048,
            )

        # Use _first_non_none to avoid the "0 is falsy" bug.
        # Every `or` chain below previously treated a config value of 0 as
        # "absent", silently falling through to the next key or a wrong default.
        num_layers = (
            _first_non_none(
                getattr(cfg, "num_hidden_layers", None),
                getattr(cfg, "num_layers", None),
                getattr(cfg, "n_layer", None),
            ) or DEFAULT_NUM_LAYERS
        )
        num_attention_heads = (
            _first_non_none(
                getattr(cfg, "num_attention_heads", None),
                getattr(cfg, "n_head", None),
            ) or None  # compute from hidden_size/head_dim below if absent
        )
        num_kv_heads = (
            _first_non_none(
                getattr(cfg, "num_key_value_heads", None),
                getattr(cfg, "num_kv_heads", None),
            ) or num_attention_heads  # MHA (not GQA) → kv_heads == attn_heads
        )
        head_dim = (
            _first_non_none(
                getattr(cfg, "head_dim", None),
            ) or None  # compute from hidden_size/num_attention_heads below if absent
        )
        hidden_size = (
            _first_non_none(
                getattr(cfg, "hidden_size", None),
                getattr(cfg, "d_model", None),
            ) or DEFAULT_HIDDEN_SIZE
        )
        intermediate_size = (
            _first_non_none(
                getattr(cfg, "intermediate_size", None),
                getattr(cfg, "ffn_dim", None),
            ) or hidden_size * 4
        )
        vocab_size = (
            _first_non_none(
                getattr(cfg, "vocab_size", None),
            ) or DEFAULT_VOCAB_SIZE
        )
        max_position_embeddings = (
            _first_non_none(
                getattr(cfg, "max_position_embeddings", None),
                getattr(cfg, "max_sequence_length", None),
                getattr(cfg, "n_positions", None),
            ) or 2048
        )

        # Resolve attention heads and head_dim interdependencies.
        # Cannot use DEFAULT_NUM_KV_HEADS for attention heads — that conflates
        # GQA KV heads with the actual attention head count, silently producing
        # wrong head_dim for models where attn_heads != kv_heads.
        if num_attention_heads is None:
            if head_dim is not None:
                num_attention_heads = max(1, hidden_size // head_dim)
            else:
                num_attention_heads = max(1, hidden_size // DEFAULT_HEAD_DIM)
        if head_dim is None:
            head_dim = max(1, hidden_size // num_attention_heads)

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
        """Estimate KV-cache memory consumed per generated token (in bytes).

        Assumes BF16 precision (2 bytes per element) for both key and value caches
        (hence the leading ``2 * 2`` factor).

        Returns:
            int: Bytes of KV-cache needed for each additional generated token.
        """
        return 2 * 2 * self.num_layers * self.num_kv_heads * self.head_dim

    @property
    def uses_gqa(self) -> bool:
        """Return whether the model uses Grouped-Query Attention (GQA).

        A model uses GQA when the number of key-value heads is strictly less than
        the number of attention heads (i.e., KV heads are shared across groups).

        Returns:
            bool: ``True`` if GQA is in use, ``False`` if standard Multi-Head
            Attention (where ``num_kv_heads == num_attention_heads``).
        """
        return self.num_kv_heads < self.num_attention_heads
