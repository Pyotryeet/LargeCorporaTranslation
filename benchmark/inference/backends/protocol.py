"""Abstract inference backend protocol (v3.0).

Defines the contract that all inference backends must fulfill.  This is the
central abstraction that decouples the harness and quality benchmark from
specific model architectures.

Backend types
-------------
==================== ==========  ==============================================
ModelType            Tokens/s   Characteristics
==================== ==========  ==============================================
AUTOREGRESSIVE       Sequential Sequential token generation via ``.generate()``.
DIFFUSION            Parallel   Iterative denoising in continuous embedding
                                space.  All tokens refined simultaneously.
CUSTOM               Variable   User-registered plugin implementing the
                                ``InferenceBackend`` protocol.
==================== ==========  ==============================================

Capability flags
----------------
Each backend declares what it supports via ``ModelCapability`` bitmask:

==================== ===========================================================
Flag                 Meaning
==================== ===========================================================
TRANSLATE            Can produce EN→TR translations.
FORWARD_ENCODE       Can produce source-side hidden states for analysis.
SCORE                Can score candidate translations.
CONFIDENCE           Can output per-token log-probabilities.
STREAMING            Can stream partial translations.
CLASSIFIER_FREE      Supports classifier-free guidance (diffusion models).
CUSTOM_KERNELS       Accepts user-provided Triton/Metal kernels.
ENSEMBLE_READY       Safe to use in ensemble translation.
QUANTIZABLE_KV       KV-cache supports quantization.
SPECULATIVE          Supports speculative decoding.
==================== ===========================================================
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntFlag, Enum
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelType(str, Enum):
    """Top-level model architecture category."""
    AUTOREGRESSIVE = "autoregressive"
    DIFFUSION = "diffusion"
    ENCODER_DECODER = "encoder_decoder"
    CUSTOM = "custom"


class ModelCapability(IntFlag):
    """Capability flags — composable via bitwise OR."""
    TRANSLATE = 1 << 0
    FORWARD_ENCODE = 1 << 1
    SCORE = 1 << 2
    CONFIDENCE = 1 << 3
    STREAMING = 1 << 4
    CLASSIFIER_FREE = 1 << 5
    CUSTOM_KERNELS = 1 << 6
    ENSEMBLE_READY = 1 << 7
    QUANTIZABLE_KV = 1 << 8
    SPECULATIVE = 1 << 9

    # Convenience presets
    FULL_TRANSLATION = TRANSLATE | FORWARD_ENCODE | CONFIDENCE
    FULL_DIFFUSION = TRANSLATE | FORWARD_ENCODE | CONFIDENCE | CLASSIFIER_FREE


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GenerationOutput:
    """Single generation output — backend-agnostic.

    Every backend must produce this structure regardless of whether it
    generates tokens autoregressively or via iterative denoising.
    """

    input_text: str
    translated_text: str
    input_tokens: int
    output_tokens: int
    total_latency_ms: float
    # Per-phase timings (backend-specific meaning).
    phase_timings: dict[str, float] = field(default_factory=dict)
    # Token-level log-probabilities (None if backend doesn't support).
    token_log_probs: Optional[list[float]] = None
    confidence: Optional[float] = None
    timestamp_utc: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchGenerationOutput:
    """Batch-level generation output."""

    batch_id: int
    generations: list[GenerationOutput] = field(default_factory=list)
    batch_size: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    total_latency_ms: float = 0.0
    phase_timings: dict[str, float] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        if self.total_latency_ms <= 0:
            return 0.0
        return (self.output_tokens_total / self.total_latency_ms) * 1000


@dataclass
class BackendConfig:
    """Configuration passed to backend constructors.

    Attributes
    ----------
    model_path : str
        HuggingFace model ID or local path.
    device_info : DeviceInfo
        Detected hardware information.
    max_input_tokens : int
        Maximum tokens per input chunk.
    max_new_tokens : int
        Maximum tokens to generate.
    dtype : str
        Precision mode.
    use_flash_attention : bool
        Enable FlashAttention / SDPA.
    use_torch_compile : bool
        Apply torch.compile() after loading.
    extra : dict
        Backend-specific configuration key-value pairs.  Recognized keys:
        safe_mode (bool), backend_type (str), do_sample (bool),
        num_beams (int), use_cuda_graph (bool), use_paged_attention (bool),
        use_tensorrt (bool), tensorrt (dict), diffusion (dict),
        plugin_name (str), plugin_config (dict), batch_size (int).
    """
    model_path: str = ""
    tokenizer_path: str = ""
    device_info: Any = None  # DeviceInfo
    max_input_tokens: int = 512
    max_new_tokens: int = 512
    temperature: float = 1.0
    dtype: str = "auto"
    use_flash_attention: bool = True
    use_torch_compile: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract backend protocol
# ---------------------------------------------------------------------------


class InferenceBackend(ABC):
    """Protocol that every inference backend must implement.

    Lifecycle
    ---------
    1. ``__init__(config)``   — construct with BackendConfig.
    2. ``load()``             — load model/weights/tokenizer.
    3. ``warmup(batches)``    — prime caches, compile, capture graphs.
    4. ``translate_batch(batch)`` — process pre-tokenised batches.
    5. ``encode(source)``     — (optional) produce encoder hidden states.

    Subclassing guide
    -----------------
    To add a new backend:

    1. Subclass ``InferenceBackend``.
    2. Implement ``load()``, ``translate_batch()``, ``warmup()``.
    3. Set ``model_type`` and ``capabilities`` class attributes.
    4. Register via ``ModelRegistry.register()`` or a ``CustomModelPlugin``.

    The harness and quality benchmark interact ONLY through this protocol —
    they never call ``model.generate()`` or any backend-specific method.
    """

    # ── Class-level identification ─────────────────────────────────────

    model_type: ModelType
    """Architecture category (AUTOREGRESSIVE, DIFFUSION, CUSTOM)."""

    capabilities: ModelCapability
    """Bitmask of supported capabilities."""

    display_name: str = "Base Backend"
    """Human-readable name for reports."""

    # ── Constructor ────────────────────────────────────────────────────

    def __init__(self, config: BackendConfig):
        self.config = config
        self._loaded = False
        self.device_info = config.device_info
        self.backend_name = self.device_info.backend if self.device_info else "unknown"
        self.devices: list[torch.device] = []
        self.tokenizer: Any = None
        self.model: Optional[nn.Module] = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """Load model weights, tokenizer, and move to device(s).

        Must set ``self._loaded = True`` on success.
        """
        ...

    @abstractmethod
    def warmup(self, batches: int = 20) -> None:
        """Run warm-up batches to prime CUDA graphs, caches, and JIT.

        Called once after ``load()``, before any ``translate()`` calls.
        """
        ...

    @abstractmethod
    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        """Translate a single pre-tokenised batch.

        Parameters
        ----------
        batch : PipelineBatch
            Batch from the data pipeline.  Contains ``input_ids``,
            ``attention_mask``, ``raw_texts``, etc.

        Returns
        -------
        BatchGenerationOutput
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is successfully loaded and ready."""
        return self._loaded

    # ── Optional capabilities ──────────────────────────────────────────

    def encode_source(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Produce source-side hidden states (optional, FORWARD_ENCODE cap).

        Default raises NotImplementedError.  Backends with the
        FORWARD_ENCODE capability should override this.

        Returns:
            torch.Tensor of shape [batch, src_len, hidden_size].
        """
        raise NotImplementedError(
            f"{self.display_name} does not support forward encoding."
        )

    def score_candidates(
        self,
        source_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> list[float]:
        """Score candidate translations (optional, SCORE cap)."""
        raise NotImplementedError(
            f"{self.display_name} does not support candidate scoring."
        )

    def get_token_log_probs(
        self, output_ids: torch.Tensor,
    ) -> list[float]:
        """Extract per-token log-probabilities from the last generation."""
        raise NotImplementedError(
            f"{self.display_name} does not support confidence estimation."
        )

    @property
    def kv_cache_config(self) -> dict[str, Any]:
        """Return KV-cache configuration for memory planning.

        Returns a dict with keys like ``num_layers``, ``num_kv_heads``,
        ``head_dim``, ``max_seq_len``.  Default returns empty dict.
        """
        return {}

    def supports_quantized_kv(self) -> bool:
        return ModelCapability.QUANTIZABLE_KV in self.capabilities

    def __repr__(self) -> str:
        return (
            f"<{self.display_name} type={self.model_type.value} "
            f"loaded={self._loaded} caps={self.capabilities!r}>"
        )
