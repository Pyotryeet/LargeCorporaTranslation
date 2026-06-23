"""Inference engine — model-agnostic translation facade (v3.0).

This module is now a thin dispatch layer over pluggable inference backends.
The heavy lifting lives in ``benchmark.inference.backends.*``.

v3.0: Model-agnostic dispatch
-----------------------------
- ``InferenceEngine`` → Creates the right backend via ``ModelRegistry``.
- ``translate(batch)`` → Delegates to ``backend.translate_batch(batch)``.
- ``load()`` → Delegates to ``backend.load()``.
- All backend-specific logic (AR generate, diffusion denoising, etc.) is
  encapsulated in the backend implementations.

Backward compatibility
-----------------------
The ``InferenceEngine`` API (``.translate()``, ``.load()``, ``.warmup()``,
``.is_loaded()``) is preserved.  Internal implementation is fully replaced
by backend dispatch.

Model type selection
--------------------
Determined by (in priority order):
  1. ``model.backend_type`` in config.yaml → explicit.
  2. Auto-detection from model architecture / config.
  3. Custom plugin match.
  4. Fallback to autoregressive.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from benchmark.hardware.backend import DeviceInfo
from benchmark.hardware.parallelism import TensorParallelConfig, get_tensor_parallel_config
from benchmark.inference.sampling import DecodingParams
from benchmark.inference.backends.protocol import (
    BackendConfig,
    BatchGenerationOutput,
    GenerationOutput,
    InferenceBackend,
)
from benchmark.inference.backends.registry import ModelRegistry

logger = logging.getLogger(__name__)

# Re-export for backward compatibility.
# These types have the same fields as before plus v3.0 additions.
TranslationResult = GenerationOutput
BatchResult = BatchGenerationOutput


class InferenceEngine:
    """Model-agnostic inference facade.

    Creates the appropriate backend (AR, diffusion, custom) based on model
    type detection and delegates all operations to it.  The public API is
    identical to v2.0 — existing code continues to work unchanged, but new
    model types are supported transparently.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device_info: DeviceInfo,
        tp_config: Optional[TensorParallelConfig] = None,
        decoding_params: Optional[DecodingParams] = None,
        use_flash_attention: bool = True,
        use_torch_compile: bool = True,
        max_input_tokens: int = 512,
        backend_type: str = "auto",
        extra: Optional[dict] = None,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device_info = device_info
        self.backend_name = device_info.backend
        self.tp_config = tp_config or get_tensor_parallel_config(device_info.num_devices)
        self.decoding_params = decoding_params or DecodingParams()
        self.use_flash_attention = use_flash_attention
        self.use_torch_compile = use_torch_compile
        self.max_input_tokens = max_input_tokens

        # Build backend config.
        backend_config = BackendConfig(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device_info=device_info,
            max_input_tokens=max_input_tokens,
            max_new_tokens=decoding_params.max_new_tokens if decoding_params else 512,
            temperature=decoding_params.temperature if decoding_params else 0.0,
            dtype="auto",
            use_flash_attention=use_flash_attention,
            use_torch_compile=use_torch_compile,
            extra=extra or {},
        )

        # Override backend_type in extra if explicitly provided.
        if backend_type != "auto":
            backend_config.extra["backend_type"] = backend_type

        # ── Create the appropriate backend ──
        registry = ModelRegistry()
        self._backend: InferenceBackend = registry.create_backend(backend_config)

        logger.info(
            "InferenceEngine: model=%s, backend=%s, type=%s",
            model_path,
            self._backend.display_name,
            self._backend.model_type.value,
        )

    @property
    def backend(self) -> InferenceBackend:
        """The underlying inference backend."""
        return self._backend

    @property
    def model(self) -> Optional[nn.Module]:
        """Raw model (delegated to backend)."""
        return self._backend.model

    @property
    def tokenizer(self):
        """Tokenizer (delegated to backend)."""
        return self._backend.tokenizer

    @property
    def devices(self) -> list[torch.device]:
        """Compute devices (delegated to backend)."""
        return self._backend.devices

    @property
    def model_type(self) -> str:
        """Model architecture type."""
        return self._backend.model_type.value

    @property
    def precision_config(self):
        """Precision configuration."""
        return self._backend.precision_config if hasattr(self._backend, 'precision_config') else None

    @property
    def _configured_batch_size(self) -> int:
        """Batch size used for warmup / CUDA graph capture (delegated to backend).

        No fallback default — every backend initialises this attribute in
        its __init__, so getattr with a hardcoded default would mask real
        initialization bugs.
        """
        return self._backend._configured_batch_size

    @_configured_batch_size.setter
    def _configured_batch_size(self, value: int) -> None:
        self._backend._configured_batch_size = value

    # ── Lifecycle ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model, tokenizer, and weights via the backend."""
        self._backend.load()

    def warmup(self, batches: int = 20) -> None:
        """Run warm-up batches via the backend."""
        self._backend.warmup(batches)

    def translate(self, batch: Any) -> BatchResult:
        """Translate a single pre-tokenised batch via the backend.

        Parameters
        ----------
        batch : Any
            Pre-tokenised batch from the data pipeline.
            Must provide the following attributes:
            - ``input_ids`` : torch.Tensor
            - ``attention_mask`` : torch.Tensor
            - ``raw_texts`` : list[str]
            - ``batch_id`` : int

        Returns
        -------
        BatchResult (BatchGenerationOutput)
        """
        return self._backend.translate_batch(batch)

    def is_loaded(self) -> bool:
        """Return True if the model is loaded and ready."""
        return self._backend.is_loaded()

    # ── Additional capabilities ────────────────────────────────────────

    def encode_source(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode source text (delegated to backend)."""
        return self._backend.encode_source(input_ids, attention_mask)

    def get_backend_info(self) -> dict:
        """Return backend metadata for reports."""
        return {
            "backend_display_name": self._backend.display_name,
            "model_type": self._backend.model_type.value,
            "capabilities": [
                c.name for c in type(self._backend.capabilities)
                if c in self._backend.capabilities
            ] if hasattr(self._backend.capabilities, '__iter__') else [],
            "is_loaded": self._backend.is_loaded(),
        }

    @property
    def display_name(self) -> str:
        return self._backend.display_name
