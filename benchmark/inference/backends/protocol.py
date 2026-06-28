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
ENCODER_DECODER      Sequential Encoder-decoder models (e.g., NLLB).
VLLM                 Sequential  vLLM-backed inference engine.
CUSTOM               Variable   User-registered plugin implementing the
                                ``InferenceBackend`` protocol.
==================== ==========  ==============================================

Capability flags
----------------
Each backend declares what it supports via ``ModelCapability`` bitmask:

==================== ===========================================================
Flag                 Meaning
==================== ===========================================================
TRANSLATE            Can produce EN->TR translations.
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
from typing import Any, Optional

import torch
import torch.nn as nn

from benchmark.hardware.backend import detect_backend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelType(str, Enum):
    """Enumeration of top-level model architecture categories.

    Values
    ------
    AUTOREGRESSIVE : str
        Standard causal language models (e.g., Gemma, GPT-style).
    DIFFUSION : str
        Diffusion-based translation models operating in continuous embedding space.
    ENCODER_DECODER : str
        Encoder-decoder models (e.g., NLLB).
    CUSTOM : str
        User-registered plugin backends.
    VLLM : str
        vLLM-backed inference engine for high-throughput serving.
    """
    AUTOREGRESSIVE = "autoregressive"
    DIFFUSION = "diffusion"
    ENCODER_DECODER = "encoder_decoder"
    CUSTOM = "custom"
    VLLM = "vllm"


class ModelCapability(IntFlag):
    """Capability flags composable via bitwise OR (IntFlag).

    These flags declare what a backend supports. Backends set their capabilities
    as a bitmask of these values in the ``capabilities`` class attribute.

    Individual flags
    ----------------
    TRANSLATE : int
        Can produce translations.
    FORWARD_ENCODE : int
        Can produce source-side hidden states via ``encode_source()``.
    SCORE : int
        Can score candidate translations via ``score_candidates()``.
    CONFIDENCE : int
        Can output per-token log-probabilities via ``get_token_log_probs()``.
    STREAMING : int
        Can stream partial translations as they are generated.
    CLASSIFIER_FREE : int
        Supports classifier-free guidance (diffusion models).
    CUSTOM_KERNELS : int
        Accepts user-provided Triton or Metal custom kernels.
    ENSEMBLE_READY : int
        Safe to use in ensemble translation setups.
    QUANTIZABLE_KV : int
        KV-cache supports quantization.
    SPECULATIVE : int
        Supports speculative (draft-model) decoding.

    Convenience presets
    -------------------
    FULL_TRANSLATION : ModelCapability
        TRANSLATE | FORWARD_ENCODE | CONFIDENCE
    FULL_DIFFUSION : ModelCapability
        TRANSLATE | FORWARD_ENCODE | CONFIDENCE | CLASSIFIER_FREE
    """
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
    """Batch-level generation output.

    Attributes
    ----------
    batch_id : int
        Sequential batch identifier.
    generations : list[GenerationOutput]
        Per-input generation results.
    batch_size : int
        Number of inputs in the batch.
    input_tokens_total : int
        Sum of input token counts across all generations.
    output_tokens_total : int
        Sum of output token counts across all generations.
    total_latency_ms : float
        End-to-end wall-clock time for the batch in milliseconds.
    phase_timings : dict[str, float]
        Per-phase timing breakdowns (backend-specific interpretation).
    """

    batch_id: int
    generations: list[GenerationOutput] = field(default_factory=list)
    batch_size: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    total_latency_ms: float = 0.0
    phase_timings: dict[str, float] = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Compute tokens-per-second throughput for the batch.

        Returns
        -------
        float
            Output tokens per second (output_tokens_total / total_latency_ms * 1000).
            Returns 0.0 if total_latency_ms is zero or negative (avoids division by zero).
        """
        if self.total_latency_ms <= 0:
            return 0.0
        return (self.output_tokens_total / self.total_latency_ms) * 1000


@dataclass
class BackendConfig:
    """Configuration dataclass passed to InferenceBackend constructors.

    Attributes
    ----------
    model_path : str
        HuggingFace model ID or local filesystem path.
    tokenizer_path : str
        Path to tokenizer files. If empty, inferred from model_path.
    device_info : DeviceInfo
        Detected hardware information (backend type, device count, memory, etc.).
    max_input_tokens : int
        Maximum tokens per input chunk (default 512).
    max_new_tokens : int
        Maximum tokens to generate (default 512).
    temperature : float
        Sampling temperature (default 1.0).
    dtype : str
        Precision mode ("auto", "fp16", "bf16", "fp32", "fp8").
    use_flash_attention : bool
        Enable FlashAttention / SDPA (default True).
    use_torch_compile : bool
        Apply torch.compile() after loading (default True).
    extra : dict[str, Any]
        Backend-specific key-value pairs.  Recognized keys: safe_mode (bool),
        backend_type (str), do_sample (bool), num_beams (int),
        use_paged_attention (bool), diffusion (dict), plugin_name (str),
        plugin_config (dict), batch_size (int).
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
        """Initialize the backend with configuration.

        Parameters
        ----------
        config : BackendConfig
            Configuration specifying model path, precision, hardware info, and
            backend-specific options via ``config.extra``.

        Notes
        -----
        Subclasses should call ``super().__init__(config)`` and then perform
        any additional setup. The model is NOT loaded at this point — call
        ``load()`` separately.
        """
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

        Must set ``self._loaded = True`` on success. May load from a HuggingFace
        model ID, a local checkpoint, or a custom serialization format.

        Raises
        ------
        OSError
            If model_path does not exist or is not accessible.
        RuntimeError
            If the model architecture is incompatible with the detected hardware.

        Notes
        -----
        This is a potentially slow, blocking I/O operation. Called once per
        backend instance.
        """
        ...

    @abstractmethod
    def warmup(self, batches: int = 20) -> None:
        """Run warm-up batches to prime CUDA graphs, caches, and JIT compilation.

        Called once after ``load()``, before any ``translate()`` / ``translate_batch()``
        calls. This triggers torch.compile() compilation, CUDA graph capture, and
        KV-cache pre-allocation so that the first real inference request does not
        incur cold-start latency.

        Parameters
        ----------
        batches : int
            Number of warm-up forward passes to run (default 20).

        Notes
        -----
        Warm-up inputs are typically dummy tensors sized to ``max_input_tokens``
        and ``max_new_tokens``. Subclasses should guard against division-by-zero
        or other errors on dummy data.
        """
        ...

    @abstractmethod
    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        """Translate a single pre-tokenised batch.

        Parameters
        ----------
        batch : PipelineBatch
            Batch from the data pipeline.  Contains ``input_ids``,
            ``attention_mask``, ``raw_texts``, and any pipeline metadata
            needed for generation.

        Returns
        -------
        BatchGenerationOutput
            Aggregated output containing per-input translations, token counts,
            latency measurements, and optional confidence scores.

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called or the model is not ready.
        """
        ...

    def is_loaded(self) -> bool:
        """Return True if the model is successfully loaded and ready for inference.

        Returns
        -------
        bool
            True if the model, tokenizer, and device are fully initialized.

        Notes
        -----
        Subclasses should set ``self._loaded = True`` in their ``load()`` method.
        Override this property only if the loaded-state check requires more than
        reading the ``_loaded`` flag.
        """
        return self._loaded

    # ── Optional capabilities ──────────────────────────────────────────

    def encode_source(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Produce source-side encoder hidden states (optional, FORWARD_ENCODE cap).

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenized source input with shape [batch, src_len].
        attention_mask : torch.Tensor
            Attention mask with shape [batch, src_len].

        Returns
        -------
        torch.Tensor
            Encoder hidden states of shape [batch, src_len, hidden_size].

        Raises
        ------
        NotImplementedError
            If the backend does not set the FORWARD_ENCODE capability flag.
            Override this method in subclasses that support forward encoding.
        """
        raise NotImplementedError(
            f"{self.display_name} does not support forward encoding."
        )

    def score_candidates(
        self,
        source_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> list[float]:
        """Score candidate translations (optional, SCORE cap).

        Parameters
        ----------
        source_ids : torch.Tensor
            Tokenized source input.
        candidate_ids : torch.Tensor
            Tokenized candidate translation(s) to score.

        Returns
        -------
        list[float]
            Scores for each candidate (higher is better). Interpretation is
            backend-specific (log-likelihood, BLEURT, COMET, etc.).

        Raises
        ------
        NotImplementedError
            If the backend does not set the SCORE capability flag.
        """
        raise NotImplementedError(
            f"{self.display_name} does not support candidate scoring."
        )

    def get_token_log_probs(
        self, output_ids: torch.Tensor,
    ) -> list[float]:
        """Extract per-token log-probabilities from the most recent generation.

        Parameters
        ----------
        output_ids : torch.Tensor
            Token IDs from the most recent ``translate_batch()`` output.

        Returns
        -------
        list[float]
            Log-probability for each token position. Length matches the number
            of generated tokens.

        Raises
        ------
        NotImplementedError
            If the backend does not set the CONFIDENCE capability flag.
        """
        raise NotImplementedError(
            f"{self.display_name} does not support confidence estimation."
        )

    @property
    def kv_cache_config(self) -> dict[str, Any]:
        """Return KV-cache geometry for memory planning and paged attention.

        Returns
        -------
        dict[str, Any]
            Dict with keys such as ``num_layers``, ``num_kv_heads``, ``head_dim``,
            ``max_seq_len``. Returns an empty dict if the backend does not expose
            KV-cache metadata.

        Notes
        -----
        Used by the batch tuner and memory planner to pre-allocate KV-cache blocks
        before starting inference.
        """
        return {}

    def supports_quantized_kv(self) -> bool:
        """Check whether the backend supports KV-cache quantization.

        Returns
        -------
        bool
            True if ``ModelCapability.QUANTIZABLE_KV`` is set in this backend's
            capability bitmask.
        """
        return ModelCapability.QUANTIZABLE_KV in self.capabilities

    def __repr__(self) -> str:
        return (
            f"<{self.display_name} type={self.model_type.value} "
            f"loaded={self._loaded} caps={self.capabilities!r}>"
        )


# ---------------------------------------------------------------------------
# Hardware dispatcher — routes to CUDA or MPS backend at load time
# ---------------------------------------------------------------------------


class HardwareDispatcherBackend(InferenceBackend):
    """Base class for backends that dispatch to platform-specific implementations.

    Subclasses set ``_cuda_module``, ``_cuda_class``, ``_mps_module``, ``_mps_class``
    and the standard ``model_type``, ``capabilities``, ``display_name`` attributes.
    The __init__ method detects hardware and imports the appropriate implementation.
    All ``InferenceBackend`` protocol methods forward to ``self._impl`` via
    ``__getattr__`` / ``__setattr__`` delegation.
    """

    _cuda_module: str = ""
    _cuda_class: str = ""
    _mps_module: str = ""
    _mps_class: str = ""

    def __init__(self, config: BackendConfig):
        """Detect hardware and import the platform-specific backend implementation.

        Parameters
        ----------
        config : BackendConfig
            Configuration used to detect hardware and initialize the delegated
            implementation.

        Notes
        -----
        - On CUDA systems, imports ``self._cuda_module`` and instantiates
          ``self._cuda_class``.
        - On Apple Silicon / MPS systems, imports ``self._mps_module`` and
          instantiates ``self._mps_class``.
        - Sets ``self._impl`` to the instantiated concrete backend.
        - All protocol methods forward to ``self._impl`` via ``__getattr__``.
        """
        import importlib

        super().__init__(config)
        self.device_info = detect_backend(config.extra.get("backend", "auto"))

        if self.device_info.backend == "cuda":
            mod = importlib.import_module(self._cuda_module)
            impl_cls = getattr(mod, self._cuda_class)
            logger.info(
                "%s dispatcher: selected %s", self.display_name, self._cuda_class,
            )
        else:
            mod = importlib.import_module(self._mps_module)
            impl_cls = getattr(mod, self._mps_class)
            logger.info(
                "%s dispatcher: selected %s", self.display_name, self._mps_class,
            )

        self._impl: InferenceBackend = impl_cls(config)
        self.tokenizer = None
        self.model = None
        self.devices = []

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the platform-specific implementation.

        Parameters
        ----------
        name : str
            Attribute name to look up.

        Returns
        -------
        Any
            The attribute value from ``self._impl`` if it exists there.

        Raises
        ------
        AttributeError
            If ``name`` is ``_impl`` itself (prevents infinite recursion during
            initialization) or if the attribute is not found on ``self._impl``
            and not defined on this dispatcher.

        Notes
        -----
        This enables transparent forwarding of all ``InferenceBackend`` protocol
        methods and properties to the concrete CUDA or MPS implementation.
        """
        if name == "_impl":
            raise AttributeError()
        if hasattr(self, "_impl"):
            return getattr(self._impl, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, delegating to the implementation when appropriate.

        Parameters
        ----------
        name : str
            Attribute name to set.
        value : Any
            Value to assign.

        Notes
        -----
        Attributes in the ``_delegated`` tuple (config, device_info, tokenizer,
        model, devices, _loaded, _impl) are always set on the dispatcher itself.
        All other attributes are forwarded to ``self._impl`` if it has a matching
        attribute, or set on the dispatcher otherwise.
        """
        _delegated = ("_impl", "config", "device_info", "tokenizer", "model", "devices", "_loaded")
        if name in _delegated:
            super().__setattr__(name, value)
        elif hasattr(self, "_impl") and hasattr(self._impl, name):
            setattr(self._impl, name, value)
        else:
            super().__setattr__(name, value)

    def load(self) -> None:
        """Load the delegated implementation and sync local references.

        Calls ``self._impl.load()``, then copies ``tokenizer``, ``model``,
        ``devices``, and ``_loaded`` from the implementation to the dispatcher
        so that direct attribute access on the dispatcher returns correct values.

        Raises
        ------
        OSError
            If the wrapped implementation's ``load()`` fails (e.g., model not found).
        """
        self._impl.load()
        self.tokenizer = self._impl.tokenizer
        self.model = self._impl.model
        self.devices = self._impl.devices
        self._loaded = self._impl.is_loaded()

    def warmup(self, batches: int = 10) -> None:
        """Run warm-up on the delegated implementation.

        Parameters
        ----------
        batches : int
            Number of warm-up forward passes (default 10). Forwarded to
            ``self._impl.warmup()``.
        """
        self._impl.warmup(batches)

    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        """Translate a batch by delegating to the platform-specific implementation.

        Parameters
        ----------
        batch : PipelineBatch
            Pre-tokenised batch from the data pipeline.

        Returns
        -------
        BatchGenerationOutput
            Aggregated generation output from the delegated implementation.
        """
        return self._impl.translate_batch(batch)

    def is_loaded(self) -> bool:
        """Check loaded state on the delegated implementation.

        Returns
        -------
        bool
            True if the wrapped ``self._impl`` reports that it is loaded and ready.
        """
        return self._impl.is_loaded()

    def close(self) -> None:
        """Close the delegated implementation and release resources.

        Calls ``self._impl.close()`` to free GPU memory, destroy CUDA graphs,
        and release any other resources. Sets ``self._loaded = False`` on the
        dispatcher.
        """
        self._impl.close()
        self._loaded = False

    def encode_source(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Produce encoder hidden states via the delegated implementation.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenized source input with shape [batch, src_len].
        attention_mask : torch.Tensor
            Attention mask with shape [batch, src_len].

        Returns
        -------
        torch.Tensor
            Encoder hidden states of shape [batch, src_len, hidden_size].

        Raises
        ------
        NotImplementedError
            If the wrapped implementation does not support forward encoding.
        """
        return self._impl.encode_source(input_ids, attention_mask)
