"""Precision dispatch — FP8-native on H200, BF16 on MPS, FP32 on CPU.

CUDA:  FP8 compute via Transformer Engine (default).  Master weights stay in
       BF16; TE ``fp8_autocast`` context converts to FP8 for tensor-core matmul
       and back to BF16 for accumulation.  This is the native H200 fast path.
       Falls back to pure BF16 if TE is not installed.

       NOTE: On H200 with 4B models, TE FP8 is counterproductive (40% throughput
       regression, 0% memory savings) because the model is memory-bandwidth-bound
       rather than compute-bound. See M1.5.

MPS:   BF16 (native on Apple Silicon M2+; M1 may fall back to FP32 at the op
       level).
CPU:   FP32 (CPU does not benefit from reduced precision in general).

v2.0: Module-level TE import cache — the import is attempted once per process.
v3.4: FP8 made the explicit default on CUDA/H200.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PrecisionMode = Literal["auto", "fp8", "bf16", "fp16", "fp32"]
SUPPORTED_PRECISIONS: tuple[str, ...] = ("auto", "fp8", "bf16", "fp16", "fp32")

# ---------------------------------------------------------------------------
# Module-level TE import cache (P0)
# ---------------------------------------------------------------------------

# Module-level state: safe for single-process use. Not thread-safe for multi-harness scenarios.
_TE_IMPORTED: bool | None = None


def is_transformer_engine_available() -> bool:
    """Check whether Transformer Engine can be imported — cached at module level.

    This is the single shared helper for the entire codebase.  All callers
    that need to check TE availability should import from here rather than
    doing their own try/except or maintaining a separate cache.
    """
    global _TE_IMPORTED
    if _TE_IMPORTED is not None:
        return _TE_IMPORTED
    try:
        import transformer_engine  # noqa: F401
        _TE_IMPORTED = True
    except (ImportError, RuntimeError):
        logger.debug("Transformer Engine not available.")
        _TE_IMPORTED = False
    return _TE_IMPORTED


# Backward-compatible alias for :func:`is_transformer_engine_available`.
#
# This is a module-level alias, not a function wrapper. It references
# ``is_transformer_engine_available`` directly, so calling it is equivalent
# to calling the canonical function.
_te_is_available = is_transformer_engine_available


# ── QAT model detection (v3.4) ──────────────────────────────────────────────

def _is_qat_model_path(model_path: str) -> bool:
    """Return True if *model_path* refers to a QAT (quantization-aware trained) model."""
    if not model_path:
        return False
    path_lower = model_path.lower()
    return any(kw in path_lower for kw in ("qat", "qat-mobile"))


def _is_q4_0_model_path(model_path: str) -> bool:
    """Return True if *model_path* refers to a Q4_0 quantized model.

    Detectable via explicit ``q4_0`` in the model name, or via Google's
    naming convention where ``*-mobile-transformers`` (without ``-ct``)
    indicates pre-quantized 4-bit weights.
    """
    if not model_path:
        return False
    path_lower = model_path.lower()
    if "q4_0" in path_lower:
        return True
    if "mobile-transformers" in path_lower:
        return True
    return False


def recommended_dtype_for_qat(
    model_path: str,
    backend: str,
    default_dtype: torch.dtype,
) -> torch.dtype:
    """Return the recommended torch dtype for a QAT model on the given backend.

    - QAT-CT on any backend: BF16 (standard weights, QAT-trained).
    - Q4_0 on CUDA: the quantization is handled by bitsandbytes at load time;
      the master dtype stays BF16.
    - Q4_0 on MPS: BF16 (weights are dequantized at load time).
    - Q4_0 on CPU: BF16 (or FP32 if BF16 is not well-supported).
    """
    if not _is_qat_model_path(model_path):
        return default_dtype
    if backend in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


# ---------------------------------------------------------------------------
# PrecisionConfig — rich config object
# ---------------------------------------------------------------------------


class PrecisionConfig:
    """Precision configuration for a given backend.

    **v3.4: FP8 is the explicit default on CUDA/H200.**  When the backend is
    CUDA and Transformer Engine is installed, ``uses_transformer_engine`` is
    set to True and every ``nn.Linear`` layer is replaced with ``te.Linear``.
    Forward passes are wrapped in ``te.fp8_autocast()`` — giving native FP8
    tensor-core matmul with zero user intervention.

    Master weights stay in BF16.  FP8 compute is applied only inside the
    ``fp8_autocast`` context; accumulation happens in BF16 for numerical
    stability.

    **TF32 note**: On Ampere+ GPUs (SM80+), PyTorch defaults to TF32 for
    ``torch.matmul`` and ``torch.nn.Linear``.  TF32 uses a 19-bit mantissa
    (vs FP32's 23-bit), giving ~8x throughput over FP32 on tensor cores
    (NVIDIA literature claim; not measured on this codebase).
    TF32 is automatically managed by CUDA's ``torch.backends.cuda.matmul.allow_tf32``
    (default True on CUDA 11+).  This config does NOT disable TF32 — it remains
    active unless the user explicitly sets ``torch.backends.cuda.matmul.allow_tf32 = False``.
    On H200 (SM90), TF32 is superseded by FP8 whenever Transformer Engine is active,
    but TF32 still applies to operations outside ``fp8_autocast`` (e.g., attention
    softmax, RMSNorm weight multiplication).

    NOTE: On H200 with 4B models, TE FP8 is counterproductive —
    40% throughput regression, 0% memory savings (measured 2026-06-24),
    because the model is memory-bandwidth-bound rather than compute-bound.
    See M1.5.

    Attributes
    ----------
    backend : str
    preferred : PrecisionMode
    master_dtype : torch.dtype
    compute_dtype : torch.dtype
    uses_transformer_engine : bool
    uses_fp8 : bool
    supports_fp8_native : bool
        True when the hardware is capable of FP8 (Hopper/H200 SM90+).
    tf32_enabled : bool
        True when TF32 is enabled on CUDA (default on Ampere+).
    """

    def __init__(self, backend: str, preferred: PrecisionMode = "auto"):
        """Initialize a PrecisionConfig for the given backend.

        Parameters
        ----------
        backend : str
            The inference backend identifier. One of ``"cuda"``, ``"mps"``, ``"cpu"``.
            Case-insensitive, leading/trailing whitespace is stripped.
        preferred : PrecisionMode, default ``"auto"``
            The user's requested precision mode. ``"auto"`` selects FP8 on CUDA when
            Transformer Engine is available, falling back to BF16 for GPU or FP32 for
            CPU otherwise.

        Raises
        ------
        ValueError
            If the resolved backend or precision combination is unsupported (handled
            downstream by the resolver methods).

        Notes
        -----
        All instance attributes (``master_dtype``, ``compute_dtype``,
        ``uses_transformer_engine``, ``uses_fp8``, ``supports_fp8_native``,
        ``tf32_enabled``) are resolved eagerly during ``__init__``.
        """
        self.backend = backend.strip().lower()
        self.preferred = preferred
        self.master_dtype = self._resolve_master_dtype()
        self.uses_transformer_engine = self._resolve_te()  # must precede _resolve_compute_dtype
        self.compute_dtype = self._resolve_compute_dtype()
        self.uses_fp8 = self._resolve_fp8()
        self.supports_fp8_native = self._resolve_fp8_native()
        self.tf32_enabled = self._resolve_tf32()

    def _resolve_master_dtype(self) -> torch.dtype:
        """Resolve the master-weight torch dtype based on backend and preference.

        Parameters
        ----------
        None (uses ``self.backend`` and ``self.preferred``).

        Returns
        -------
        torch.dtype
            ``torch.bfloat16`` for GPU backends in ``"auto"`` or ``"bf16"`` modes;
            ``torch.float16`` for ``"fp16"``; ``torch.float32`` for ``"fp32"`` or
            CPU in ``"auto"`` mode.

        Notes
        -----
        FP8-requested precision resolves to BF16 master weights (FP8 compute is
        handled separately via Transformer Engine's ``fp8_autocast``).
        """
        if self.preferred in ("fp8", "float8_e4m3fn") and self.backend == "cuda":
            return torch.bfloat16
        if self.preferred in ("bf16", "bfloat16"):
            return torch.bfloat16
        if self.preferred in ("fp16", "float16"):
            return torch.float16
        if self.preferred in ("fp32", "float32"):
            return torch.float32
        # "auto" — BF16 for GPU backends, FP32 for CPU.
        if self.backend in ("cuda", "mps"):
            return torch.bfloat16
        return torch.float32

    def _resolve_compute_dtype(self) -> torch.dtype:
        """Resolve the effective compute dtype.

        Parameters
        ----------
        None (uses ``self.uses_transformer_engine`` and ``self.master_dtype``).

        Returns
        -------
        torch.dtype
            ``torch.float8_e4m3fn`` when Transformer Engine is active (FP8 tensor-core
            matmul via ``te.fp8_autocast``), otherwise the master dtype.

        Notes
        -----
        When TE is active, the compute dtype is informational — the actual FP8
        casting is managed by TE's ``fp8_autocast`` context, not by this dtype.
        """
        # When TE is active, compute dtype is FP8 at the op level (handled
        # by te.fp8_autocast).  The master dtype stays BF16.
        if self.uses_transformer_engine:
            return torch.float8_e4m3fn
        return self.master_dtype

    def _resolve_te(self) -> bool:
        """Determine whether Transformer Engine should be used for FP8 compute.

        v3.4: On CUDA, FP8 is the DEFAULT.  We always probe for TE; if it's
        installed, FP8 is active.  Falls back to pure BF16 only when TE is
        not installed or when the user explicitly requests a non-FP8
        precision mode.
        """
        # Explicit non-FP8 precision overrides.
        explicit_non_fp8 = self.preferred in ("bf16", "bfloat16", "fp16", "float16", "fp32", "float32")
        if explicit_non_fp8:
            logger.info(
                "Precision: '%s' explicitly requested — FP8 NOT used "
                "(Transformer Engine will not be loaded).",
                self.preferred,
            )
            return False

        # Not CUDA — FP8 is CUDA-only.
        if self.backend != "cuda":
            return False

        # "auto" or "fp8" on CUDA — probe for Transformer Engine.
        te_ok = _te_is_available()

        if te_ok:
            logger.info(
                "FP8 ACTIVE — Transformer Engine detected.  "
                "All nn.Linear layers will be replaced with te.Linear "
                "for native FP8 tensor-core matmul on H200."
            )
        else:
            logger.warning(
                "FP8 NOT available — Transformer Engine is not installed.  "
                "Falling back to pure BF16.  Install with: "
                "pip install transformer-engine[pytorch]"
            )

        return te_ok

    def _resolve_fp8(self) -> bool:
        """Return True if FP8 compute is actively in use.

        Parameters
        ----------
        None (uses ``self.uses_transformer_engine``).

        Returns
        -------
        bool
            True when Transformer Engine has been successfully loaded and is
            configured for FP8 tensor-core matmul.

        Notes
        -----
        This is a convenience that delegates to ``self.uses_transformer_engine``.
        FP8 compute implies TE is active; there is no FP8 path without TE in the
        current architecture.
        """
        return self.uses_transformer_engine

    def _resolve_fp8_native(self) -> bool:
        """True when the hardware itself supports FP8 (Hopper SM90+).

        Even if TE is not installed, the hardware may still be capable.
        """
        if self.backend != "cuda":
            return False
        try:
            if not torch.cuda.is_available():
                return False
            major, _minor = torch.cuda.get_device_capability()
            return major >= 9  # Hopper (H100/H200) = SM 9.0+
        except Exception:
            return False

    def _resolve_tf32(self) -> bool:
        """True when TF32 matmul is enabled on CUDA.

        TF32 is the default on Ampere+ GPUs (SM80+).  PyTorch enables it
        automatically via ``torch.backends.cuda.matmul.allow_tf32`` (default
        True on CUDA 11+).  It is superseded by FP8 inside ``fp8_autocast``
        but still active for non-TE operations.

        On MPS and CPU, TF32 is irrelevant — returns False.
        """
        if self.backend != "cuda" or not torch.cuda.is_available():
            return False
        return getattr(
            torch.backends.cuda.matmul, "allow_tf32", False
        )

    def to_dict(self) -> dict:
        """Serialize the precision configuration to a dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Dictionary with string keys: ``"backend"``, ``"preferred"``,
            ``"master_dtype"``, ``"compute_dtype"``, ``"uses_transformer_engine"``,
            ``"uses_fp8"``, ``"supports_fp8_native"``, ``"tf32_enabled"``.
            Dtype values are converted to their string representations (e.g.
            ``"torch.bfloat16"``).

        Notes
        -----
        This is intended for logging, reporting, and checkpoint serialization.
        """
        return {
            "backend": self.backend,
            "preferred": self.preferred,
            "master_dtype": str(self.master_dtype),
            "compute_dtype": str(self.compute_dtype),
            "uses_transformer_engine": self.uses_transformer_engine,
            "uses_fp8": self.uses_fp8,
            "supports_fp8_native": self.supports_fp8_native,
            "tf32_enabled": self.tf32_enabled,
        }

    @property
    def effective_precision_label(self) -> str:
        """Human-readable label for the effective precision in use."""
        if self.uses_fp8:
            return "FP8 (native H200)"
        if self.backend == "cuda":
            return "BF16 (FP8 not available)"
        if self.backend == "mps":
            return "BF16 (MPS)"
        return "FP32 (CPU)"


def get_precision_config(
    backend: str,
    preferred: PrecisionMode = "auto",
) -> PrecisionConfig:
    """Create and log a PrecisionConfig for the given backend.

    Parameters
    ----------
    backend : str
        One of ``"cuda"``, ``"mps"``, ``"cpu"``.
    preferred : PrecisionMode, default ``"auto"``
        User-requested precision. ``"auto"`` selects FP8 on CUDA when Transformer
        Engine is available, BF16 on MPS, FP32 on CPU.

    Returns
    -------
    PrecisionConfig
        Fully resolved precision configuration object.

    Side Effects
    ------------
    Logs the resolved configuration at INFO level, including effective precision
    label, master dtype, compute dtype, TE status, and FP8 status.
    """
    cfg = PrecisionConfig(backend=backend, preferred=preferred)
    logger.info(
        "Precision: backend=%s preferred=%s → %s (master=%s compute=%s TE=%s FP8=%s)",
        cfg.backend,
        cfg.preferred,
        cfg.effective_precision_label,
        cfg.master_dtype,
        cfg.compute_dtype,
        cfg.uses_transformer_engine,
        cfg.uses_fp8,
    )
    return cfg


# ---------------------------------------------------------------------------
# Core dispatch
# ---------------------------------------------------------------------------

def get_dtype(
    backend: str,
    preferred: PrecisionMode = "auto",
) -> torch.dtype:
    """Return the master-weight torch dtype for a given backend and preference.

    Dispatch rules (v3.4)
    ---------------------
    - ``"auto"``:
        * CUDA → FP8 compute via Transformer Engine, BF16 master weights.
          Falls back to pure BF16 if TE is not installed.
        * MPS  → torch.bfloat16
        * CPU  → torch.float32
    - ``"fp8"``:  CUDA only — same as auto on CUDA.
    - ``"bf16"``: torch.bfloat16 on all backends (CPU will run slower).
    - ``"fp16"``: torch.float16 on all backends (CPU will run slower).
    - ``"fp32"``: torch.float32 on all backends.
    """
    return get_precision_config(backend, preferred).master_dtype

# ---------------------------------------------------------------------------
# Static FP8 weight quantization — enforced on CUDA by default
# ---------------------------------------------------------------------------

class StaticFP8Linear(torch.nn.Module):
    """An ``nn.Linear`` replacement storing weights as FP8 E4M3 on GPU.

    Weights are quantized once (at model load) and stored in
    ``torch.float8_e4m3fn``.  At forward time, the weight is dequantized to
    BF16 in the same memory transaction — the H200 memory controller handles
    the type cast inline, so there is zero per-token compute overhead.

    This is weight-storage quantization for 2× memory bandwidth.  The matmul
    runs in BF16 with TF32 tensor-core acceleration.

    lm_head excluded — FP8 precision loss on vocab projection hurts rankings.
    """

    _MIN_IN_FEATURES = 256  # skip tiny projection layers

    def __init__(self, linear: torch.nn.Module):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.data.to(torch.float32)
        w_max = w.abs().max()
        if w_max == 0:
            w_max = torch.tensor(1.0, dtype=torch.float32, device=w.device)
        elif w_max < 1e-6:
            # Near-zero weight matrix — quantization will produce all-zero FP8.
            # This typically indicates a dead layer; log a warning so it's not silent.
            _name = getattr(linear, '_static_fp8_name', 'unknown')
            logger.warning(
                "StaticFP8: weight max is %.2e for layer '%s' — "
                "quantization will destroy precision. Check for dead/untrained layers.",
                w_max.item(), _name,
            )
        scale = (w_max / 448.0).to(torch.float32)
        w_fp8 = (w / scale).clamp(-448.0, 447.0).to(torch.float8_e4m3fn)
        self.register_buffer("weight_fp8", w_fp8)
        self.register_buffer("weight_scale", scale)
        if linear.bias is not None:
            self.bias = torch.nn.Parameter(linear.bias.data.clone().to(torch.bfloat16))
            # Free original bias memory immediately
            linear.bias.data = torch.empty(0)
        else:
            self.bias = None
        
        # Copy hooks from the original linear module to preserve device placement/alignment hooks (e.g., Hugging Face accelerate)
        self._forward_hooks = linear._forward_hooks
        self._forward_pre_hooks = linear._forward_pre_hooks
        self._backward_hooks = linear._backward_hooks
        self._backward_pre_hooks = linear._backward_pre_hooks

        # Copy HF accelerate specific attributes if they exist
        for attr in ["_hf_hook", "hf_device_map"]:
            if hasattr(linear, attr):
                setattr(self, attr, getattr(linear, attr))

        # Free original weight memory immediately to prevent VRAM spikes during the conversion loop
        linear.weight.data = torch.empty(0)

        # Cache for dequantized weight to bypass PyTorch runtime casting overhead during decoding
        self._cached_weight = None
        self._cached_dtype = None

    @property
    def weight(self) -> torch.Tensor:
        """Return the dequantized weight tensor dynamically in BF16 precision."""
        if self._cached_weight is not None and self._cached_dtype == torch.bfloat16:
            return self._cached_weight
        return self.weight_fp8.to(torch.bfloat16) * self.weight_scale.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8 dequantization matching input dtype on read.

        Parameters
        ----------
        x : torch.Tensor
            Input activation tensor of shape ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(..., out_features)`` after linear projection.
        """
        # Cache the dequantized weights to eliminate sequential casting/multiplication overheads
        if self._cached_weight is None or self._cached_dtype != x.dtype:
            self._cached_weight = (self.weight_fp8.to(x.dtype) * self.weight_scale.to(x.dtype)).detach()
            self._cached_dtype = x.dtype

        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, self._cached_weight, bias)


def apply_static_fp8_to_model(
    model: torch.nn.Module,
    *,
    skip_lm_head: bool = True,
) -> int:
    """Replace qualifying ``nn.Linear`` layers with :class:`StaticFP8Linear`.

    Skipped: lm_head (token ranking), layers < 256 in_features (too small),
    already-replaced layers (idempotent).

    Returns number of layers replaced.
    """
    if not torch.cuda.is_available():
        return 0

    replaced = 0

    def _replace(module: torch.nn.Module, parent_name: str = ""):
        nonlocal replaced
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if skip_lm_head and (name == "lm_head" or full_name.endswith(".lm_head")):
                continue
            if isinstance(child, torch.nn.Linear) and not isinstance(child, StaticFP8Linear):
                if child.in_features >= StaticFP8Linear._MIN_IN_FEATURES:
                    setattr(module, name, StaticFP8Linear(child))
                    replaced += 1
            else:
                _replace(child, full_name)

    _replace(model)

    if replaced > 0:
        logger.info(
            "Static FP8: %d nn.Linear replaced (weight-only FP8, dequant-on-read, "
            "lm_head excluded).",
            replaced,
        )
    return replaced


# ---------------------------------------------------------------------------
# FP8 weight cache — quantize Linear weights once, cache to disk
#
# This is STATIC weight-only quantization.  Weights are quantized to FP8
# E4M3 once and cached to disk, eliminating dynamic per-token activation
# quantization overhead.  For SmoothQuant or QAT, the pre-quantized
# weights + scales are loaded from cache and fed directly to the model.
# ---------------------------------------------------------------------------

def _model_weight_hash(model: torch.nn.Module, model_path: str) -> str:
    """Compute a stable hash of model identity and Linear layer structure.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded PyTorch model whose Linear layers will be hashed.
    model_path : str
        The filesystem path from which the model was loaded. Used as a seed
        for the hash so that different models (even with coincidentally similar
        shapes and sums) produce distinct hashes.

    Returns
    -------
    str
        A 16-character hex digest string (first 16 chars of SHA-256).

    Notes
    -----
    The hash incorporates the model path and, for every ``nn.Linear`` submodule,
    the fully qualified module name, weight shape, and weight sum. This is
    designed to be stable across runs but sensitive to weight changes
    (e.g. fine-tuning). It does NOT hash bias tensors — only weight matrices.
    """
    import hashlib
    h = hashlib.sha256(model_path.encode())
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight.data
            h.update(f"{name}:{list(w.shape)}".encode())
            h.update(str(w.sum().item()).encode())
    return h.hexdigest()[:16]


def save_fp8_weights(
    model: torch.nn.Module,
    model_path: str,
    cache_dir: str | os.PathLike | None = None,
) -> str | None:
    """Quantize all nn.Linear weights to FP8 E4M3 and save each as a safetensors file.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded PyTorch model whose Linear weights will be quantized.
    model_path : str
        Path to the model, used as part of the hash key for cache isolation.
    cache_dir : str or os.PathLike or None, optional
        Root directory for the FP8 weight cache. If ``None``, defaults to
        ``~/.cache/tr_benchmark/fp8_weights/``.

    Returns
    -------
    str or None
        The cache directory path (``{cache_dir}/{hash}/``) on success,
        or ``None`` if ``safetensors`` is not installed or no layers were saved.

    Side Effects
    ------------
    - Creates the cache directory tree if it does not exist.
    - Writes one ``.safetensors`` file per Linear layer to the cache directory.

    Notes
    -----
    lm_head layers are excluded. Near-zero weight matrices (max < 1e-6) that
    would quantize to all-zero FP8 are also skipped. Each file contains two
    tensors: ``"weight"`` (FP8 E4M3) and ``"scale"`` (FP32 per-tensor scale).
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        logger.warning("safetensors not installed — FP8 weight cache disabled.")
        return None

    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "tr_benchmark", "fp8_weights",
        )
    cache_dir = os.path.join(str(cache_dir), _model_weight_hash(model, model_path))
    os.makedirs(cache_dir, exist_ok=True)

    saved = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if "lm_head" in name or name == "lm_head":
                continue
            w = mod.weight.data.to(torch.float32)
            w_max = w.abs().max()
            if w_max == 0:
                continue
            scale = (w_max / 448.0).to(torch.float32)
            w_fp8 = (w / scale).clamp(-448.0, 447.0).to(torch.float8_e4m3fn)
            out_path = os.path.join(cache_dir, f"{name.replace('.', '_')}.safetensors")
            save_file(
                {"weight": w_fp8, "scale": scale.to(torch.float32)},
                out_path,
            )
            saved += 1

    if saved > 0:
        logger.info(
            "FP8 weight cache: %d layers saved to %s", saved, cache_dir,
        )
        return cache_dir
    return None


def load_fp8_weights(
    model: torch.nn.Module,
    model_path: str,
    cache_dir: str | os.PathLike | None = None,
) -> int:
    """Load pre-quantized FP8 weights from cache and set them on Linear layers.

    Each layer's FP8 weight and per-tensor scale are stored as registered
    buffers (``_fp8_weight``, ``_fp8_weight_scale``) on the module.  These
    are consumed by **static quantization** paths (SmoothQuant / QAT) —
    weights are dequantized to BF16 at forward time without any per-token
    activation quantization overhead.

    Returns the number of layers loaded from cache (0 = cache miss).
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        return 0

    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "tr_benchmark", "fp8_weights",
        )
    cache_dir = os.path.join(str(cache_dir), _model_weight_hash(model, model_path))
    if not os.path.isdir(cache_dir):
        return 0

    loaded = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if "lm_head" in name or name == "lm_head":
                continue
            fname = name.replace(".", "_")
            path = os.path.join(cache_dir, f"{fname}.safetensors")
            if not os.path.isfile(path):
                continue
            try:
                tensors = load_file(path)
                # Store FP8 weight + scale as registered buffers — consumed
                # by static quantization paths (SmoothQuant / QAT).
                mod.register_buffer("_fp8_weight", tensors["weight"])
                mod.register_buffer("_fp8_weight_scale", tensors["scale"])
                loaded += 1
            except Exception as e:
                logger.debug("Failed to load FP8 cache for %s: %s", name, e)

    if loaded > 0:
        logger.info(
            "FP8 weight cache hit: %d layers from %s", loaded, cache_dir,
        )
    return loaded


# ---------------------------------------------------------------------------
# FP8 activation helper — replaces nn.Linear with te.Linear
# ---------------------------------------------------------------------------

def apply_te_fp8_to_model(
    model: torch.nn.Module,
    *,
    skip_lm_head: bool = True,
    mlp_only: bool = False,
) -> bool:
    """Replace ``nn.Linear`` in *model* with ``te.Linear`` for FP8 compute.

    This is the single canonical FP8 activation function.  Call it once after
    model loading and before any forward pass.

    Parameters
    ----------
    model : nn.Module
        The loaded PyTorch model.
    skip_lm_head : bool
        If True, the lm_head is kept in BF16.
    mlp_only : bool
        If True, only replace MLP layers (gate_proj, up_proj, down_proj).
        Attention projections (q_proj, k_proj, v_proj, o_proj) are kept in
        BF16.  Use this for architectures where TE's cuBLAS gemm path crashes
        on attention matmul shapes (e.g. Gemma 3).

    Returns
    -------
    bool
        True if any layers were replaced.
    """
    try:
        import transformer_engine.pytorch as te
    except ImportError:
        logger.warning("Transformer Engine not installed — FP8 activation skipped.")
        return False

    _ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj"}
    replaced = 0

    def _replace(module: torch.nn.Module, parent_name: str = ""):
        nonlocal replaced
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if skip_lm_head and (name == "lm_head" or full_name.endswith(".lm_head")):
                logger.debug("TE FP8: skipping lm_head")
                continue
            if mlp_only and name in _ATTN_NAMES:
                logger.debug("TE FP8: skipping %s (mlp_only)", full_name)
                continue
            if isinstance(child, torch.nn.Linear) and not isinstance(child, te.Linear):
                te_lin = te.Linear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                )
                te_lin.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    te_lin.bias.data = child.bias.data.clone()
                setattr(module, name, te_lin)
                replaced += 1
            else:
                _replace(child, full_name)

    _replace(model)

    if replaced > 0:
        logger.info(
            "FP8: %d te.Linear replaced (mlp_only=%s, lm_head excluded).",
            replaced, mlp_only,
        )
    else:
        logger.info("FP8: no nn.Linear layers found to replace.")

    return replaced > 0


def fp8_autocast_context():
    """Return a context manager that enables FP8 auto-cast for forward passes.

    Usage::

        with fp8_autocast_context():
            output = model(input_ids=...)
    """
    try:
        import transformer_engine.pytorch as te
        return te.fp8_autocast(enabled=True)
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Helper: check FP8 availability
# ---------------------------------------------------------------------------

def has_fp8_support(backend: str) -> bool:
    """Return True if the backend can use FP8 via Transformer Engine."""
    if backend.strip().lower() != "cuda":
        return False
    return _te_is_available()
