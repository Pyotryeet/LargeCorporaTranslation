"""Precision dispatch — FP8-native on H200, BF16 on MPS, FP32 on CPU.

CUDA:  FP8 compute via Transformer Engine (default).  Master weights stay in
       BF16; TE ``fp8_autocast`` context converts to FP8 for tensor-core matmul
       and back to BF16 for accumulation.  This is the native H200 fast path.
       Falls back to pure BF16 if TE is not installed.
MPS:   BF16 (native on Apple Silicon M2+; M1 may fall back to FP32 at the op
       level).
CPU:   FP32 (CPU does not benefit from reduced precision in general).

v2.0: Module-level TE import cache — the import is attempted once per process.
v3.4: FP8 made the explicit default on CUDA/H200.
"""

from __future__ import annotations

import logging
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


# Backward-compatible private alias.
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
    (vs FP32's 23-bit), giving ~8x throughput over FP32 on tensor cores.
    TF32 is automatically managed by CUDA's ``torch.backends.cuda.matmul.allow_tf32``
    (default True on CUDA 11+).  This config does NOT disable TF32 — it remains
    active unless the user explicitly sets ``torch.backends.cuda.matmul.allow_tf32 = False``.
    On H200 (SM90), TF32 is superseded by FP8 whenever Transformer Engine is active,
    but TF32 still applies to operations outside ``fp8_autocast`` (e.g., attention
    softmax, RMSNorm weight multiplication).

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
        self.backend = backend.strip().lower()
        self.preferred = preferred
        self.master_dtype = self._resolve_master_dtype()
        self.uses_transformer_engine = self._resolve_te()  # must precede _resolve_compute_dtype
        self.compute_dtype = self._resolve_compute_dtype()
        self.uses_fp8 = self._resolve_fp8()
        self.supports_fp8_native = self._resolve_fp8_native()
        self.tf32_enabled = self._resolve_tf32()

    def _resolve_master_dtype(self) -> torch.dtype:
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
    """Create a PrecisionConfig for the given backend.

    Parameters
    ----------
    backend : str
        One of ``"cuda"``, ``"mps"``, ``"cpu"``.
    preferred : PrecisionMode
        User's requested precision.  ``"auto"`` (default) selects FP8 on
        CUDA when Transformer Engine is available, BF16 otherwise.

    Returns
    -------
    PrecisionConfig
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
# FP8 activation helper — replaces nn.Linear with te.Linear
# ---------------------------------------------------------------------------

def apply_te_fp8_to_model(model: torch.nn.Module, *, skip_lm_head: bool = True) -> bool:
    """Replace every ``nn.Linear`` in *model* with ``te.Linear`` for FP8 compute.

    This is the single canonical FP8 activation function.  Call it once after
    model loading and before any forward pass.

    Parameters
    ----------
    model : nn.Module
        The loaded PyTorch model.
    skip_lm_head : bool
        If True (default), the lm_head (final vocabulary projection) is kept
        in BF16 — FP8 precision loss on the 262K-dimensional vocab projection
        directly corrupts token probability rankings.

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

    replaced = 0

    def _replace(module: torch.nn.Module, parent_name: str = ""):
        nonlocal replaced
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            # Skip lm_head — FP8 on vocab projection hurts token selection.
            if skip_lm_head and (name == "lm_head" or full_name.endswith(".lm_head")):
                logger.debug("TE FP8: skipping lm_head (kept in BF16)")
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
            "FP8: %d nn.Linear layers replaced with te.Linear "
            "(lm_head excluded — kept in BF16).  "
            "Forward passes must be wrapped in te.fp8_autocast().",
            replaced,
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
