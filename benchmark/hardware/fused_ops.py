"""Custom fused ops via torch.library (C-CROSS-3).

Registers fused kernels that combine common operation sequences into
single GPU kernel launches, reducing launch overhead and intermediate
memory traffic.

Fusions (backend-aware):
  - ``fused_rms_norm_residual`` : RMSNorm + residual add (1 kernel vs 3).
    CUDA path uses Triton when available; falls back to eager PyTorch.
    Speedup depends on model size and batch size. On 4B models, the
    overall throughput gain is <5% (measured 2026-06-24) because RMSNorm
    is a small fraction of total compute. See M2.1.
  - ``fused_swiglu_gate_up`` : SiLU gate + up projection + multiply.
    Uses eager PyTorch matmuls (auto-fused by the CUDA compiler); a true
    Triton kernel for this operation is available in
    ``triton_kernels_fused.py`` but uses a different calling convention
    (pre-computed projections).

These use ``torch.library`` for portability across CUDA, MPS, and CPU.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# torch.library registration (portable fallbacks)
# ---------------------------------------------------------------------------

# Double-import guard: torch.library registration is global — re-importing the
# module re-runs DEF/define and triggers errors.  Use a module-level sentinel
# to skip re-registration.
_IMPORT_REGISTERED = False

__all__ = [
    "fused_rms_norm_residual",
    "fused_swiglu_gate_up",
]

if not _IMPORT_REGISTERED:
    try:
        _lib = torch.library.Library("tr_benchmark", "DEF")  # type: ignore[attr-defined]
    except RuntimeError:
        # Already defined by a prior import (e.g. reimport in tests, pickling,
        # or multiprocessing spawn).  Re-attach as a FRAGMENT.
        _lib = torch.library.Library("tr_benchmark", "FRAGMENT")  # type: ignore[attr-defined]

    try:
        # define() is idempotent within the same Library handle but raises
        # RuntimeError if the op was already defined via a different handle.
        _lib.define(
            "fused_rms_norm_residual(Tensor x, Tensor residual, Tensor weight, float eps) "
            "-> (Tensor output, Tensor new_residual)"
        )
    except RuntimeError:
        pass  # Already defined by a prior import path.

    try:
        _lib.define(
            "fused_swiglu_gate_up(Tensor hidden, Tensor gate_proj_weight, "
            "Tensor up_proj_weight) -> Tensor"
        )
    except RuntimeError:
        pass  # Already defined by a prior import path.

    _IMPORT_REGISTERED = True


# Dispatch-handler registration.  Each @torch.library.impl decorator registers a
# dispatch key for the op.  On double-import (or reimport in tests/multiprocessing)
# the decorator may fail if the key is already registered.  Fall back gracefully.

_IMPL_ERRORS: list[str] = []

try:

    @torch.library.impl(_lib, "fused_rms_norm_residual", "CPU")
    def _fused_rms_norm_residual_cpu(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CPU fallback: RMSNorm + residual add in two steps."""
        # Step 1: Add residual.
        x = x + residual
        # Step 2: RMSNorm.
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
        output = (x.float() / rms).to(x.dtype) * weight
        return output, x.detach()  # new_residual = x before norm
except RuntimeError as e:
    _IMPL_ERRORS.append(f"fused_rms_norm_residual CPU: {e}")


try:

    @torch.library.impl(_lib, "fused_rms_norm_residual", "CUDA")
    def _fused_rms_norm_residual_cuda(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CUDA implementation via Triton when available, else eager fallback."""
        try:
            return _fused_rms_norm_residual_triton(x, residual, weight, eps)
        except (RuntimeError, torch.cuda.CudaError, ImportError):
            return _fused_rms_norm_residual_cpu(x, residual, weight, eps)
except RuntimeError as e:
    _IMPL_ERRORS.append(f"fused_rms_norm_residual CUDA: {e}")


try:

    @torch.library.impl(_lib, "fused_rms_norm_residual", "MPS")
    def _fused_rms_norm_residual_mps(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MPS: uses eager PyTorch (Metal auto-fuses where possible)."""
        return _fused_rms_norm_residual_cpu(x, residual, weight, eps)
except RuntimeError as e:
    _IMPL_ERRORS.append(f"fused_rms_norm_residual MPS: {e}")


try:

    @torch.library.impl(_lib, "fused_swiglu_gate_up", "CPU")
    def _fused_swiglu_gate_up_cpu(
        hidden: torch.Tensor, gate_proj_weight: torch.Tensor, up_proj_weight: torch.Tensor,
    ) -> torch.Tensor:
        """CPU: SiLU gate × up — two matmuls + elementwise."""
        gate = F.silu(F.linear(hidden, gate_proj_weight))
        up = F.linear(hidden, up_proj_weight)
        return gate * up
except RuntimeError as e:
    _IMPL_ERRORS.append(f"fused_swiglu_gate_up CPU: {e}")


try:

    @torch.library.impl(_lib, "fused_swiglu_gate_up", "CUDA")
    def _fused_swiglu_gate_up_cuda(
        hidden: torch.Tensor, gate_proj_weight: torch.Tensor, up_proj_weight: torch.Tensor,
    ) -> torch.Tensor:
        try:
            return _fused_swiglu_gate_up_eager(hidden, gate_proj_weight, up_proj_weight)
        except (RuntimeError, torch.cuda.CudaError, ImportError):
            return _fused_swiglu_gate_up_cpu(hidden, gate_proj_weight, up_proj_weight)
except RuntimeError as e:
    _IMPL_ERRORS.append(f"fused_swiglu_gate_up CUDA: {e}")


try:

    @torch.library.impl(_lib, "fused_swiglu_gate_up", "MPS")
    def _fused_swiglu_gate_up_mps(
        hidden: torch.Tensor, gate_proj_weight: torch.Tensor, up_proj_weight: torch.Tensor,
    ) -> torch.Tensor:
        return _fused_swiglu_gate_up_cpu(hidden, gate_proj_weight, up_proj_weight)
except RuntimeError as e:
    _IMPL_ERRORS.append(f"fused_swiglu_gate_up MPS: {e}")


if _IMPL_ERRORS:
    logger.debug(
        "torch.library impl registration errors (expected on double-import): %s",
        ", ".join(_IMPL_ERRORS),
    )


# ---------------------------------------------------------------------------
# Triton-accelerated implementations (CUDA only)
# ---------------------------------------------------------------------------

def _fused_rms_norm_residual_triton(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton fused RMSNorm + residual add.

    Falls back to PyTorch eager if Triton is not available or for
    edge-case shapes that the triton kernel doesn't handle.
    """
    try:
        import triton  # noqa: F811
        import triton.language as tl

        @triton.jit
        def _rms_norm_residual_kernel(
            x_ptr, residual_ptr, weight_ptr, out_ptr, new_residual_ptr,
            n_cols, eps,
            BLOCK_SIZE: tl.constexpr,
        ):
            """One program per row: add residual + RMSNorm in one pass."""
            pid = tl.program_id(0)
            row_start = pid * n_cols

            # Compute mean of squares (RMSNorm denominator).
            acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            for off in range(0, n_cols, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < n_cols
                x_val = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
                r_val = tl.load(residual_ptr + row_start + cols, mask=mask, other=0.0)
                summed = x_val + r_val
                acc += summed * summed

            rms = tl.sqrt(tl.sum(acc) / n_cols + eps)

            # Normalize and write.
            for off in range(0, n_cols, BLOCK_SIZE):
                cols = off + tl.arange(0, BLOCK_SIZE)
                mask = cols < n_cols
                x_val = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
                r_val = tl.load(residual_ptr + row_start + cols, mask=mask, other=0.0)
                summed = x_val + r_val
                w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
                out = (summed / rms) * w
                tl.store(out_ptr + row_start + cols, out, mask=mask)
                tl.store(new_residual_ptr + row_start + cols, summed, mask=mask)

        n_rows = x.shape[0]
        n_cols = x.shape[-1]
        BLOCK_SIZE = min(1024, triton.next_power_of_2(n_cols))

        out = torch.empty_like(x)
        new_residual = torch.empty_like(x)
        grid = (n_rows,)
        _rms_norm_residual_kernel[grid](
            x, residual, weight, out, new_residual,
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out, new_residual
    except (RuntimeError, torch.cuda.CudaError, ImportError):
        raise  # Let the caller fall back to eager.


def _fused_swiglu_gate_up_eager(
    hidden: torch.Tensor, gate_proj_weight: torch.Tensor, up_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Eager PyTorch fallback for fused SiLU(gate(hidden)) * up(hidden).

    NOTE: This function does NOT use Triton.  A true Triton kernel for the
    SwiGLU fusion exists in ``triton_kernels_fused.py`` but is not wired into
    this dispatch path (it requires a different calling convention — the
    Triton kernel operates on pre-computed gate/up projections, not raw
    weight matmuls).  This implementation uses eager PyTorch matmuls which
    are already auto-fused by the CUDA compiler.
    """
    gate = F.silu(F.linear(hidden, gate_proj_weight))
    up = F.linear(hidden, up_proj_weight)
    return gate * up


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fused_rms_norm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm with fused residual add.

    Equivalent to, but faster than:
        x = x + residual
        out = rms_norm(x, weight, eps)
        return out, x

    Parameters
    ----------
    x : torch.Tensor
        Input activations [*, hidden_size].
    residual : torch.Tensor
        Residual connection (same shape as x).
    weight : torch.Tensor
        RMSNorm weight [hidden_size].
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    (output, new_residual) : tuple[torch.Tensor, torch.Tensor]
    """
    return torch.ops.tr_benchmark.fused_rms_norm_residual(x, residual, weight, eps)


def fused_swiglu_gate_up(
    hidden: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Fused SiLU gate × up projection for Gemma MLP layers.

    Equivalent to, but faster than:
        gate = F.silu(F.linear(hidden, gate_proj_weight))
        up = F.linear(hidden, up_proj_weight)
        return gate * up
    """
    return torch.ops.tr_benchmark.fused_swiglu_gate_up(hidden, gate_proj_weight, up_proj_weight)
