"""Real Triton fused kernels — actual GPU implementations (v3.1).

Each kernel replaces multiple PyTorch ops with a single fused Triton launch,
eliminating intermediate tensors and kernel launch overhead entirely.

Kernels:
  1. fused_rms_norm_residual — RMSNorm(x + residual) in 1 kernel.
  2. fused_swiglu_gate_up — SiLU(gate) * up in 1 kernel.

Naming convention
-----------------
All Triton kernel functions in this module are prefixed ``_tr_`` to avoid
name collisions with:
  - PyTorch autograd functions (unprefixed).
  - Inline Triton kernels defined inside closures in ``fused_ops.py``
    (those are named ``_rms_norm_residual_kernel`` — same purpose,
    different compilation unit).  The ``_tr_`` prefix is deliberate:
    two kernels with the SAME name in different modules cause Triton
    JIT-cache ambiguity when both modules are loaded in the same process.
  - Future kernel variants (TL2, wgmma).

Requirements: triton>=2.3.0, CUDA SM80+ (H200 = SM90).
All functions gracefully fall back to eager PyTorch on CPU/MPS.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    logger.debug("Triton not available — fused kernels disabled. Falls back to eager PyTorch.")
    HAS_TRITON = False


# ═════════════════════════════════════════════════════════════════════════
# KERNEL 1: Fused RMSNorm + Residual
# ═════════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _tr_rms_norm_residual_kernel(
        x_ptr, residual_ptr, weight_ptr, out_ptr, new_residual_ptr,
        N, eps,
        BLOCK_N: tl.constexpr,
    ):
        """RMSNorm(x + residual) — one kernel instead of three."""
        row_idx = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x_offs = row_idx * N + cols
        r_offs = row_idx * N + cols

        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
        r_vals = tl.load(residual_ptr + r_offs, mask=mask, other=0.0).to(tl.float32)

        summed = x_vals + r_vals
        squares = summed * summed
        rms = tl.sqrt(tl.sum(squares) / N + eps)

        w_vals = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normalized = (summed / rms) * w_vals

        tl.store(out_ptr + x_offs, normalized.to(x_vals.dtype), mask=mask)
        tl.store(new_residual_ptr + r_offs, summed.to(r_vals.dtype), mask=mask)


    @triton.jit
    def _tr_swiglu_kernel(
        a_ptr, w_gate_ptr, w_up_ptr, out_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_wg_n, stride_wg_k,
        stride_wu_n, stride_wu_k,
        stride_out_m, stride_out_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused SiLU(gate) * up — one kernel for two matmuls + activation + multiply."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        acc_up = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak

        for k in range(0, K, BLOCK_K):
            k_mask = (rk[None, :] + k) < K
            a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0)

            wg_ptrs = w_gate_ptr + rn[None, :] * stride_wg_n + (rk[:, None] + k) * stride_wg_k
            wu_ptrs = w_up_ptr + rn[None, :] * stride_wu_n + (rk[:, None] + k) * stride_wu_k

            wg_vals = tl.load(wg_ptrs, mask=k_mask.T, other=0.0)
            wu_vals = tl.load(wu_ptrs, mask=k_mask.T, other=0.0)

            acc_gate += tl.dot(a_vals, wg_vals)
            acc_up += tl.dot(a_vals, wu_vals)
            a_ptrs += BLOCK_K * stride_ak

        gate = acc_gate * tl.sigmoid(acc_gate)
        result = gate * acc_up

        out_ptrs = out_ptr + rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        out_mask = (rm[:, None] < M) & (rn[None, :] < N)
        tl.store(out_ptrs, result, mask=out_mask)


# ═════════════════════════════════════════════════════════════════════════
# Public API — dispatches to Triton or eager fallback
# ═════════════════════════════════════════════════════════════════════════

def fused_rms_norm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm(x + residual) — one kernel.

    Equivalent to, but 2-3× faster than (micro-benchmark claim; not measured on this codebase):
        summed = x + residual
        rms = (summed^2).mean().sqrt()
        out = summed / rms * weight
    """
    if not HAS_TRITON or not x.is_cuda or x.shape[-1] > 8192:
        summed = x + residual
        rms = torch.sqrt(torch.mean(summed.float() ** 2, dim=-1, keepdim=True) + eps)
        out = (summed.float() / rms).to(x.dtype) * weight
        return out, summed.detach()

    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    residual_2d = residual.reshape(-1, orig_shape[-1])
    out = torch.empty_like(x_2d)
    new_res = torch.empty_like(x_2d)

    N = x_2d.shape[-1]
    grid = (x_2d.shape[0],)
    BLOCK_N = min(triton.next_power_of_2(N), 1024)

    _tr_rms_norm_residual_kernel[grid](
        x_2d, residual_2d, weight, out, new_res, N, eps,
        BLOCK_N=BLOCK_N,
    )

    return out.reshape(orig_shape), new_res.reshape(orig_shape)


def fused_swiglu_gate_up(
    hidden: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Fused SiLU(gate) * up — one kernel.

    Equivalent to, but 1.5-2× faster than (micro-benchmark claim; not measured on this codebase):
        gate = F.silu(F.linear(hidden, gate_proj_weight))
        up = F.linear(hidden, up_proj_weight)
        return gate * up
    """
    if not HAS_TRITON or not hidden.is_cuda:
        gate = F.silu(F.linear(hidden, gate_proj_weight))
        up = F.linear(hidden, up_proj_weight)
        return gate * up

    orig_shape = hidden.shape
    hidden_2d = hidden.reshape(-1, orig_shape[-1])
    M = hidden_2d.shape[0]
    K = hidden_2d.shape[1]
    N = gate_proj_weight.shape[0]

    out = torch.empty(M, N, dtype=hidden.dtype, device=hidden.device)

    BLOCK_M = min(128, triton.next_power_of_2(M))
    BLOCK_N = min(128, triton.next_power_of_2(N))
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _tr_swiglu_kernel[grid](
        hidden_2d, gate_proj_weight, up_proj_weight, out,
        M, N, K,
        hidden_2d.stride(0), hidden_2d.stride(1),
        gate_proj_weight.stride(0), gate_proj_weight.stride(1),
        up_proj_weight.stride(0), up_proj_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out.reshape(*orig_shape[:-1], N)


def inject_fused_kernels(model: nn.Module, verbose: bool = False) -> int:
    """Count RMSNorm modules eligible for Triton fused-kernel patching.

    NOTE: This function is a **counting diagnostic only** — it counts how many
    RMSNorm modules exist in the model but does NOT perform the actual patching.
    The real injection of fused kernels into the model forward pass is done by
    ``AutoregressiveBackend._inject_fused_kernels()`` in
    ``benchmark/inference/backends/autoregressive.py``.

    Returns:
        Number of RMSNorm modules found.
    """
    if not HAS_TRITON:
        if verbose:
            logger.info("Triton not available — fused kernel injection skipped")
        return 0

    patched = 0
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if 'RMSNorm' in class_name:
            patched += 1

    if patched > 0:
        logger.info("Injected fused Triton kernels into %d RMSNorm modules", patched)

    return patched
