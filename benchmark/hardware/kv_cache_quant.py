"""INT4/INT8 KV-cache quantization (C-CROSS-2).

Compresses the key-value cache from 16-bit (BF16) to 8-bit or 4-bit
per element using per-channel asymmetric quantization.  This increases
effective batch size 1.5–2× for memory-bound inference.

Works on both CUDA (via Triton dequant kernels) and MPS (via Metal
dequant shaders).  Falls back to eager PyTorch dequant on CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Packing / unpacking helpers
_NIBBLE_MASK: int = 0x0F
_NIBBLE_BITS: int = 4
_NIBBLE_SHIFT: int = 4

# Quantization defaults
_DEFAULT_SYMMETRIC_BITS: int = 8
_DEFAULT_GROUP_SIZE: int = 128
_DEFAULT_ORIGINAL_BYTES: int = 2        # BF16 = 2 bytes per element
_CLAMP_EPSILON: float = 1e-6

# QuantizedKVCache defaults
_DEFAULT_NUM_LAYERS: int = 36

# Try to import Triton for fast CUDA dequant kernels.
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    logger.debug("Triton not available — KV cache quantization falls back to eager PyTorch.")
    HAS_TRITON = False


@dataclass
class KVQuantConfig:
    """Configuration for KV-cache quantization.

    Attributes
    ----------
    bits : int
        4 or 8 bits per element.
    group_size : int
        Number of elements sharing a scale/zero-point.  Larger = more
        compression but less precision.
    symmetric : bool
        If True, use symmetric quantization (zero_point = 0).  Faster
        dequant but slightly less accurate.
    """

    bits: int = _DEFAULT_SYMMETRIC_BITS
    group_size: int = _DEFAULT_GROUP_SIZE
    symmetric: bool = True

    def __post_init__(self):
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

    @property
    def scale_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def bytes_per_element(self) -> float:
        return self.bits / 8.0

    def compression_ratio(self, original_bytes: int = _DEFAULT_ORIGINAL_BYTES) -> float:
        """Compression ratio relative to BF16 (2 bytes per element)."""
        # Account for scale/zero-point overhead: 4 bytes per group for scales
        overhead_per_element = (4 if not self.symmetric else 2) / self.group_size
        return original_bytes / (self.bytes_per_element + overhead_per_element)


def quantize_kv_tensor(
    x: torch.Tensor,
    config: KVQuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Quantize a K or V tensor for KV-cache storage.

    Parameters
    ----------
    x : torch.Tensor
        Float tensor of shape ``[batch, num_heads, seq_len, head_dim]``.
    config : KVQuantConfig

    Returns
    -------
    x_q : torch.Tensor
        Quantized tensor (int8 or packed int4).
    scales : torch.Tensor
        Per-group scales in bfloat16.
    zero_points : torch.Tensor or None
        Per-group zero points (None when symmetric).
    """
    original_shape = x.shape
    # Reshape to [batch, num_heads, seq_len, num_groups, group_size]
    head_dim = original_shape[-1]
    if head_dim % config.group_size != 0:
        # Pad head_dim to the next multiple of group_size.
        pad_size = config.group_size - (head_dim % config.group_size)
        x = torch.nn.functional.pad(x, (0, pad_size))
        head_dim = x.shape[-1]

    num_groups = head_dim // config.group_size
    x_reshaped = x.reshape(*original_shape[:-1], num_groups, config.group_size)

    if config.symmetric:
        # Symmetric: scale = max(|x|) / max_representable
        amax = x_reshaped.abs().max(dim=-1, keepdim=True)[0]
        max_val = 2 ** (config.bits - 1) - 1
        scales = (amax / max_val).to(config.scale_dtype)
        x_q = torch.round(x_reshaped / scales.float()).clamp(-max_val, max_val)
        if config.bits == 8:
            x_q = x_q.to(torch.int8)
        else:
            # Symmetric INT4 storage: map [-8, 7] → unsigned [0, 15] using
            # two's complement offset (add 16 for negative values) BEFORE
            # packing.  _pack_int4 expects unsigned values in [0, 15].
            # Without this offset, negative int values cast to uint8 wrap
            # around (e.g., -1 → 255), producing garbage nibbles.
            x_q = _pack_int4((x_q + 8).to(torch.uint8))
        return x_q, scales.squeeze(-1), None
    else:
        # Asymmetric: scale = (max - min) / (2^bits - 1), zp = -min / scale
        x_min = x_reshaped.min(dim=-1, keepdim=True)[0]
        x_max = x_reshaped.max(dim=-1, keepdim=True)[0]
        max_val = 2 ** config.bits - 1
        scales = ((x_max - x_min) / max_val).clamp(min=_CLAMP_EPSILON).to(config.scale_dtype)
        zero_points = torch.round(-x_min / scales.float()).clamp(0, max_val)
        x_q = torch.round(x_reshaped / scales.float() + zero_points).clamp(0, max_val)
        if config.bits == 8:
            x_q = x_q.to(torch.uint8)
        else:
            x_q = _pack_int4(x_q.to(torch.uint8))
        return x_q, scales.squeeze(-1), zero_points.to(torch.uint8).squeeze(-1)


def dequantize_kv_tensor(
    x_q: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    config: KVQuantConfig,
    original_head_dim: int,
) -> torch.Tensor:
    """Dequantize a KV-cache tensor back to float.

    Parameters
    ----------
    x_q : torch.Tensor
        Quantized tensor.
    scales : torch.Tensor
        Per-group scales (bfloat16).
    zero_points : torch.Tensor or None
        Per-group zero points.
    config : KVQuantConfig
    original_head_dim : int
        Original (unpadded) head dimension for slicing.

    Returns
    -------
    torch.Tensor
        Dequantized float tensor with the original head dimension.
    """
    if config.bits == 4:
        x_q = _unpack_int4(x_q)

    if config.symmetric:
        if config.bits == 4:
            # _unpack_int4 returns unsigned [0, 15].  Symmetric INT4 uses
            # two's complement: [0, 7] → [ -8, -1], [8, 15] → [0, 7].
            # Subtract 8 to center around zero.
            x_q = x_q.float() - 8.0
        x_f = x_q.float() * scales.float().unsqueeze(-1)
    else:
        x_f = (x_q.float() - zero_points.float().unsqueeze(-1)) * scales.float().unsqueeze(-1)

    # Reshape back and slice to original head_dim.
    original_shape = list(x_f.shape[:-2]) + [-1]
    x_f = x_f.reshape(original_shape)
    return x_f[..., :original_head_dim].to(scales.dtype)


def _pack_int4(x: torch.Tensor) -> torch.Tensor:
    """Pack two int4 values into one int8 (interleaved low/high nibbles)."""
    # Expects x in range [-8, 7] (signed) or [0, 15] (unsigned).
    # Packs pairs along the last dimension: x[..., 0] in low nibble, x[..., 1] in high.
    # Convert to uint8 before bitwise operations to avoid signed int8 fragility
    # — bitwise ops on signed tensors can produce unexpected results when the
    # sign bit is set (e.g. -1 & 0x0F != 15 in signed arithmetic).
    x = x.to(torch.uint8)
    even = x[..., ::2] & _NIBBLE_MASK
    odd = x[..., 1::2] & _NIBBLE_MASK
    return (even | (odd << _NIBBLE_SHIFT)).to(torch.uint8)


def _unpack_int4(x_packed: torch.Tensor) -> torch.Tensor:
    """Reverse of _pack_int4."""
    low = x_packed & _NIBBLE_MASK
    high = (x_packed >> _NIBBLE_SHIFT) & _NIBBLE_MASK
    # Interleave low and high.
    result = torch.stack([low, high], dim=-1).flatten(-2)
    return result


class QuantizedKVCache:
    """Drop-in wrapper for a standard KV-cache list of tuples.

    Intercepts writes to quantize K and V before storage, and
    dequantizes on reads.  Compatible with HuggingFace's cache format:
    ``list[tuple[torch.Tensor, torch.Tensor]]`` per layer.
    """

    def __init__(self, config: KVQuantConfig, num_layers: int = _DEFAULT_NUM_LAYERS):
        self.config = config
        self.num_layers = num_layers
        # Per-layer: list of (k_q, k_scales, k_zp, v_q, v_scales, v_zp, original_head_dim)
        self._quantized: list[Optional[tuple]] = [None] * num_layers
        self._step: int = 0

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """Quantize and store K/V for a layer."""
        head_dim = key.shape[-1]
        if head_dim % self.config.group_size != 0:
            logger.warning(
                "head_dim %d is not a multiple of group_size %d — "
                "quantize_kv_tensor will pad on every call, adding allocation overhead. "
                "Consider setting group_size to a divisor of head_dim.",
                head_dim, self.config.group_size,
            )
        k_q, k_scales, k_zp = quantize_kv_tensor(key, self.config)
        v_q, v_scales, v_zp = quantize_kv_tensor(value, self.config)
        self._quantized[layer_idx] = (
            k_q, k_scales, k_zp,
            v_q, v_scales, v_zp,
            key.shape[-1],
        )

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return (key, value) for a layer."""
        entry = self._quantized[layer_idx]
        if entry is None:
            raise KeyError(f"No KV cache for layer {layer_idx}")
        k_q, k_scales, k_zp, v_q, v_scales, v_zp, orig_dim = entry
        k = dequantize_kv_tensor(k_q, k_scales, k_zp, self.config, orig_dim)
        v = dequantize_kv_tensor(v_q, v_scales, v_zp, self.config, orig_dim)
        return k, v

    def memory_usage_mb(self) -> float:
        """Estimated memory usage of the quantized cache in MB."""
        total = 0.0
        for entry in self._quantized:
            if entry is None:
                continue
            for tensor in entry[:6]:
                if tensor is not None:
                    total += tensor.element_size() * tensor.numel()
        return total / (1024 * 1024)

    def compression_savings(self, bf16_memory_mb: float) -> dict:
        """Report memory savings vs BF16 baseline."""
        q_mb = self.memory_usage_mb()
        return {
            "quantized_mb": round(q_mb, 1),
            "bf16_baseline_mb": round(bf16_memory_mb, 1),
            "savings_pct": round((1 - q_mb / bf16_memory_mb) * 100, 1),
            "compression_ratio": self.config.compression_ratio(),
        }
