"""Quantization-Aware Training (QAT) for static FP8 weight quantization.

Provides fake-quantization modules and training utilities that simulate FP8
precision during fine-tuning, allowing the model to adapt to quantization
error.  After QAT, the trained weights are exported to static FP8 via
:func:`~benchmark.hardware.precision.save_fp8_weights`.

This module is DECOUPLED from the inference backend and the transformers
library.  It operates on raw nn.Module graphs and can be applied to any
PyTorch model.

Architecture
------------
- :class:`FP8FakeQuantize` — autograd function that simulates FP8 precision
  in the forward pass with a Straight-Through Estimator (STE) in the backward.
- :class:`FakeQuantizedLinear` — drop-in replacement for nn.Linear with
  per-layer FP8 fake-quantization.
- :func:`prepare_qat` — replaces nn.Linear layers with FakeQuantizedLinear
  for QAT fine-tuning.
- :func:`export_qat_weights` — extracts quantized weights from a QAT-trained
  model, ready for static FP8 inference.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

FP8_E4M3_MAX = 448.0


# ── Fake Quantize ──────────────────────────────────────────────────────────


class FP8FakeQuantize(torch.autograd.Function):
    """Simulate FP8 E4M3 quantization with Straight-Through Estimator.

    Forward:  quantize to FP8 → dequantize (simulates precision loss)
    Backward: pass gradient through unmodified (STE)

    This is the standard QAT approach used by NVIDIA's TensorRT and
    PyTorch's own quantization toolkit.

    Usage::

        x_q = FP8FakeQuantize.apply(x, scale)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Scale → quantize → dequantize
        scaled = x * scale
        clamped = torch.clamp(scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX)
        # Quantize: round to nearest representable FP8 value.
        # FP8 E4M3 has 4 exponent bits and 3 mantissa bits → granular
        # quantization that round() approximates for STE purposes.
        quantized = torch.round(clamped)
        # Save for backward
        ctx.save_for_backward(x, scale)
        ctx.clamp_mask = (scaled >= -FP8_E4M3_MAX) & (scaled <= FP8_E4M3_MAX)
        return quantized / scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: gradient passes through unchanged.
        return grad_output, None


# ── Fake-Quantized Linear Layer ────────────────────────────────────────────


class FakeQuantizedLinear(nn.Module):
    """An nn.Linear that fake-quantizes weights to FP8 during training.

    The forward pass runs in BF16 with FP8-precision weights (simulated
    via fake-quantize).  The backward pass uses STE to train through the
    quantization.

    After QAT, call :func:`export_qat_weights` to extract the final
    quantized weights for static FP8 inference.

    Parameters
    ----------
    in_features, out_features : int
    bias : bool
    weight_scale : float, optional
        Per-tensor scale for weight quantization.  Auto-computed from
        max(|weight|) / 448 if not provided.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_scale: Optional[float] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_scale", torch.tensor(weight_scale or 1.0))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._scale_initialized = weight_scale is not None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if not self._scale_initialized:
            self._update_scale()

    def _update_scale(self) -> None:
        """Recompute weight scale from current weight statistics."""
        with torch.no_grad():
            w_max = self.weight.data.abs().max()
            if w_max > 0:
                self.weight_scale.fill_(w_max / FP8_E4M3_MAX)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update scale before fake-quantization (cheap — no grad).
        if self.training:
            self._update_scale()
        # Fake-quantize weight to FP8 precision.
        w_q = FP8FakeQuantize.apply(self.weight, self.weight_scale)
        return nn.functional.linear(x, w_q, self.bias)


def make_fake_quantized(linear: nn.Linear) -> FakeQuantizedLinear:
    """Convert a standard nn.Linear to a FakeQuantizedLinear.

    Copies weight and bias from the source layer.
    """
    w_max = linear.weight.data.abs().max().item()
    scale = (w_max / FP8_E4M3_MAX) if w_max > 0 else 1.0
    fq = FakeQuantizedLinear(
        linear.in_features, linear.out_features,
        bias=linear.bias is not None,
        weight_scale=scale,
    )
    fq.weight.data.copy_(linear.weight.data)
    if linear.bias is not None:
        fq.bias.data.copy_(linear.bias.data)
    return fq


# ── QAT Prep & Export ──────────────────────────────────────────────────────


def prepare_qat(model: nn.Module) -> int:
    """Replace all nn.Linear layers with :class:`FakeQuantizedLinear`.

    The lm_head is intentionally excluded — FP8 precision loss on the
    vocabulary projection hurts token probability rankings.

    Returns the number of layers replaced.
    """
    replaced = 0

    def _replace(module: nn.Module, parent_name: str = ""):
        nonlocal replaced
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if name == "lm_head" or full_name.endswith(".lm_head"):
                continue
            if isinstance(child, nn.Linear) and not isinstance(child, FakeQuantizedLinear):
                setattr(module, name, make_fake_quantized(child))
                replaced += 1
            else:
                _replace(child, full_name)

    _replace(model)
    logger.info("QAT prep: %d layers replaced with FakeQuantizedLinear", replaced)
    return replaced


def export_qat_weights(model: nn.Module) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Extract FP8-quantized weights from a QAT-trained model.

    Returns a dict mapping layer name → (weight_fp8, scale) where
    weight_fp8 is ``torch.float8_e4m3fn`` and scale is ``torch.float32``.

    These can be fed directly to :func:`benchmark.hardware.precision.save_fp8_weights`.
    """
    weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for n, m in model.named_modules():
        if isinstance(m, FakeQuantizedLinear):
            w = m.weight.data.float()
            w_max = w.abs().max()
            scale = torch.tensor(
                (w_max / FP8_E4M3_MAX).item() if w_max > 0 else 1.0,
                dtype=torch.float32,
            )
            w_fp8 = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(
                torch.float8_e4m3fn,
            )
            weights[n] = (w_fp8, scale)
    return weights


import math  # noqa: E402 (used in reset_parameters)
