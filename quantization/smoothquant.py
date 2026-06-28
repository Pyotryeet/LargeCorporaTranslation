"""SmoothQuant calibration for static FP8 weight quantization.

SmoothQuant (Xiao et al., ICML 2024) migrates activation outliers into weights
*before* quantization, eliminating the need for dynamic per-token scaling.

For an FP8 weight-only quantization scheme, SmoothQuant's key insight is:

    Y = XW = (X · diag(s)⁻¹) · (diag(s) · W) = X̂ · Ŵ

where s_j = max(|X_j|)^α / max(|W_j|)^(1-α) for each channel j.

The smoothed weights Ŵ are then statically quantized to FP8 E4M3.
Activations stay in BF16 — the smoothing removes outlier spikes that would
otherwise cause catastrophic quantization error.

This module is DECOUPLED from the transformers library.  It operates on raw
tensors extracted from a single calibration forward pass.  No runtime hooks,
no monkey-patching, no dependency on the inference backend.

References
----------
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
  Quantization for Large Language Models", ICML 2024.
- NVIDIA FP8 E4M3 format: ±448 max, 4 exponent bits, 3 mantissa bits.
"""

from __future__ import annotations

import logging
import math
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0

# Default SmoothQuant hyperparameter: α = 0.5 balances migration between
# weights and activations.  α closer to 1.0 migrates more to weights;
# α closer to 0.0 migrates more to activations.
DEFAULT_ALPHA = 0.5


# ── Activation capture ─────────────────────────────────────────────────────


class ActivationCapture:
    """Capture intermediate activations from a calibration forward pass.

    Registers forward hooks on ``nn.Linear`` layers to collect input tensors
    without modifying the model or requiring access to its internals.

    Usage::

        capture = ActivationCapture(model)
        capture.start()
        for batch in calibration_data:
            model(batch)
        capture.stop()
        # Now capture.activations[name] = [tensor_from_batch0, tensor_from_batch1, ...]
    """

    def __init__(self, model: torch.nn.Module, *, capture_inputs: bool = True):
        """Initialize the activation capture harness.

        Parameters
        ----------
        model : torch.nn.Module
            The model to capture activations from. Must contain ``nn.Linear`` layers.
        capture_inputs : bool, keyword-only
            If True (default), capture the input tensor to each Linear layer.
            If False, only register hooks without saving tensors (useful for
            warm-up or debugging).

        Attributes
        ----------
        activations : dict[str, list[Tensor]]
            Mapping from layer name to a list of captured input tensors (one per
            forward call). Populated after ``start()`` / forward / ``stop()``.
        _hooks : list[RemovableHandle]
            Registered forward hooks; cleaned up in ``stop()``.

        Notes
        -----
        Tensors are detached but stay on-device during capture. GPU→CPU transfer
        is deferred to ``stop()`` to avoid per-layer synchronous blocking copies.
        """
        self.model = model
        self.capture_inputs = capture_inputs
        self.activations: dict[str, list[torch.Tensor]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _hook_fn(self, name: str, _module, _input, _output):
        """Forward hook callback invoked after each Linear layer executes.

        Parameters
        ----------
        name : str
            The fully-qualified module name (e.g. ``model.layers.0.self_attn.q_proj``).
        _module : nn.Module
            The Linear module (unused).
        _input : tuple[Tensor, ...]
            The input tuple (``_input[0]`` is the activation tensor).
        _output : Tensor
            The layer output (unused).

        Side Effects
        ------------
        Appends ``_input[0].detach()`` to ``self.activations[name]``.
        Tensors are detached to avoid holding the computation graph but
        stay on-device — GPU→CPU transfer is deferred to ``stop()``.

        Notes
        -----
        This is a private method intended only as a forward-hook callback.
        It is never called directly by user code.
        """
        if self.capture_inputs and _input:
            # Detach to avoid holding the compute graph, but keep on device.
            # GPU→CPU transfer is deferred to stop() to avoid synchronous
            # per-layer blocking copies during the calibration forward pass.
            self.activations.setdefault(name, []).append(_input[0].detach())

    def start(self) -> None:
        """Register activation-capture hooks on all nn.Linear layers."""
        self.activations.clear()
        for n, m in self.model.named_modules():
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n:
                h = m.register_forward_hook(lambda mod, inp, out, name=n: self._hook_fn(
                    name, mod, inp, out,
                ))
                self._hooks.append(h)

    def stop(self) -> None:
        """Remove all registered hooks and transfer captured activations to CPU."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        # Deferred GPU→CPU transfer — avoids synchronous per-layer copies
        # during the calibration forward pass.
        for name in list(self.activations.keys()):
            self.activations[name] = [t.cpu() for t in self.activations[name]]

    def stacked(self, name: str) -> Optional[torch.Tensor]:
        """Return all captured activations for *name* stacked along dim 0."""
        acts = self.activations.get(name)
        if acts is None or not acts:
            return None
        return torch.cat(acts, dim=0)


# ── SmoothQuant scaling factors ────────────────────────────────────────────


def compute_smooth_scales(
    weights: dict[str, torch.Tensor],
    activations: dict[str, torch.Tensor],
    *,
    alpha: float = DEFAULT_ALPHA,
    quant_max: float = FP8_E4M3_MAX,
) -> dict[str, torch.Tensor]:
    """Compute per-channel SmoothQuant scaling factors.

    For each linear layer (name → weight, name → stacked activation):

        s_j = max(|X_j|)^α / max(|W_j|)^(1 - α)

    Parameters
    ----------
    weights : dict[str, Tensor]
        Layer name → weight tensor [out_features, in_features].
    activations : dict[str, Tensor]
        Layer name → stacked calibration activations [N_total, in_features].
    alpha : float
        Migration hyperparameter.  0.5 = balanced (default).
    quant_max : float
        Maximum representable value in the target format (448 for E4M3).

    Returns
    -------
    dict[str, Tensor]
        Layer name → per-channel scale vector [in_features].
    """
    scales: dict[str, torch.Tensor] = {}
    for name, w in weights.items():
        x = activations.get(name)
        if x is None or x.numel() == 0:
            continue

        # Per-channel maximum absolute activation (across all tokens/batches).
        # Activations may be 2D [tokens, in_features] or 3D [batch, seq, in_features].
        #
        # IMPORTANT: assumes weights are [out_features, in_features] layout
        # (PyTorch/HF default).  Models loaded with bitsandbytes may have
        # transposed weights [in_features, out_features]; in that case
        # max(dim=0) gives [out_features] instead of [in_features] and
        # the scale dimension is wrong.
        # Flatten to 2D before max to ensure correct [in_features] output.
        x_flat = x.reshape(-1, x.shape[-1])  # [total_tokens, in_features]
        x_max = x_flat.abs().max(dim=0).values  # [in_features]
        # Per-channel maximum absolute weight.
        w_max = w.abs().max(dim=0).values  # [in_features] — for col-major, this is input-dim max

        # SmoothQuant scale: s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
        # Clamp denominators to avoid division by zero.
        w_max_safe = w_max.clamp(min=1e-8)
        s = (x_max.pow(alpha) / w_max_safe.pow(1.0 - alpha)).float()

        # Clamp to prevent extreme scales.
        s = s.clamp(min=1e-4, max=1e4)
        scales[name] = s

    return scales


def apply_smooth_scales(
    weights: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply SmoothQuant scaling to weights in-place.

    Ŵ = W · diag(s)  →  Ŵ[:, j] = W[:, j] * s_j

    Parameters
    ----------
    weights : dict[str, Tensor]
        Layer name → weight [out_features, in_features].
    scales : dict[str, Tensor]
        Layer name → scale vector [in_features].

    Returns
    -------
    dict[str, Tensor]
        Smoothed weights (same shapes as input).
    """
    smoothed: dict[str, torch.Tensor] = {}
    for name, w in weights.items():
        s = scales.get(name)
        if s is None:
            smoothed[name] = w
            continue
        # Ŵ = W * diag(s): each column j gets multiplied by s_j.
        # For weight [out, in]: s_j acts on the input channel dimension.
        s_dev = s.to(w.device, dtype=torch.float32)
        smoothed[name] = (w.float() * s_dev.unsqueeze(0)).to(w.dtype)
    return smoothed


def compute_activation_scales(
    scales: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute the inverse activation scales for the smoothed activations.

    X̂ = X · diag(s)⁻¹  →  scale_x_j = 1 / s_j

    Parameters
    ----------
    scales : dict[str, Tensor]
        SmoothQuant scale vectors [in_features] per layer.

    Returns
    -------
    dict[str, Tensor]
        Activation scale per layer [in_features] — used at inference to
        dequantize/scale smoothed activations.
    """
    act_scales: dict[str, torch.Tensor] = {}
    for name, s in scales.items():
        act_scales[name] = 1.0 / s.clamp(min=1e-8)
    return act_scales


# ── Full calibration pipeline ──────────────────────────────────────────────


class SmoothQuantCalibrator:
    """End-to-end SmoothQuant calibration for a HuggingFace model.

    Runs a single calibration forward pass, captures activations, computes
    SmoothQuant scaling factors, and applies them to produce smoothed,
    quantization-ready weights.

    Usage::

        calibrator = SmoothQuantCalibrator(model, tokenizer, alpha=0.5)
        calibrator.calibrate(calibration_texts)
        # Weights are now smoothed in-place on the model.
        # Apply static FP8 quantization:
        from benchmark.hardware.precision import apply_static_fp8_to_model
        apply_static_fp8_to_model(model)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        *,
        alpha: float = DEFAULT_ALPHA,
        quant_max: float = FP8_E4M3_MAX,
        max_calibration_tokens: int = 2048,
        device: str | torch.device | None = None,
    ):
        """Initialize a SmoothQuant calibrator.

        Parameters
        ----------
        model : torch.nn.Module
            HuggingFace or PyTorch model containing ``nn.Linear`` layers.
        tokenizer : PreTrainedTokenizer or callable
            Tokenizer with a ``__call__`` that accepts a list of strings and
            returns a dict with ``input_ids`` and ``attention_mask``.
        alpha : float, keyword-only
            Migration hyperparameter (default: 0.5). Values closer to 1.0
            migrate more activation range into weights; values closer to 0.0
            keep more range in activations.
        quant_max : float, keyword-only
            Maximum representable value in the target FP8 format (default:
            448.0 for E4M3).
        max_calibration_tokens : int, keyword-only
            Maximum number of tokens to process during calibration (default:
            2048). Exceeding this stops the calibration forward pass early.
        device : str or torch.device or None, keyword-only
            Device to run calibration on. If None, inferred from the model's
            first parameter.

        Attributes
        ----------
        _scales : dict[str, Tensor]
            SmoothQuant weight scaling factors (s_j per input channel), exposed
            via the ``scales`` property.
        _act_scales : dict[str, Tensor]
            Inverse activation scaling factors (1/s_j per input channel),
            exposed via the ``activation_scales`` property.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.quant_max = quant_max
        self.max_calibration_tokens = max_calibration_tokens
        self.device = device or next(model.parameters()).device
        self._scales: dict[str, torch.Tensor] = {}
        self._act_scales: dict[str, torch.Tensor] = {}

    @property
    def scales(self) -> dict[str, torch.Tensor]:
        """SmoothQuant weight scales (s_j per channel)."""
        return self._scales

    @property
    def activation_scales(self) -> dict[str, torch.Tensor]:
        """Inverse scales for activations (1/s_j per channel)."""
        return self._act_scales

    def calibrate(
        self,
        texts: Iterator[str] | list[str],
        *,
        max_batches: int = 10,
    ) -> int:
        """Run calibration forward pass and apply SmoothQuant smoothing.

        Parameters
        ----------
        texts : iterable of str
            Calibration text data.  100-500 sentences is typical.
        max_batches : int
            Maximum calibration batches to process.

        Returns
        -------
        int
            Number of layers smoothed.
        """
        capture = ActivationCapture(self.model)
        self.model.eval()

        # Guard: empty calibration data produces uncalibrated weights.
        if not texts:
            logger.warning(
                "SmoothQuant calibrate() called with empty texts — no calibration "
                "data available. Weights will NOT be smoothed. Set "
                "TR_SKIP_SMOOTHQUANT=1 to skip this step entirely."
            )
            return 0

        total_tokens = 0
        batch_texts: list[str] = []

        capture.start()
        try:
            with torch.no_grad():
                for text in texts:
                    batch_texts.append(text)

                    if len(batch_texts) >= 8:
                        encoded = self.tokenizer(
                            batch_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=512,
                        ).to(self.device)
                        # CausalLM forward: use input_ids + attention_mask
                        self.model(input_ids=encoded.input_ids,
                                   attention_mask=encoded.attention_mask)
                        total_tokens += encoded.input_ids.numel()
                        batch_texts.clear()

                    if total_tokens >= self.max_calibration_tokens:
                        break

                # Drain remaining batch.
                if batch_texts and total_tokens < self.max_calibration_tokens:
                    encoded = self.tokenizer(
                        batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512,
                    ).to(self.device)
                    self.model(input_ids=encoded.input_ids,
                               attention_mask=encoded.attention_mask)
        finally:
            capture.stop()

        # Extract weights and activations.
        weights: dict[str, torch.Tensor] = {}
        activations: dict[str, torch.Tensor] = {}
        for n, m in self.model.named_modules():
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n:
                weights[n] = m.weight.data.detach().float().cpu()
                stacked = capture.stacked(n)
                if stacked is not None:
                    activations[n] = stacked

        if not weights:
            logger.warning("SmoothQuant: no Linear layers found in model.")
            return 0
        if not activations:
            logger.warning(
                "SmoothQuant: no activations captured — calibration data may "
                "be empty or the model produced no output. Weights will NOT "
                "be smoothed. Set TR_SKIP_SMOOTHQUANT=1 to suppress."
            )
            return 0

        # Compute scales and smooth weights.
        self._scales = compute_smooth_scales(
            weights, activations, alpha=self.alpha, quant_max=self.quant_max,
        )
        self._act_scales = compute_activation_scales(self._scales)

        # Apply smoothing to model weights in-place.
        smoothed_count = 0
        for n, m in self.model.named_modules():
            if isinstance(m, torch.nn.Linear) and n in self._scales:
                s = self._scales[n].to(m.weight.device, dtype=torch.float32)
                m.weight.data = (m.weight.data.float() * s.unsqueeze(0)).to(m.weight.dtype)
                smoothed_count += 1

        logger.info(
            "SmoothQuant: smoothed %d Linear layers (alpha=%.2f, "
            "%d calibration tokens)",
            smoothed_count, self.alpha, total_tokens,
        )
        return smoothed_count
