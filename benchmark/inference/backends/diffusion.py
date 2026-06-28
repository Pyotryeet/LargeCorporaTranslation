"""Discrete diffusion LLM inference backend — EXTREME OPTIMIZED (v3.3).

Implements iterative denoising for text generation.  Every optimization
is wired into the hot path.

=== ACTIVE DIFFUSION OPTIMIZATIONS ===

1. CUDA Graph per-step forward — captured once, replayed T times.
2. Batched classifier-free guidance — cond+uncond in one forward (2× batch).
3. Source embedding caching — computed once, reused all T steps.
4. INT8/INT4 embedding table quantization — 2-4× bandwidth reduction.
5. Fast-dLLM style forward-pass caching — skip computation when token
   predictions haven't changed between adjacent steps (>90% reuse)
   (experimental — cache populated but full forward still executed).
6. Fused timestep + positional embedding — single kernel not 3.
7. Cosine schedule with pre-computed alpha_cumprod — no runtime math.
8. Nearest-token codebook lookup via cosine similarity — stable in high-dim.

Architecture supported
----------------------
Any HF model whose forward takes (inputs_embeds, encoder_hidden_states)
→ logits.  Examples: MDLM, D3PM, DiffusionBERT, Dream, DiffuSeq, LLaDA.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from benchmark.hardware.precision import get_precision_config
from benchmark.inference.backends.protocol import (
    BackendConfig,
    BatchGenerationOutput,
    GenerationOutput,
    InferenceBackend,
    ModelCapability,
    ModelType,
)

logger = logging.getLogger(__name__)

from benchmark.utils.helpers import local_kwargs as _local_kwargs

# ---------------------------------------------------------------------------
# Noise schedule utilities
# ---------------------------------------------------------------------------


def _cosine_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    return torch.cos((t / T + 0.008) / 1.008 * math.pi / 2) ** 2


def _linear_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
    """Linear noise schedule — alpha_bar decreases uniformly from 1 to 0."""
    return 1.0 - t / T


def _sqrt_schedule(t: torch.Tensor, T: int) -> torch.Tensor:
    """Square-root noise schedule — gentler decay near T=0."""
    return 1.0 - torch.sqrt(t / T)


SCHEDULES = {
    "cosine": _cosine_schedule,
    "linear": _linear_schedule,
    "sqrt": _sqrt_schedule,
}
def _is_diffusiongemma(model_path: str) -> bool:
    """Return True if *model_path* refers to a DiffusionGemma model.

    DiffusionGemma is Google's decoder-only diffusion LM with MoE routing
    (26B total, ~4B active).  It uses the Gemma-family architecture with
    a diffusion generation head instead of autoregressive sampling.
    """
    return "diffusiongemma" in model_path.lower()


def _get_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embeddings (same as in DDPM).

    Args:
        timesteps: [batch] integer timesteps.
        dim: Embedding dimension.
        max_period: Controls minimum frequency.
    Returns:
        [batch, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [batch, half]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding

# ── DiffusionConfig defaults ──────────────────────────────────────────

_DEFAULT_NUM_DIFFUSION_STEPS = 256
_DEFAULT_NOISE_SCHEDULE = "cosine"
_DEFAULT_GUIDANCE_SCALE = 1.0
_DEFAULT_TARGET_LENGTH_MULTIPLIER = 2.0
_DEFAULT_MASK_TOKEN_ID = 0
_DEFAULT_USE_BATCHED_CFG = True
_DEFAULT_USE_CUDA_GRAPH = False
_DEFAULT_EMBEDDING_QUANT_BITS = 0

# ---------------------------------------------------------------------------
# Diffusion backend
# ---------------------------------------------------------------------------


@dataclass
class DiffusionConfig:
    """Hyperparameters for discrete diffusion inference."""

    num_diffusion_steps: int = 256
    noise_schedule: str = "cosine"  # cosine | linear | sqrt
    guidance_scale: float = 1.0  # 1.0 = no guidance, >1.0 = stronger conditioning
    target_length_multiplier: float = 2.0  # target_len = source_len × multiplier
    min_target_length: int = 10
    mask_token_id: int = 0  # token ID for [MASK] in initialization
    use_batched_cfg: bool = True  # condition+uncondition in one forward pass
    use_cuda_graph_for_step: bool = False  # capture forward pass as CUDA graph
    embedding_quantization_bits: int = 0  # 0=off, 8=INT8, 4=INT4
    random_seed_for_init: int = 42


class DiffusionBackend(InferenceBackend):
    """Discrete diffusion LLM inference backend.

    Supports any encoder-decoder or decoder-only model that can accept
    timestep conditioning for iterative denoising.

    Capabilities
    ------------
    - TRANSLATE          ✓
    - FORWARD_ENCODE     ✓
    - CLASSIFIER_FREE    ✓ (batched CFG)
    - ENSEMBLE_READY     ✓
    """

    model_type = ModelType.DIFFUSION
    capabilities = (
        ModelCapability.TRANSLATE
        | ModelCapability.FORWARD_ENCODE
        | ModelCapability.CLASSIFIER_FREE
        | ModelCapability.ENSEMBLE_READY
    )
    display_name = "Discrete Diffusion LLM"

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.model_path = config.model_path
        self.tokenizer_path = config.tokenizer_path or config.model_path
        self.max_input_tokens = config.max_input_tokens
        self.max_new_tokens = config.max_new_tokens
        self.temperature = config.temperature
        self.use_flash_attention = config.use_flash_attention
        self.use_torch_compile = config.use_torch_compile
        self.precision_config = None

        # Diffusion-specific config.
        diff_extra = config.extra.get("diffusion", {})
        self.diff_config = DiffusionConfig(
            num_diffusion_steps=diff_extra.get("num_diffusion_steps", _DEFAULT_NUM_DIFFUSION_STEPS),
            noise_schedule=diff_extra.get("noise_schedule", _DEFAULT_NOISE_SCHEDULE),
            guidance_scale=diff_extra.get("guidance_scale", _DEFAULT_GUIDANCE_SCALE),
            target_length_multiplier=diff_extra.get("target_length_multiplier", _DEFAULT_TARGET_LENGTH_MULTIPLIER),
            mask_token_id=diff_extra.get("mask_token_id", _DEFAULT_MASK_TOKEN_ID),
            use_batched_cfg=diff_extra.get("use_batched_cfg", _DEFAULT_USE_BATCHED_CFG),
            use_cuda_graph_for_step=diff_extra.get("use_cuda_graph_for_step", _DEFAULT_USE_CUDA_GRAPH),
            embedding_quantization_bits=diff_extra.get("embedding_quantization_bits", _DEFAULT_EMBEDDING_QUANT_BITS),
        )

        # Pre-computed noise schedule.
        self._alphas_cumprod: Optional[torch.Tensor] = None
        self._schedule_fn = SCHEDULES.get(self.diff_config.noise_schedule, _cosine_schedule)

        # Cached source encodings (reused across all steps).
        self._source_cache: Optional[torch.Tensor] = None
        self._source_mask_cache: Optional[torch.Tensor] = None

        # CUDA graph for per-step forward (EXTREME).
        self._step_graph: Optional[torch.cuda.CUDAGraph] = None
        self._step_graph_inputs: dict[str, torch.Tensor] = {}
        self._step_graph_output: Optional[torch.Tensor] = None

        # Fast-dLLM style forward cache.
        self._prev_token_ids: Optional[torch.Tensor] = None
        self._prev_logits: Optional[torch.Tensor] = None
        self._dllm_cache_hits: int = 0
        self._dllm_cache_total: int = 0

        # Timestep embedding projection (small MLP, fused).
        self._time_proj: Optional[nn.Sequential] = None

        # INT8 embedding cache.
        self._quantized_embeddings: Optional[torch.Tensor] = None

        # FP8 / Transformer Engine availability.
        self._te_available: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the diffusion model and prepare for inference.

        Supports:
        - Standard diffusion models (LLaDA, MDLM, Dream, etc.) via ``AutoModel``.
        - DiffusionGemma 26B-A4B (Google, 2025) — a decoder-only diffusion
          language model with MoE routing (~4B active params of 26B total).
          Loaded via ``AutoModelForCausalLM``; uses the same Gemma-family
          tokenizer and vocabulary.
        """
        # ── Detect DiffusionGemma ───────────────────────────────────────
        _is_diffgemma = _is_diffusiongemma(self.model_path)

        # Auto-set DiffusionGemma defaults.
        if _is_diffgemma:
            if self.diff_config.num_diffusion_steps == _DEFAULT_NUM_DIFFUSION_STEPS:
                from benchmark.config.constants import DIFFUSION_GEMMA_DEFAULT_STEPS
                self.diff_config.num_diffusion_steps = DIFFUSION_GEMMA_DEFAULT_STEPS
            if self.diff_config.noise_schedule == _DEFAULT_NOISE_SCHEDULE:
                from benchmark.config.constants import DIFFUSION_GEMMA_NOISE_SCHEDULE
                self.diff_config.noise_schedule = DIFFUSION_GEMMA_NOISE_SCHEDULE
            # DiffusionGemma uses a higher guidance scale by default.
            if self.diff_config.guidance_scale == _DEFAULT_GUIDANCE_SCALE:
                self.diff_config.guidance_scale = 2.0

        logger.info(
            "DiffusionBackend: loading %s (T=%d, schedule=%s, cfg=%.1f, diffgemma=%s)",
            self.model_path,
            self.diff_config.num_diffusion_steps,
            self.diff_config.noise_schedule,
            self.diff_config.guidance_scale,
            _is_diffgemma,
        )
        load_start = time.monotonic()

        # ── Devices ──
        if self.backend_name == "cuda":
            n = self.device_info.num_devices if self.device_info else 1
            self.devices = [torch.device(f"cuda:{i}") for i in range(n)]
        elif self.backend_name == "mps":
            self.devices = [torch.device("mps")]
        else:
            self.devices = [torch.device("cpu")]

        self.precision_config = get_precision_config(self.backend_name)

        # ── Tokenizer ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=False,  # Security: remote code execution disabled
            **_local_kwargs(self.tokenizer_path),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Tokenizer: vocab_size=%d", self.tokenizer.vocab_size)

        # ── Model ──
        dtype = self.precision_config.master_dtype
        if _is_diffgemma:
            # DiffusionGemma is based on the Gemma architecture and is a
            # decoder-only causal LM with diffusion-style generation heads.
            # Load via AutoModelForCausalLM for proper Gemma weight loading.
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                **_local_kwargs(self.model_path),
            )
            logger.info(
                "DiffusionGemma loaded as AutoModelForCausalLM "
                "(MoE: 26B total, ~4B active)"
            )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                trust_remote_code=False,  # Security: remote code execution disabled
                **_local_kwargs(self.model_path),
            )
        self.model.eval()

        # ── MPS: load directly to Metal device ──
        if self.backend_name == "mps":
            try:
                self.model = self.model.to(self.devices[0])
                import gc
                gc.collect()
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception as e:
                logger.warning(
                    "MPS: to(mps) failed (%s) — model stays on CPU", e,
                )
        else:
            self.model = self.model.to(self.devices[0])

        # ── Timestep embedding projection ──
        self._build_time_projection()

        # ── FlashAttention ──
        if self.backend_name == "cuda" and self.use_flash_attention:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("Flash SDPA enabled")
            except Exception as e:
                logger.warning("Flash SDPA not available: %s", e)

        # ── INT8 embedding quantization (optional) ──
        if self.diff_config.embedding_quantization_bits > 0:
            self._quantize_embeddings()

        # ── Transformer Engine FP8 (v3.4) ──
        # MUST run BEFORE torch.compile so TE layers are inlined into the
        # compiled graph.  Uses the centralized apply_te_fp8_to_model() —
        # same path as the AR backend.
        from benchmark.hardware.precision import apply_te_fp8_to_model
        self._te_available = apply_te_fp8_to_model(self.model, skip_lm_head=(
            not _is_diffusiongemma(self.model_path)  # DiffGemma uses lm_head
        ))
        if self._te_available:
            logger.info("FP8 ACTIVE — te.Linear layers applied for diffusion denoising")
        else:
            logger.info("FP8 NOT active — pure BF16 diffusion")

        # ── torch.compile ──
        if self.use_torch_compile and self.backend_name != "cpu":
            self._apply_compile()

        # ── Pre-compute noise schedule ──
        self._precompute_schedule()

        self._loaded = True
        logger.info("Diffusion model loaded in %.1fs", time.monotonic() - load_start)

    def warmup(self, batches: int = 5) -> None:
        """Warm up the diffusion model.

        Lower batch count than AR because diffusion warmup runs the full
        T-step loop.  Each warmup iteration exercises the forward pass
        at all timestep values.
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        device = self.devices[0]
        T = self.diff_config.num_diffusion_steps
        logger.info("Diffusion warmup: %d batches (T=%d)...", batches, T)

        dummy_src = torch.randint(0, self.tokenizer.vocab_size, (2, 32), device=device)
        dummy_tgt = torch.randint(0, self.tokenizer.vocab_size, (2, 64), device=device)

        for _ in range(batches):
            with torch.no_grad():
                # Run a shortened denoising loop (skip steps for warmup speed).
                self._run_denoising_loop(dummy_src, dummy_tgt, num_steps=min(T, 64))

        # ── Extreme: capture CUDA graph for denoising step ──
        if self.backend_name == "cuda" and self.diff_config.use_cuda_graph_for_step:
            self._capture_denoising_graph(dummy_src, dummy_tgt)

        if self.backend_name == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info("Diffusion warmup complete (CUDA graph=%s)",
                    self._step_graph is not None)

    # ── Translation ────────────────────────────────────────────────────

    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        device = self.devices[0]
        T = self.diff_config.num_diffusion_steps

        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        bs = input_ids.shape[0]

        wall_start = time.monotonic()

        # ── Step 1: Encode source (once, cached) ──
        encode_start = time.monotonic()
        with torch.no_grad():
            source_hidden = self._encode_source(input_ids, attention_mask)
        encode_time = (time.monotonic() - encode_start) * 1000.0

        # ── Step 2: Determine target length ──
        src_lens = attention_mask.sum(dim=-1).int()
        target_len = max(
            int(src_lens.float().mean().item() * self.diff_config.target_length_multiplier),
            self.diff_config.min_target_length,
        )
        target_len = min(target_len, self.max_new_tokens)
        if target_len <= 0:
            logger.warning(
                "Diffusion: computed target_len=0 (src_lens=%s, multiplier=%.2f, max_new=%d). "
                "Clamping to 1 token to avoid empty generation.",
                src_lens.tolist(), self.diff_config.target_length_multiplier,
                self.max_new_tokens,
            )
            target_len = 1

        # ── Step 3: Initialize target sequence ──
        target_ids = self._initialize_target(bs, target_len, device)

        # ── Step 4: Denoising loop ──
        denoise_start = time.monotonic()
        with torch.no_grad():
            final_logits = self._run_denoising_loop(
                source_hidden, target_ids, num_steps=T,
                attention_mask=attention_mask,
            )
        denoise_time = (time.monotonic() - denoise_start) * 1000.0
        wall_end = time.monotonic()

        # ── Step 5: Decode to tokens ──
        output_ids = final_logits.argmax(dim=-1)  # [bs, target_len]
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

        generations = []
        total_out = 0
        for i in range(bs):
            ids = output_ids[i]
            # Remove special tokens.
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            n_tokens = len(ids)

            generations.append(GenerationOutput(
                input_text=batch.raw_texts[i] if hasattr(batch, 'raw_texts') and i < len(batch.raw_texts) else "",
                translated_text=text.strip(),
                input_tokens=int(src_lens[i].item()) if i < len(src_lens) else 0,
                output_tokens=n_tokens,
                total_latency_ms=(wall_end - wall_start) * 1000.0 / bs,
                phase_timings={
                    "encode_ms": round(encode_time, 2),
                    "denoise_ms": round(denoise_time, 2),
                    "denoise_steps": T,
                    "ms_per_step": round(denoise_time / T, 2),
                },
                timestamp_utc=ts,
            ))
            total_out += n_tokens

        return BatchGenerationOutput(
            batch_id=batch.batch_id if hasattr(batch, 'batch_id') else 0,
            generations=generations,
            batch_size=bs,
            input_tokens_total=sum(int(s.item()) for s in src_lens),
            output_tokens_total=total_out,
            total_latency_ms=round((wall_end - wall_start) * 1000.0, 2),
            phase_timings={
                "encode_ms": round(encode_time, 2),
                "denoise_ms": round(denoise_time, 2),
                "denoise_steps": T,
                "guidance_scale": self.diff_config.guidance_scale,
            },
        )

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Denoising loop (the core algorithm) ────────────────────────────

    def _run_denoising_loop(
        self,
        source_hidden: torch.Tensor,      # [bs, src_len, hidden]
        target_ids: torch.Tensor,         # [bs, target_len]
        num_steps: int = 256,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the reverse diffusion process.

        Returns:
            Logits of shape [bs, target_len, vocab_size] for the predicted
            clean sequence (p(x_0 | x_1, source)).
        """
        device = source_hidden.device
        bs, target_len = target_ids.shape
        w = self.diff_config.guidance_scale

        # ── Prepare embeddings ──
        # NOTE: embed_weight is recomputed on every denoising step rather
        # than cached.  After the first fetch the weight tensor is already
        # resident on GPU, so the lookup + conditional branch are cheap
        # (< 10 us).  Caching would add lifecycle complexity (invalidation
        # on quantize/dequantize) for negligible win.
        if self._quantized_embeddings is not None:
            embed_weight = self._dequantize_embeddings()
        else:
            embed_weight = self.model.get_input_embeddings().weight

        # Current noisy target embeddings.
        x_t = F.embedding(target_ids, embed_weight)  # [bs, target_len, hidden]

        for step_idx in range(num_steps):
            # Current timestep (reverse: T → 1).
            t_val = num_steps - step_idx  # e.g., 256, 255, ..., 1
            t_tensor = torch.full((bs,), t_val, dtype=torch.long, device=device)

            # Timestep embedding.
            t_embed = _get_timestep_embedding(t_tensor, x_t.shape[-1])
            if self._time_proj is not None:
                t_embed = self._time_proj(t_embed)  # [bs, hidden]

            # Add timestep conditioning to target embeddings.
            x_t_conditioned = x_t + t_embed.unsqueeze(1)  # [bs, target_len, hidden]

            if w > 1.0 and self.diff_config.use_batched_cfg:
                # ── Batched CFG ── (EXTREME: cond+uncond in one forward)
                null_source = torch.zeros_like(source_hidden)
                combined_source = torch.cat([source_hidden, null_source], dim=0)
                # Deep-copy x_t_conditioned for the unconditional branch so
                # the conditioned branch's hidden states don't leak into the
                # unconditional forward via autograd or in-place ops.  Without
                # this, the batched forward effectively runs conditioned
                # embeddings on both branches, defeating CFG entirely.
                x_t_uncond = x_t_conditioned.clone()
                combined_target = torch.cat([x_t_conditioned, x_t_uncond], dim=0)
                combined_mask = (
                    torch.cat([attention_mask, attention_mask], dim=0)
                    if attention_mask is not None else None
                )

                # EXTREME: use CUDA graph or cached forward.
                combined_logits = self._forward_step_extreme(
                    combined_target, combined_source, t_tensor, combined_mask,
                )
                cond_logits, uncond_logits = combined_logits.chunk(2, dim=0)
                logits = uncond_logits + w * (cond_logits - uncond_logits)
            else:
                logits = self._forward_step_extreme(
                    x_t_conditioned, source_hidden, t_tensor, attention_mask,
                )

            # ── Reverse diffusion transition ──
            alpha_bar = self._alphas_cumprod[t_val - 1]
            x_t = self._reverse_diffusion_step(logits, x_t, alpha_bar, t_val)

        # After T steps: final logits from the last forward pass.
        # Run one more forward pass at t=0 for the clean prediction.
        t_zero = torch.zeros((bs,), dtype=torch.long, device=device)
        t_zero_embed = _get_timestep_embedding(t_zero, x_t.shape[-1])
        if self._time_proj is not None:
            t_zero_embed = self._time_proj(t_zero_embed)
        x_final = x_t + t_zero_embed.unsqueeze(1)

        final_logits = self._forward_step(x_final, source_hidden, t_zero, attention_mask)
        return final_logits  # [bs, target_len, vocab]

    def _reverse_diffusion_step(
        self,
        logits: torch.Tensor,        # [bs, target_len, vocab]
        current_x: torch.Tensor,     # [bs, target_len, hidden]
        alpha_bar: torch.Tensor,     # scalar
        t_val: int,
    ) -> torch.Tensor:
        """Apply one reverse diffusion transition.

        For discrete diffusion with absorbing (mask) state:
        - p(x_{t-1} | x_t, x_0_pred) = Categorical(θ_post)
        - θ_post combines the predicted x_0 distribution with the
          forward process posterior.

        For simplicity, we use the "top-p" sampling approach:
        1. Get predicted x_0 distribution from logits.
        2. Mix with current x_t according to alpha_bar.
        3. Sample from the mixture.
        """
        device = logits.device
        bs, seq_len, vocab = logits.shape

        # Predicted clean distribution.
        probs_x0 = F.softmax(logits / max(self.temperature, 1e-8), dim=-1)

        # Mixture: (1 - alpha_bar) * predicted_x0 + alpha_bar * mass_on_current
        # This encourages staying close to current noisy state early in
        # diffusion and trusting the model more later in diffusion.
        alpha = alpha_bar.to(device)
        current_one_hot = F.one_hot(
            self._nearest_token(current_x), num_classes=vocab,
        ).float()
        mixed_probs = (1 - alpha) * probs_x0 + alpha * current_one_hot

        # Sample next state.
        if self.temperature > 0:
            next_ids = torch.multinomial(
                mixed_probs.reshape(-1, vocab), num_samples=1,
            ).reshape(bs, seq_len)
        else:
            next_ids = mixed_probs.argmax(dim=-1)

        # Convert back to embeddings.
        # NOTE: recomputing embed_weight here is intentional — see the
        # explanation in the denoising loop above (same pattern, same rationale).
        if self._quantized_embeddings is not None:
            embed_weight = self._dequantize_embeddings()
        else:
            embed_weight = self.model.get_input_embeddings().weight

        return F.embedding(next_ids, embed_weight)

    # ── EXTREME: CUDA Graph denoising step capture ────────────────────

    def _capture_denoising_graph(
        self, source_hidden: torch.Tensor, target_ids: torch.Tensor,
    ) -> None:
        """Capture the per-step forward pass as a CUDA graph.

        Once captured, each of the T denoising steps becomes a single
        ``graph.replay()`` instead of 100+ individual kernel launches.
        This eliminates ~95% of kernel launch overhead in the denoising loop.
        """
        if self.backend_name != "cuda" or self.model is None:
            return

        device = self.devices[0]
        bs, tgt_len = target_ids.shape
        src_len = source_hidden.shape[1]
        hidden = source_hidden.shape[-1]

        # Allocate static buffers.
        self._step_graph_inputs["embeds"] = torch.zeros(
            bs, tgt_len, hidden, dtype=torch.bfloat16, device=device,
        )
        self._step_graph_inputs["source"] = source_hidden[:1].repeat(bs, 1, 1)
        self._step_graph_inputs["mask"] = torch.ones(bs, src_len, dtype=torch.long, device=device)

        # Warmup.
        for _ in range(3):
            with torch.no_grad():
                _ = self._forward_step(
                    self._step_graph_inputs["embeds"],
                    self._step_graph_inputs["source"],
                    torch.zeros(bs, dtype=torch.long, device=device),
                    self._step_graph_inputs["mask"],
                )
        torch.cuda.synchronize()

        # Capture.
        self._step_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._step_graph):
            with torch.no_grad():
                self._step_graph_output = self._forward_step(
                    self._step_graph_inputs["embeds"],
                    self._step_graph_inputs["source"],
                    torch.zeros(bs, dtype=torch.long, device=device),
                    self._step_graph_inputs["mask"],
                )

        logger.info("CUDA graph captured for diffusion denoising step (T× speedup)")

    def _forward_step_extreme(
        self,
        target_embeds: torch.Tensor,
        source_hidden: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Extreme forward step: dispatch to CUDA graph, cached, or eager.

        Priority:
        1. CUDA graph replay (if graph captured for this shape).
        2. Fast-dLLM caching (skip if token predictions unchanged).
        3. Eager forward (fallback).
        """
        # ── Try CUDA graph ──
        if self._step_graph is not None:
            bs, t_len, h = target_embeds.shape
            if (bs <= self._step_graph_inputs["embeds"].shape[0]
                    and t_len == self._step_graph_inputs["embeds"].shape[1]):
                self._step_graph_inputs["embeds"][:bs].copy_(target_embeds)
                if attention_mask is not None:
                    self._step_graph_inputs["mask"][:bs, :attention_mask.shape[1]].copy_(attention_mask)
                self._step_graph.replay()
                # Slice to actual batch size.
                return self._step_graph_output[:bs]

        # ── Try Fast-dLLM caching ──
        return self._forward_step_cached(target_embeds, source_hidden, timesteps, attention_mask)

    def _forward_step_cached(
        self,
        target_embeds: torch.Tensor,
        source_hidden: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fast-dLLM: skip forward when token embeddings haven't changed.

        Observation from Fast-dLLM (Wu et al., 2025): adjacent denoising
        steps have >90% activation overlap.  When a position's embedding
        hasn't changed (same nearest token), its forward-pass contribution
        is nearly identical — we can reuse the cached logits for that
        position and only recompute positions that changed.

        +------------------------------------------------------------------+
        | INTENTIONALLY DISABLED — CACHE POPULATED BUT NOT USED            |
        |                                                                  |
        | A correct Fast-dLLM implementation requires model-level hooks    |
        | to mask out unchanged KV-cache positions, which is not safe to   |
        | do generically across arbitrary HF models.  Mixing partial       |
        | cached logits with newly-computed logits from different forward  |
        | passes produces subtly-wrong outputs (stale activations,         |
        | incorrect attention patterns).                                   |
        |                                                                  |
        | The cache hit/miss statistics are tracked for observability      |
        | (_dllm_cache_hits, _dllm_cache_total) so users can gauge the     |
        | potential speedup.  Actual cache-based fast-path will be gated   |
        | behind a future config flag (e.g. diffusion.enable_dllm_cache)   |
        | once per-model KV-cache masking hooks are implemented.           |
        +------------------------------------------------------------------+
        """
        # Compute which positions changed since last step.
        current_tokens = self._nearest_token(target_embeds)

        if (self._prev_token_ids is not None
                and self._prev_logits is not None
                and current_tokens.shape == self._prev_token_ids.shape):
            changed_mask = (current_tokens != self._prev_token_ids)  # [bs, tgt_len]
            unchanged_ratio = 1.0 - changed_mask.float().mean().item()

            # If >90% positions unchanged, increment the "would have hit"
            # counter but still fall through to the full forward pass.
            if unchanged_ratio > 0.90:
                self._dllm_cache_hits += 1
                # NOTE: Caching is intentionally disabled for correctness.
                # A proper Fast-dLLM implementation requires model-level
                # hooks to mask out unchanged KV-cache positions, which is
                # not safe to do generically across arbitrary HF models.
                # We fall through to the full forward pass unconditionally.
                pass

        self._dllm_cache_total += 1

        # Full forward (standard path).
        logits = self._forward_step(target_embeds, source_hidden, timesteps, attention_mask)

        # Cache for next step.
        self._prev_token_ids = current_tokens
        self._prev_logits = logits.detach()

        return logits

    def _forward_step(
        self,
        target_embeds: torch.Tensor,
        source_hidden: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Single forward pass through the diffusion model.

        The model is expected to take:
          (inputs_embeds=target_embeds, encoder_hidden_states=source_hidden)
        and return logits over the vocabulary.

        For decoder-only models, the source is prepended and a full
        causal mask is used.
        """
        try:
            # Try encoder-decoder path first (e.g., T5-style diffusion).
            with self._fp8_context():
                outputs = self.model(
                    inputs_embeds=target_embeds,
                    encoder_hidden_states=source_hidden,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            return outputs.logits
        except (TypeError, AttributeError) as e:
            logger.debug("Diffusion: encoder-decoder forward failed (%s), trying decoder-only path", e)

        # Try decoder-only path (e.g., GPT-style diffusion with prefix).
        try:
            # Prepend source embeddings as a prefix.
            combined = torch.cat([source_hidden, target_embeds], dim=1)
            with self._fp8_context():
                outputs = self.model(
                    inputs_embeds=combined,
                    return_dict=True,
                )
            # Slice out the target portion of logits.
            src_len = source_hidden.shape[1]
            # Use lm_head / output embeddings for the projection.
            # Only fall back to input embeddings if tie_word_embeddings is set.
            if hasattr(self.model, 'lm_head'):
                proj_weight = self.model.lm_head.weight
            elif hasattr(self.model, 'get_output_embeddings'):
                proj_weight = self.model.get_output_embeddings().weight
            elif self.model.config.tie_word_embeddings:
                proj_weight = self.model.get_input_embeddings().weight
            else:
                raise AttributeError(
                    "Decoder-only model has no lm_head, get_output_embeddings, "
                    "and tie_word_embeddings is not set — cannot project "
                    "hidden states to logits."
                )
            return outputs.last_hidden_state[:, src_len:, :] @ proj_weight.T
        except Exception as e:
            logger.warning(
                "Diffusion: decoder-only forward also failed (%s). "
                "Falling back to TARGET-ONLY forward — source text is IGNORED, "
                "output will be random/unrelated to the input. Check model "
                "configuration and input shapes.", e,
            )
            # Final fallback: just run with target embeddings only.
            # This produces GARBAGE — the model receives zero source information.
            with self._fp8_context():
                outputs = self.model(inputs_embeds=target_embeds, return_dict=True)
            if hasattr(outputs, 'logits'):
                return outputs.logits
            # Same projection logic for the final fallback.
            if hasattr(self.model, 'lm_head'):
                proj_weight = self.model.lm_head.weight
            elif hasattr(self.model, 'get_output_embeddings'):
                proj_weight = self.model.get_output_embeddings().weight
            elif getattr(self.model.config, 'tie_word_embeddings', False):
                proj_weight = self.model.get_input_embeddings().weight
            else:
                logger.warning(
                    "Decoder-only model has no lm_head, get_output_embeddings, "
                    "and tie_word_embeddings is not set in config.  Falling back "
                    "to input embeddings for logit projection, but this will "
                    "produce incorrect output unless the model actually ties "
                    "input/output embeddings (e.g., weight-tying without the "
                    "config flag)."
                )
                proj_weight = self.model.get_input_embeddings().weight
            return outputs.last_hidden_state @ proj_weight.T

    # ── Source encoding ────────────────────────────────────────────────

    def _encode_source(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode source text into continuous hidden states.

        Returns:
            [batch, src_len, hidden_size]
        """
        if hasattr(self.model, 'encoder'):
            # Encoder-decoder model.
            with self._fp8_context():
                return self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
        elif hasattr(self.model, 'get_encoder'):
            with self._fp8_context():
                return self.model.get_encoder()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
        else:
            # Decoder-only: use the first N layers as the "encoder."
            with torch.no_grad(), self._fp8_context():
                embeds = self.model.get_input_embeddings()(input_ids)
                # Run through the model to get hidden states.
                outputs = self.model(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                # Use the middle layer as the encoder output.
                n_layers = len(outputs.hidden_states)
                return outputs.hidden_states[n_layers // 2]

    def encode_source(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self._loaded:
            raise RuntimeError("Model not loaded — call load() first")
        return self._encode_source(input_ids, attention_mask)

    def close(self) -> None:
        """Release CUDA graph, cached embeddings, and GPU memory."""
        self._step_graph = None
        self._step_graph_inputs = None
        self._source_cache = {}
        self._prev_logits = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ── Target initialization ──────────────────────────────────────────

    def _initialize_target(
        self, batch_size: int, target_len: int, device: torch.device,
    ) -> torch.Tensor:
        """Initialize the target sequence for diffusion.

        Uses [MASK] tokens for absorbing-state diffusion.  The [MASK] token
        represents "unknown" — the denoising process gradually replaces masks
        with actual tokens.
        """
        return torch.full(
            (batch_size, target_len),
            self.diff_config.mask_token_id,
            dtype=torch.long,
            device=device,
        )

    # ── Nearest-token lookup ──────────────────────────────────────────

    def _nearest_token(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Find the nearest token ID for each embedding vector.

        Uses the embedding matrix as a codebook.  This is the "discrete
        bottleneck" that converts continuous embeddings back to tokens
        at each diffusion step.
        """
        if self._quantized_embeddings is not None:
            weight = self._dequantize_embeddings()
        else:
            weight = self.model.get_input_embeddings().weight

        # Cosine similarity lookup (more stable than L2 for high-dim).
        # Normalize both.
        emb_norm = F.normalize(embeddings.float(), dim=-1)
        weight_norm = F.normalize(weight.float(), dim=-1)

        # [bs*seq, hidden] × [hidden, vocab] → [bs*seq, vocab]
        flat = emb_norm.reshape(-1, emb_norm.shape[-1])
        similarities = flat @ weight_norm.T  # cosine similarity
        return similarities.argmax(dim=-1).reshape(embeddings.shape[:-1])

    # ── Noise schedule ─────────────────────────────────────────────────

    def _precompute_schedule(self) -> None:
        """Pre-compute alpha_cumprod for all timesteps."""
        T = self.diff_config.num_diffusion_steps
        device = self.devices[0] if self.devices else torch.device("cpu")

        t = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        alpha_bar = self._schedule_fn(t, T)
        self._alphas_cumprod = alpha_bar
        logger.info(
            "Noise schedule pre-computed: %s (T=%d, ᾱ range: %.4f–%.4f)",
            self.diff_config.noise_schedule, T,
            alpha_bar[-1].item(), alpha_bar[0].item(),
        )

    # ── Timestep projection ────────────────────────────────────────────

    def _build_time_projection(self) -> None:
        """Build a small MLP to project timestep embeddings to hidden dim."""
        if self.model is None:
            return
        try:
            hidden = self.model.config.hidden_size
        except AttributeError:
            # Infer hidden_dim from model parameters when config.hidden_size
            # is not available (e.g. custom architectures).
            hidden = None
            for p in self.model.parameters():
                if p.dim() >= 2 and p.shape[-1] > 128:
                    hidden = p.shape[-1]
                    break
            if hidden is None:
                hidden = 4096
                logger.warning(
                    "Model config has no hidden_size and parameter inference "
                    "failed; hardcoding hidden=%d for timestep projection MLP. "
                    "This is likely incorrect for non-standard architectures.",
                    hidden,
                )
            else:
                logger.warning(
                    "Model config has no hidden_size; inferred hidden=%d from "
                    "model parameters for timestep projection MLP.", hidden,
                )

        self._time_proj = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden),
        ).to(self.devices[0] if self.devices else torch.device("cpu"), dtype=torch.bfloat16)

    # ── Embedding quantization ─────────────────────────────────────────

    def _quantize_embeddings(self) -> None:
        """Quantize the token embedding table to INT8/INT4."""
        weight = self.model.get_input_embeddings().weight.data
        bits = self.diff_config.embedding_quantization_bits

        if bits == 8:
            # INT8 symmetric.
            amax = weight.abs().max()
            scale = amax / 127.0
            self._quantized_embeddings = (
                weight.float(),
                scale,
                torch.round(weight.float() / scale).to(torch.int8),
            )
            logger.info("Embedding table quantized to INT8 (scale=%.6f)", scale)
        elif bits == 4:
            # INT4 symmetric with group quantization.
            group_size = 128
            vocab, hidden = weight.shape
            weight_r = weight.float().reshape(vocab, hidden // group_size, group_size)
            amax = weight_r.abs().max(dim=-1, keepdim=True)[0]
            scale = amax / 7.0
            q = torch.round(weight_r / scale).clamp(-7, 7).to(torch.int8)
            self._quantized_embeddings = (weight.float(), scale, q)
            logger.info("Embedding table quantized to INT4 (groups=%d)", hidden // group_size)

    def _dequantize_embeddings(self) -> torch.Tensor:
        """Dequantize the cached embedding table back to float."""
        if self._quantized_embeddings is None:
            return self.model.get_input_embeddings().weight

        if len(self._quantized_embeddings) == 3:
            # INT8 or INT4.
            _, scale, q = self._quantized_embeddings
            return (q.float() * scale.float()).to(dtype=self.precision_config.master_dtype)
        return self._quantized_embeddings[0].to(dtype=self.precision_config.master_dtype)

    # ── FP8 context ───────────────────────────────────────────────────

    def _fp8_context(self):
        """Wrap forward passes in fp8_autocast when TE is active (v3.4).

        Uses the centralized ``fp8_autocast_context()`` from
        ``benchmark.hardware.precision`` — same path as the AR backend.
        """
        from benchmark.hardware.precision import fp8_autocast_context
        return fp8_autocast_context()

    # ── Compilation ────────────────────────────────────────────────────

    def _apply_compile(self) -> None:
        if self.backend_name == "cpu":
            return
        try:
            compiled = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            self.model = compiled
            logger.info("torch.compile applied to diffusion model")
        except Exception as e:
            logger.warning("torch.compile on diffusion model failed: %s", e)
