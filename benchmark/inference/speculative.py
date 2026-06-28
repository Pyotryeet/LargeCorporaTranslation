"""Speculative decoding — draft→verify→accept.

Wire point
----------
``AutoregressiveBackend`` is the integration surface.  When
``use_speculative=True`` appears in the backend config extra dict, the
backend creates a ``SpeculativeDecoder`` in ``load()`` and dispatches
``translate_batch()`` through it instead of the standard decode loop.

Performance expectation
------------------------
Self-speculative (early-layer draft): ~1.1–1.5× in best case, depending
on draft/verify layer ratio and token acceptance rate.  The current
implementation processes sequences one-at-a-time in a per-sequence for
loop — true batch-level vectorization requires tree-attention support to
verify K speculative tokens across all B sequences in the batch
simultaneously.  The draft_model mode with a well-matched small model
can achieve 1.5–2.5× in wall-clock terms but requires a shared tokenizer.

WARNING — Re-forward fallback on crop failure
---------------------------------------------
When ``DynamicCache.crop()`` fails (e.g., older transformers versions),
the verify phase falls back to a full-model re-forward on the accepted
tokens only.  This re-forward erases most of the speculative speedup for
that step because it runs the full L layers instead of just the verify
layers (L-D).  The fallback is correctness-preserving but a performance
pessimisation.

Activation gate
---------------
Speculative decoding is activated via the ``use_speculative=True`` config
flag (set by ``--speculative`` on the CLI, or directly in config YAML).
No environment variable is required.  The backend reads
``extra.get("use_speculative", False)`` in ``load()`` and creates the
appropriate ``SpeculativeDecoder``.  When the flag is absent or False,
the factory returns ``None`` and the backend falls through to standard
autoregressive decode.

Decoders
--------
``SelfSpeculativeDecoder`` (default)
    Splits the main model into an early-layer "draft" and a late-layer
    "verifier."  No second model, no extra VRAM, always tokenizer-compatible.
    The draft runs layers[0:D] autoregressively for K steps; the verifier
    runs layers[D:L] + norm + lm_head on all K candidates in one batched
    forward pass.

``DraftModelSpeculativeDecoder``
    Loads a separate small draft model alongside the main model.
    Requires the draft and main models to **share a tokenizer** — the
    verification step compares token IDs directly, so vocabularies must
    be identical.

Architecture
------------
  ┌─ Prefill (full model, same as standard) ─┐
  │  Populates KV-cache for all layers.       │
  └───────────────────────────────────────────┘
                    │
                    ▼
  ╔═══════════════════════════════════════════╗
  ║  Decode loop (per token position)         ║
  ║                                           ║
  ║  ┌─ Draft ─┐                              ║
  ║  │  K steps of early layers only          ║
  ║  │  → K candidate tokens                  ║
  ║  └─────────┘                              ║
  ║       │                                    ║
  ║       ▼                                    ║
  ║  ┌─ Verify ─┐                             ║
  ║  │  One batched forward through           ║
  ║  │  remaining layers + lm_head            ║
  ║  │  → logits for all K positions          ║
  ║  └──────────┘                             ║
  ║       │                                    ║
  ║       ▼                                    ║
  ║  ┌─ Accept ─┐                             ║
  ║  │  Compare token IDs → accept prefix     ║
  ║  │  until first mismatch                  ║
  ║  └──────────┘                             ║
  ╚═══════════════════════════════════════════╝

Reference:
  - Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"
  - Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling"
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from benchmark.inference.backends.protocol import BatchGenerationOutput

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Attributes
    ----------
    mode : str
        ``"self"`` = self-speculative (early-layer draft, always compatible).
        ``"draft_model"`` = separate small model (requires shared tokenizer).
    draft_model_name : str
        HuggingFace model ID for the small draft model (draft_model mode only).
    num_speculative_tokens : int
        Number of tokens the draft predicts ahead (K).  Smaller values (2-3)
        work better for self-speculative; larger values (5-8) need a
        well-matched draft model.
    num_draft_layers : int
        For self-speculative: how many of the model's early layers to use
        as the draft.  Default 0 means auto-compute as total_layers // 4.
    max_batch_size : int
        Maximum batch size for batched verification.
    """

    mode: str = "self"                       # "self" | "draft_model"
    draft_model_name: str = ""
    num_speculative_tokens: int = 3          # K
    num_draft_layers: int = 0                # 0 = auto (total_layers // 4)
    max_batch_size: int = 32


# ── Abstract decoder interface ───────────────────────────────────────────────


class SpeculativeDecoder(ABC):
    """Abstract speculative decode strategy.

    Subclasses implement ``translate_batch(batch, backend)`` which replaces
    the standard autoregressive decode loop.
    """

    @abstractmethod
    def load(self) -> None:
        """Load any additional resources (e.g. draft model weights)."""
        ...

    @abstractmethod
    def translate_batch(self, batch: Any, backend: Any) -> "BatchGenerationOutput":
        """Translate a pre-tokenised batch using speculative decoding.

        Parameters
        ----------
        batch : PipelineBatch
            Pre-tokenised batch with ``input_ids``, ``attention_mask``,
            ``raw_texts``, ``batch_id``.
        backend : AutoregressiveBackend
            The backend providing ``model``, ``tokenizer``, ``devices``, etc.

        Returns
        -------
        BatchGenerationOutput
        """
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the decoder is ready for translation."""
        ...

    @property
    @abstractmethod
    def stats(self) -> dict:
        """Return cumulative statistics: drafted, accepted, acceptance_rate."""
        ...


# ── Model introspection helpers ──────────────────────────────────────────────


def _find_model_layers(model: nn.Module) -> nn.ModuleList:
    """Heuristically locate the transformer layer list in a HF model.

    Tries common patterns:
      - Gemma3 multimodal: model.model.language_model.layers
      - Gemma / LLaMA:     model.model.layers
      - GPT-2 / OPT:       model.transformer.h
      - T5 encoder-decoder: model.model.decoder.layers
      - Generic:           recursive ModuleList search

    Returns the ``ModuleList`` of decoder layers.

    Raises
    ------
    AttributeError
        If no layer list can be found.
    """
    # Gemma3 multimodal (Gemma3ForConditionalGeneration → Gemma3Model → Gemma3TextModel)
    if (hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")):
        return model.model.language_model.layers

    # GemmaForCausalLM / LLaMAForCausalLM / SmolLMForCausalLM
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # GPT2LMHeadModel / OPTForCausalLM
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    # T5-style encoder-decoder (decoder only)
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        decoder = model.model.decoder
        if hasattr(decoder, "layers"):
            return decoder.layers

    # Fallback: recursive search for a ModuleList with > 4 modules
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            # Check that modules inside look like transformer layers
            first = module[0]
            if hasattr(first, "self_attn") or hasattr(first, "attention"):
                logger.info(
                    "Auto-detected %d layers at %s", len(module), name,
                )
                return module

    raise AttributeError(
        f"Cannot locate transformer layers in model of type "
        f"{type(model).__name__}.  Set speculative_mode='draft_model' "
        f"to use a separate draft model instead."
    )


def _find_embedding(model: nn.Module) -> nn.Module:
    """Find the token embedding module."""
    # Gemma3 multimodal
    if (hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "embed_tokens")):
        return model.model.language_model.embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    raise AttributeError("Cannot locate embedding layer")


def _find_final_norm(model: nn.Module) -> nn.Module:
    """Find the final layer norm before the lm_head."""
    # Gemma3 multimodal
    if (hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "norm")):
        return model.model.language_model.norm
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise AttributeError("Cannot locate final layer norm")


def _find_lm_head(model: nn.Module) -> nn.Module:
    """Find the language model head."""
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        return model.model.lm_head
    # Gemma3 multimodal: lm_head is on the top-level model
    raise AttributeError("Cannot locate lm_head")


# ── KV-cache utility functions (used by both speculative decoder classes) ──


def _clone_kv_cache(past_kv):
    """Build a fresh ``DynamicCache`` from a model's ``past_key_values``.

    Returns a ``DynamicCache`` if available.  Falls back to returning
    ``past_kv`` unchanged when ``DynamicCache`` is not importable (older
    transformers) or when *past_kv* is already a plain tuple (used by
    tiny test models).

    The returned cache is independent — cloning the tensors so draft
    updates don't corrupt the prefill KV.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return past_kv  # pre-4.45 transformers — tuple of (k,v) tuples

    # If past_kv is already a DynamicCache, clone its entries.
    if isinstance(past_kv, DynamicCache):
        cache = DynamicCache()
        for i in range(len(past_kv)):
            k, v = past_kv[i]
            cache.update(k.clone(), v.clone(), i)
        return cache

    # Tuple-of-tuples format (older transformers or test models).
    # Return as-is — individual layer calls accept plain tuples too.
    return past_kv


def _expand_cache_for_batch_fn(past_kv, target_batch_size: int):
    """Expand a ``DynamicCache`` to *target_batch_size* by repeating
    each layer's KV entries.

    Used when the verify phase processes K speculative candidates as a
    batch — the prefill cache has batch=1 but the verify forward needs
    batch=K.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return past_kv

    if not isinstance(past_kv, DynamicCache):
        return past_kv

    cache = DynamicCache()
    for i in range(len(past_kv)):
        k, v = past_kv[i]
        # k, v: [1, num_heads, seq_len, head_dim]
        k_expanded = k.repeat(target_batch_size, 1, 1, 1)
        v_expanded = v.repeat(target_batch_size, 1, 1, 1)
        cache.update(k_expanded, v_expanded, i)
    return cache


def _safe_crop_kv(past_kv, target_length: int):
    """Crop a KV-cache to *target_length* sequence positions.

    Falls back to manual tensor slicing when ``.crop()`` is unavailable
    (pre-4.45 transformers, tuple-of-tuples from tiny test models, etc.)
    instead of a costly full-model re-forward.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    # DynamicCache with .crop() support (transformers >= 4.45)
    if DynamicCache is not None and isinstance(past_kv, DynamicCache):
        past_kv.crop(target_length)
        return past_kv

    # Tuple-of-tuples: ((k0,v0), (k1,v1), ...) — older transformers or
    # tiny test models.  Manually slice each (k,v) to [..., :target_length, :].
    if isinstance(past_kv, tuple) and len(past_kv) > 0 and isinstance(past_kv[0], tuple):
        def _slice_entry(entry):
            if len(entry) == 2:
                return (entry[0][..., :target_length, :],
                        entry[1][..., :target_length, :])
            # Single-element cache entry (tiny test models use (hidden_state,))
            return (entry[0][..., :target_length, :],)
        cropped = tuple(_slice_entry(e) for e in past_kv)
        return cropped

    # EncoderDecoderCache — crop only the self-attention cache, leaving
    # cross-attention KV intact.
    if hasattr(past_kv, 'self_attention_cache') and past_kv.self_attention_cache is not None:
        past_kv.self_attention_cache = _safe_crop_kv(
            past_kv.self_attention_cache, target_length,
        )
        return past_kv

    # Last resort: .crop() may exist on a custom object — try and let it
    # raise if not available (caller still handles the exception).
    past_kv.crop(target_length)
    return past_kv


# ═════════════════════════════════════════════════════════════════════════════
# Self-Speculative Decoder — zero extra VRAM, always tokenizer-compatible
# ═════════════════════════════════════════════════════════════════════════════


class SelfSpeculativeDecoder(SpeculativeDecoder):
    """Self-speculative decoding using early-layer exit as the draft model.

    Splits the main model into two stages:
      - **Draft stage** (layers[0:D]): runs autoregressively for K steps
        on a single token per step, producing K candidate tokens.
      - **Verify stage** (layers[D:L] + norm + lm_head): runs once on all
        K candidate hidden states as a batch, producing the final logits.

    Token IDs from the verify stage are compared against draft tokens.
    Matching prefixes are accepted; at the first mismatch, the verify
    token is used instead.

    This requires **zero extra GPU memory** and is **always tokenizer-
    compatible** since it uses the same model.

    Performance model
    -----------------
    For a model with L layers and D draft layers, K speculative tokens:
      - Draft cost:  D/L × K forward passes (single-token each)
      - Verify cost: (L-D)/L × 1 forward pass  (batched K tokens)
      - Speedup:     K × acceptance_rate / (K × D/L + (L-D)/L)

    With L=16, D=4, K=3, acceptance_rate=0.75:
      - Speedup ≈ 3 × 0.75 / (3 × 0.25 + 0.75) = 2.25 / 1.50 = 1.5×
    """

    def __init__(self, backend: Any, config: SpeculativeConfig | None = None):
        """Initialize the self-speculative decoder.

        Locates model components (layers, embedding, final norm, lm_head),
        rotary embedding, and determines whether the model uses single or
        dual RoPE. Sets draft layer count either from config or automatically
        (total_layers // 4, minimum 1).

        Parameters
        ----------
        backend : AutoregressiveBackend
            The backend providing ``model``, ``devices``, etc. The model
            may be ``torch.compile``-wrapped -- this unwraps ``_orig_mod``
            to access internal attributes.
        config : SpeculativeConfig, optional
            Speculative decoding hyperparameters. Defaults to
            ``SpeculativeConfig()``.

        Side effects
        ------------
        Sets ``self.total_drafted``, ``self.total_accepted``,
        ``self.total_draft_ms``, ``self.total_verify_ms`` to 0.

        Raises
        ------
        AttributeError
            If model components (layers, embedding, norm, lm_head) cannot
            be located by the introspection helpers.
        """
        cfg = config or SpeculativeConfig()
        self.K = max(cfg.num_speculative_tokens, 1)
        self._loaded = False

        # ── Locate model components ──
        model = backend.model
        # If the model is torch.compiled, unwrap to access internals.
        raw = model
        if hasattr(model, "_orig_mod"):          # torch.compile wrapper
            raw = model._orig_mod
        self._inner_model = raw.model if hasattr(raw, "model") else raw
        self._layers = _find_model_layers(raw)
        self._embed = _find_embedding(raw)
        self._final_norm = _find_final_norm(raw)
        self._lm_head = _find_lm_head(raw)
        self._total_layers = len(self._layers)

        # ── Rotary embeddings (RoPE) — required by LLaMA/Gemma layers ──
        self._rotary_emb = None
        if hasattr(self._inner_model, "rotary_emb"):
            self._rotary_emb = self._inner_model.rotary_emb
        elif (hasattr(self._inner_model, "language_model")
              and hasattr(self._inner_model.language_model, "rotary_emb")):
            # Gemma3 multimodal: rotary_emb lives inside language_model
            self._rotary_emb = self._inner_model.language_model.rotary_emb

        # ── Detect dual-RoPE (Gemma3) vs single-RoPE (LLaMA/Gemma2) ──
        import inspect
        _first_layer = self._layers[0]
        _sig = inspect.signature(_first_layer.forward)
        self._needs_dual_rope = 'position_embeddings_global' in _sig.parameters

        # ── Compute draft layer count ──
        if cfg.num_draft_layers > 0:
            self._num_draft_layers = min(cfg.num_draft_layers, self._total_layers - 1)
        else:
            self._num_draft_layers = max(1, self._total_layers // 4)

        logger.info(
            "SelfSpeculativeDecoder: %d draft / %d verify layers, K=%d (RoPE=%s)",
            self._num_draft_layers,
            self._total_layers - self._num_draft_layers,
            self.K,
            self._rotary_emb is not None,
        )

        # ── Stats ──
        self.total_drafted: int = 0
        self.total_accepted: int = 0
        self.total_draft_ms: float = 0.0
        self.total_verify_ms: float = 0.0

    def load(self) -> None:
        """No-op — self-speculative uses the already-loaded main model."""
        self._loaded = True

    # ── Position embeddings helper ────────────────────────────────────────

    def _rope_embeddings(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin for the given positions."""
        if self._rotary_emb is None:
            return None
        cos, sin = self._rotary_emb(hidden_states, position_ids)
        return cos, sin

    def _layer_kwargs(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
        layer_kv=None, attention_mask=None,
    ) -> dict:
        """Build the keyword args dict for a layer forward call.

        ``layer_kv`` may be:
          - ``None`` — first forward, no pre-existing cache.
          - A ``DynamicCache`` — shared across draft steps; each layer
            reads/updates its own slot via ``self.layer_idx``.

        Handles both single-RoPE (LLaMA/Gemma2) and dual-RoPE (Gemma3)
        architectures.  Classic models receive ``position_embeddings``;
        Gemma3 receives ``position_embeddings_global`` AND
        ``position_embeddings_local`` — both set to the same (cos, sin)
        from the rotary embedding, since the decoder layer dispatches
        internally based on its attention pattern.
        """
        kwargs: dict = {}
        kwargs["use_cache"] = True
        kwargs["position_ids"] = position_ids
        kwargs["cache_position"] = position_ids[0]  # [seq_len]

        if layer_kv is not None:
            kwargs["past_key_value"] = layer_kv

        # RoPE position embeddings — compatible with both single and dual RoPE
        pe = self._rope_embeddings(hidden_states, position_ids)
        if pe is not None:
            if self._needs_dual_rope:
                kwargs["position_embeddings_global"] = pe
                kwargs["position_embeddings_local"] = pe
            else:
                kwargs["position_embeddings"] = pe

        # Attention mask — causal by default; only needed for multi-token verify
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        return kwargs

    @staticmethod
    def _full_cache_from_past(past_kv):
        """Build a fresh, independent KV-cache clone.

        Delegates to the module-level ``_clone_kv_cache()`` function.
        Kept as a static method for backward compatibility.
        """
        return _clone_kv_cache(past_kv)

    @staticmethod
    def _expand_cache_for_batch(past_kv, target_batch_size: int):
        """Expand a cache to *target_batch_size* by repeating KV entries.

        Delegates to the module-level ``_expand_cache_for_batch_fn()``.
        Kept as a static method for backward compatibility.
        """
        return _expand_cache_for_batch_fn(past_kv, target_batch_size)

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if the decoder is ready for translation.

        Returns
        -------
        bool
            ``True`` after ``load()`` completes, ``False`` otherwise.
        """
        return self._loaded

    @property
    def acceptance_rate(self) -> float:
        """Compute the cumulative token acceptance rate.

        Returns
        -------
        float
            ``total_accepted / total_drafted``, or 0.0 if no tokens have
            been drafted yet.
        """
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    @property
    def stats(self) -> dict:
        return {
            "mode": "self",
            "num_draft_layers": self._num_draft_layers,
            "total_layers": self._total_layers,
            "K": self.K,
            "total_drafted": self.total_drafted,
            "total_accepted": self.total_accepted,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "total_draft_ms": round(self.total_draft_ms, 2),
            "total_verify_ms": round(self.total_verify_ms, 2),
        }

    # ── Main entry point ─────────────────────────────────────────────────

    def translate_batch(self, batch: Any, backend: Any) -> "BatchGenerationOutput":
        """Translate a batch using self-speculative decoding.

        Each sequence is processed independently.  The per-sequence verify
        phase runs only layers[D:L] + norm + lm_head (not the full model),
        giving ~(D/L) × K compute savings per speculative step.
        """
        from benchmark.inference.backends.protocol import (
            BatchGenerationOutput, GenerationOutput,
        )
        from datetime import datetime, timezone

        device = backend.devices[0]
        tokenizer = backend.tokenizer
        max_new = backend.max_new_tokens
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or 0
        K = self.K

        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        B = input_ids.shape[0]

        wall_start = time.monotonic()
        total_draft_ms = 0.0
        total_verify_ms = 0.0

        generations: list[GenerationOutput] = []

        with torch.no_grad():
            for seq_idx in range(B):
                seq_start = time.monotonic()
                seq_ids = input_ids[seq_idx:seq_idx + 1]
                seq_mask = attention_mask[seq_idx:seq_idx + 1]

                # ── Prefill ──
                prefill_out = backend.model(
                    input_ids=seq_ids,
                    attention_mask=seq_mask,
                    use_cache=True,
                )
                past_kv = prefill_out.past_key_values
                generated_ids: list[int] = []
                seq_draft_ms = 0.0
                seq_verify_ms = 0.0

                seq_tokens_generated = 0
                next_token = seq_ids[:, -1:]
                prefill_len = seq_ids.shape[1]

                # Pre-allocate separate buffers — one for position IDs,
                # one for token inputs. These MUST NOT alias or the draft
                # loop will embed position numbers as token IDs.
                _pos_buf = torch.empty(1, 1, dtype=torch.long, device=device)
                _tok_buf = torch.empty(1, 1, dtype=torch.long, device=device)

                while seq_tokens_generated < max_new:
                    # ── DRAFT: layers[0:D], K steps ──
                    draft_start = time.monotonic()
                    draft_hidden_states: list[torch.Tensor] = []
                    draft_tokens: list[int] = []

                    current_input = next_token
                    current_kv = past_kv

                    for _k in range(K):
                        cur_pos = prefill_len + seq_tokens_generated + _k
                        _pos_buf[0, 0] = cur_pos
                        pos_ids = _pos_buf

                        hidden = self._embed(current_input)
                        draft_cache = self._full_cache_from_past(current_kv)
                        for layer_idx in range(self._num_draft_layers):
                            lkwargs = self._layer_kwargs(hidden, pos_ids, layer_kv=draft_cache)
                            layer_out = self._layers[layer_idx](hidden, **lkwargs)
                            hidden = layer_out[0]
                            if hidden.dim() == 2:
                                hidden = hidden.unsqueeze(1)

                        # Propagate the accumulated KV to the next draft step.
                        # Without this, every step clones from the prefill KV
                        # independently — the draft becomes K single-token
                        # predictions instead of a K-step autoregressive chain,
                        # destroying speculative acceptance rate.
                        current_kv = draft_cache

                        draft_logits = self._lm_head(self._final_norm(hidden))
                        next_tok = draft_logits[:, -1, :].argmax(dim=-1).item()
                        draft_hidden_states.append(hidden[:, -1, :])
                        draft_tokens.append(next_tok)
                        _tok_buf[0, 0] = next_tok
                        current_input = _tok_buf

                    draft_end = time.monotonic()
                    draft_step_ms = (draft_end - draft_start) * 1000.0
                    seq_draft_ms += draft_step_ms

                    # ── VERIFY: layers[D:L] + norm + lm_head ──
                    verify_start = time.monotonic()

                    verify_kv = self._full_cache_from_past(past_kv)
                    draft_hidden = torch.stack(draft_hidden_states, dim=1)
                    base_pos = prefill_len + seq_tokens_generated
                    verify_pos_ids = torch.arange(
                        base_pos, base_pos + K, device=device, dtype=torch.long,
                    ).unsqueeze(0)

                    for global_idx in range(self._num_draft_layers, self._total_layers):
                        layer = self._layers[global_idx]
                        lkwargs = self._layer_kwargs(draft_hidden, verify_pos_ids, layer_kv=verify_kv)
                        layer_out = layer(draft_hidden, **lkwargs)
                        draft_hidden = layer_out[0]
                        if draft_hidden.dim() == 2:
                            draft_hidden = draft_hidden.unsqueeze(1)

                    verify_logits = self._lm_head(self._final_norm(draft_hidden))
                    verify_logits = verify_logits.squeeze(0)

                    verify_end = time.monotonic()
                    verify_step_ms = (verify_end - verify_start) * 1000.0
                    seq_verify_ms += verify_step_ms

                    # ── ACCEPT (GPU-side comparison — no per-token .item() syncs) ──
                    main_preds = verify_logits.argmax(dim=-1)  # [K]

                    # Build draft tensor on GPU for vectorized comparison.
                    draft_t = torch.tensor(draft_tokens, device=device, dtype=torch.long)
                    matches = main_preds == draft_t  # [K] bool — single GPU op

                    n_accepted_this_round = 0
                    for k in range(K):
                        gen_id = draft_tokens[k]
                        if matches[k].item():  # bool .item() is cheap (single byte D2H)
                            generated_ids.append(gen_id)
                            n_accepted_this_round += 1
                            seq_tokens_generated += 1
                            if gen_id == eos_id or seq_tokens_generated >= max_new:
                                break
                        else:
                            generated_ids.append(main_preds[k].item())
                            n_accepted_this_round += 1
                            seq_tokens_generated += 1
                            break

                    self.total_drafted += K
                    self.total_accepted += n_accepted_this_round

                    if n_accepted_this_round > 0:
                        new_len = prefill_len + len(generated_ids)
                        past_kv = _safe_crop_kv(verify_kv, new_len)

                    if generated_ids and generated_ids[-1] == eos_id:
                        break
                    if seq_tokens_generated >= max_new:
                        break
                    if n_accepted_this_round == 0:
                        break

                    _tok_buf[0, 0] = generated_ids[-1]
                    next_token = _tok_buf

                # ── Decode ──
                seq_end = time.monotonic()
                translated_text = ""
                if generated_ids:
                    translated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                seq_latency_ms = (seq_end - seq_start) * 1000.0
                total_draft_ms += seq_draft_ms
                total_verify_ms += seq_verify_ms

                src = batch.raw_texts[seq_idx] if hasattr(batch, 'raw_texts') and seq_idx < len(batch.raw_texts) else ""
                in_tok = len(batch.input_ids[seq_idx]) if hasattr(batch, 'input_ids') and seq_idx < len(batch.input_ids) else 0

                generations.append(GenerationOutput(
                    input_text=src, translated_text=translated_text,
                    input_tokens=in_tok, output_tokens=len(generated_ids),
                    total_latency_ms=seq_latency_ms,
                    phase_timings={
                        "draft_ms": round(seq_draft_ms, 2),
                        "verify_ms": round(seq_verify_ms, 2),
                        "acceptance_rate": round(self.total_accepted / max(self.total_drafted, 1), 4),
                    },
                    timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ))

        wall_end = time.monotonic()
        total_wall_ms = (wall_end - wall_start) * 1000.0

        self.total_draft_ms += total_draft_ms
        self.total_verify_ms += total_verify_ms

        total_out = sum(r.output_tokens for r in generations)
        total_in = sum(r.input_tokens for r in generations)

        return BatchGenerationOutput(
            batch_id=batch.batch_id if hasattr(batch, 'batch_id') else 0,
            generations=generations, batch_size=B,
            input_tokens_total=total_in, output_tokens_total=total_out,
            total_latency_ms=round(total_wall_ms, 2),
            phase_timings={
                "draft_ms": round(total_draft_ms, 2),
                "verify_ms": round(total_verify_ms, 2),
                "acceptance_rate": round(self.acceptance_rate, 4),
                "method": "self_speculative_layers_d_to_l",
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# Draft-Model Speculative Decoder — separate small model as draft
# ═════════════════════════════════════════════════════════════════════════════


class DraftModelSpeculativeDecoder(SpeculativeDecoder):
    """Speculative decoding with a separate small draft model.

    Loads a small model (e.g. 135M params) alongside the main model.
    The draft generates K candidate tokens autoregressively; the main
    model verifies all K in one forward pass.

    **Requires the draft and main models to share a tokenizer.**
    Token IDs are compared directly during verification, so a mismatch
    produces meaningless results.

    Performance model
    -----------------
    With a draft model D× smaller than the main model:
      - Draft cost:  K single-token forwards on the small model
      - Verify cost: 1 batched forward on the main model
      - Speedup:     K × acceptance_rate / (1 + K / D)

    For a 135M draft with a 4B main (D≈30), K=5, acceptance=0.8:
      - Speedup ≈ 5 × 0.8 / (1 + 5/30) ≈ 4.0 / 1.17 ≈ 3.4×
    """

    def __init__(self, backend: Any, config: SpeculativeConfig | None = None):
        """Initialize the draft-model speculative decoder.

        Validates that ``draft_model_name`` is set in config and initializes
        statistics counters.

        Parameters
        ----------
        backend : AutoregressiveBackend
            The backend providing the main model, tokenizer, and device.
        config : SpeculativeConfig, optional
            Speculative decoding hyperparameters. Must have
            ``draft_model_name`` set to a valid HuggingFace model ID.

        Raises
        ------
        ValueError
            If ``config.draft_model_name`` is empty -- a draft model name
            is required for draft_model mode.

        Side effects
        ------------
        Sets ``self.total_drafted``, ``self.total_accepted``,
        ``self.total_draft_ms``, ``self.total_verify_ms`` to 0.
        """
        cfg = config or SpeculativeConfig()
        self.backend = backend
        self.config = cfg
        self.draft_model: Optional[Any] = None  # PreTrainedModel
        self._loaded = False
        self.K = max(cfg.num_speculative_tokens, 1)

        if not cfg.draft_model_name:
            raise ValueError(
                "speculative_mode='draft_model' requires "
                "speculative_draft_model to be set (e.g. a small model "
                "sharing the main model's tokenizer)."
            )

        # ── Stats ──
        self.total_drafted: int = 0
        self.total_accepted: int = 0
        self.total_draft_ms: float = 0.0
        self.total_verify_ms: float = 0.0

    def load(self) -> None:
        """Load the draft model."""
        if self.draft_model is not None:
            return

        device = self.backend.devices[0]
        model_name = self.config.draft_model_name
        logger.info(
            "Loading draft model: %s (device=%s, K=%d)",
            model_name, device, self.K,
        )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ── Tokenizer compatibility check ──
        main_vocab = self.backend.tokenizer.get_vocab()
        draft_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False,
        )
        draft_vocab = draft_tokenizer.get_vocab()

        if main_vocab.keys() != draft_vocab.keys():
            n_shared = len(set(main_vocab.keys()) & set(draft_vocab.keys()))
            n_main = len(main_vocab)
            raise ValueError(
                f"Draft model tokenizer incompatible with main model: "
                f"{n_shared}/{n_main} tokens overlap "
                f"({100 * n_shared / max(n_main, 1):.1f}%). "
                f"Speculative decoding with draft_model requires "
                f"identical vocabularies."
            )
        else:
            logger.info("Draft tokenizer verified: vocabulary matches main model.")

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.backend.precision_config.master_dtype
            if self.backend.precision_config else torch.bfloat16,
            trust_remote_code=False,
        )
        self.draft_model = self.draft_model.to(device)
        self.draft_model.eval()
        self._loaded = True

        n_params = sum(p.numel() for p in self.draft_model.parameters())
        logger.info(
            "Draft model loaded: %.0fM params, K=%d",
            n_params / 1e6, self.K,
        )

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if the draft model has been loaded.

        Returns
        -------
        bool
            ``True`` after ``load()`` completes successfully,
            ``False`` otherwise.
        """
        return self._loaded

    @property
    def acceptance_rate(self) -> float:
        """Compute the cumulative token acceptance rate.

        Returns
        -------
        float
            ``total_accepted / total_drafted``, or 0.0 if no tokens have
            been drafted yet.
        """
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    @property
    def stats(self) -> dict:
        return {
            "mode": "draft_model",
            "draft_model": self.config.draft_model_name,
            "K": self.K,
            "total_drafted": self.total_drafted,
            "total_accepted": self.total_accepted,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "total_draft_ms": round(self.total_draft_ms, 2),
            "total_verify_ms": round(self.total_verify_ms, 2),
        }

    def translate_batch(self, batch: Any, backend: Any) -> "BatchGenerationOutput":
        """Translate using a separate draft model."""
        from benchmark.inference.backends.protocol import (
            BatchGenerationOutput, GenerationOutput,
        )
        from datetime import datetime, timezone

        if not self._loaded:
            raise RuntimeError("Draft model not loaded. Call load() first.")

        device = backend.devices[0]
        tokenizer = backend.tokenizer
        max_new = backend.max_new_tokens
        K = self.K
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id or 0

        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        B = input_ids.shape[0]

        wall_start = time.monotonic()
        total_draft_ms = 0.0
        total_verify_ms = 0.0
        generations: list[GenerationOutput] = []

        with torch.no_grad():
            for seq_idx in range(B):
                seq_start = time.monotonic()
                seq_ids = input_ids[seq_idx:seq_idx + 1]
                seq_mask = attention_mask[seq_idx:seq_idx + 1]
                seq_draft_ms = 0.0
                seq_verify_ms = 0.0

                # ── Prefill ──
                main_out = backend.model(
                    input_ids=seq_ids,
                    attention_mask=seq_mask,
                    use_cache=True,
                )
                past_kv = main_out.past_key_values
                generated_ids: list[int] = []

                seq_tokens_generated = 0
                _tok_buf = torch.empty(1, 1, dtype=torch.long, device=device)
                while seq_tokens_generated < max_new:
                    # ── Draft phase ──
                    draft_start = time.monotonic()
                    draft_input = seq_ids[:, -1:]  # last prompt token
                    draft_tokens_list: list[int] = []

                    for _k in range(K):
                        draft_out = self.draft_model(
                            input_ids=draft_input,
                            use_cache=False,
                        )
                        draft_logits = draft_out.logits[:, -1, :]
                        next_tok = draft_logits.argmax(dim=-1).item()
                        draft_tokens_list.append(next_tok)
                        _tok_buf[0, 0] = next_tok
                        draft_input = _tok_buf

                    draft_end = time.monotonic()
                    draft_step_ms = (draft_end - draft_start) * 1000.0
                    seq_draft_ms += draft_step_ms

                    # ── Verify phase ──
                    verify_start = time.monotonic()
                    candidates = torch.tensor(
                        [draft_tokens_list], device=device, dtype=torch.long,
                    )  # [1, K]

                    # Clone past_kv before verify to prevent mutation.
                    # DynamicCache.update() modifies the cache in-place even
                    # when use_cache=False in some HF versions.
                    verify_kv = _clone_kv_cache(past_kv)

                    verify_out = backend.model(
                        input_ids=candidates,
                        past_key_values=verify_kv,
                        use_cache=True,
                    )
                    verify_logits = verify_out.logits  # [1, K, vocab]

                    verify_end = time.monotonic()
                    verify_step_ms = (verify_end - verify_start) * 1000.0
                    seq_verify_ms += verify_step_ms

                    # ── Accept phase (GPU-side comparison) ──
                    main_preds = verify_logits[0, :K, :].argmax(dim=-1)  # [K]
                    draft_t = torch.tensor(draft_tokens_list, device=device, dtype=torch.long)
                    matches = main_preds == draft_t  # [K] — single GPU op

                    n_accepted_this_round = 0
                    for k in range(K):
                        if matches[k].item():  # bool .item() — single byte
                            generated_ids.append(draft_tokens_list[k])
                            n_accepted_this_round += 1
                            seq_tokens_generated += 1
                            if draft_tokens_list[k] == eos_id or seq_tokens_generated >= max_new:
                                break
                        else:
                            generated_ids.append(main_preds[k].item())
                            n_accepted_this_round += 1
                            seq_tokens_generated += 1
                            break

                    self.total_drafted += K
                    self.total_accepted += n_accepted_this_round

                    # Update KV-cache: crop verify output to only accepted
                    # tokens to prevent cache pollution from unaccepted drafts.
                    if n_accepted_this_round > 0:
                        new_len = seq_ids.shape[1] + len(generated_ids)
                        past_kv = _safe_crop_kv(verify_kv, new_len)

                    if generated_ids and generated_ids[-1] == eos_id:
                        break
                    if seq_tokens_generated >= max_new:
                        break
                    if n_accepted_this_round == 0:
                        break

                # ── Decode ──
                seq_end = time.monotonic()
                translated_text = ""
                if generated_ids:
                    translated_text = tokenizer.decode(
                        generated_ids, skip_special_tokens=True,
                    ).strip()

                seq_latency_ms = (seq_end - seq_start) * 1000.0
                total_draft_ms += seq_draft_ms
                total_verify_ms += seq_verify_ms

                src = (
                    batch.raw_texts[seq_idx]
                    if hasattr(batch, 'raw_texts') and seq_idx < len(batch.raw_texts)
                    else ""
                )
                in_tok = (
                    len(batch.input_ids[seq_idx])
                    if hasattr(batch, 'input_ids') and seq_idx < len(batch.input_ids)
                    else 0
                )

                generations.append(GenerationOutput(
                    input_text=src,
                    translated_text=translated_text,
                    input_tokens=in_tok,
                    output_tokens=len(generated_ids),
                    total_latency_ms=seq_latency_ms,
                    phase_timings={
                        "draft_ms": round(seq_draft_ms, 2),
                        "verify_ms": round(seq_verify_ms, 2),
                        "acceptance_rate": round(
                            self.total_accepted / max(self.total_drafted, 1), 4,
                        ),
                    },
                    timestamp_utc=datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                ))

        wall_end = time.monotonic()
        total_wall_ms = (wall_end - wall_start) * 1000.0

        self.total_draft_ms += total_draft_ms
        self.total_verify_ms += total_verify_ms

        total_out = sum(r.output_tokens for r in generations)
        total_in = sum(r.input_tokens for r in generations)

        return BatchGenerationOutput(
            batch_id=batch.batch_id if hasattr(batch, 'batch_id') else 0,
            generations=generations,
            batch_size=B,
            input_tokens_total=total_in,
            output_tokens_total=total_out,
            total_latency_ms=round(total_wall_ms, 2),
            phase_timings={
                "draft_ms": round(total_draft_ms, 2),
                "verify_ms": round(total_verify_ms, 2),
                "acceptance_rate": round(self.acceptance_rate, 4),
                "method": "draft_model_speculative",
            },
        )


# ── Factory ──────────────────────────────────────────────────────────────────


def create_speculative_decoder(
    backend: Any,
    mode: str = "self",
    draft_model_name: str = "",
    num_speculative_tokens: int = 3,
    num_draft_layers: int = 0,
) -> Optional[SpeculativeDecoder]:
    """Create the appropriate speculative decoder for the given mode.

    Parameters
    ----------
    backend : AutoregressiveBackend
        The backend providing the main model.
    mode : str
        ``"self"`` or ``"draft_model"``.
    draft_model_name : str
        HuggingFace model ID for draft model (draft_model mode only).
    num_speculative_tokens : int
        K — number of speculative tokens per step.
    num_draft_layers : int
        For self-speculative: number of early layers used as draft.
        0 = auto-compute.

    Returns
    -------
    SpeculativeDecoder or None
        Returns the decoder instance.  When speculative mode is not applicable
        (e.g., non-CUDA backend), returns ``None``.
    """
    import os

    config = SpeculativeConfig(
        mode=mode,
        draft_model_name=draft_model_name,
        num_speculative_tokens=num_speculative_tokens,
        num_draft_layers=num_draft_layers,
    )

    if mode == "self":
        return SelfSpeculativeDecoder(backend, config)
    elif mode == "draft_model":
        return DraftModelSpeculativeDecoder(backend, config)
    else:
        raise ValueError(
            f"Unknown speculative mode: {mode!r}. "
            f"Use 'self' or 'draft_model'."
        )
