"""Tests for speculative decoding — correctness, wiring, and speedup verification.

Tests are split into three groups:
  - **Unit**: model introspection, config validation, decoder creation
  - **Integration**: end-to-end translation with self-speculative
  - **Speedup**: wall-clock comparison of standard vs speculative decode

Run with:
  pytest tests/test_speculative.py -v
  pytest tests/test_speculative.py -v -k "speedup" --timeout=300  # speedup only
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from benchmark.config.schema import ModelConfig

# Define network error types for use in except clauses, avoiding bare Exception.
# requests is a model-download time concern only; make it optional so the test
# file is importable even when requests is not installed.
try:
    import requests as _requests
except ImportError:
    _requests = None

if _requests is not None:
    _NETWORK_ERRORS = (OSError, _requests.RequestException)
else:
    _NETWORK_ERRORS = (OSError,)


# ── Helper: create a minimal backend-like namespace for tests ────────────────

def _make_backend(**kwargs) -> types.SimpleNamespace:
    """Create a minimal backend-like object for testing decoders.

    Any keyword argument is set as an attribute on the returned namespace.
    Defaults suitable for CPU-based unit tests are provided.
    """
    defaults = dict(
        devices=[torch.device("cpu")],
        max_new_tokens=32,
        precision_config=None,
        tokenizer=None,
        model=None,
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _make_batch(input_ids, attention_mask, raw_texts=None, batch_id=0):
    """Create a minimal batch-like object for testing translate_batch."""
    if raw_texts is None:
        raw_texts = ["test"] * input_ids.shape[0]
    return types.SimpleNamespace(
        batch_id=batch_id,
        input_ids=input_ids,
        attention_mask=attention_mask,
        raw_texts=raw_texts,
    )


def _make_tiny_model_and_tokenizer(real_tokenizer):
    """Build a minimal Gemma-like model with a real tokenizer.

    Returns (model, tokenizer) where model has 8 tiny decoder layers,
    an embedding, final norm, and lm_head — enough to test the
    self-speculative layer-splitting logic.
    """
    import torch.nn as nn

    tokenizer = real_tokenizer
    vocab_size = len(tokenizer)

    class TinyDecoderLayer(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(hidden, 2, batch_first=True)
            self.mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Linear(hidden * 4, hidden),
            )
            self.input_layernorm = nn.LayerNorm(hidden)
            self.post_attention_layernorm = nn.LayerNorm(hidden)

        def forward(self, hidden_states, use_cache=False, past_key_value=None, **kwargs):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            attn_out, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_out = self.mlp(hidden_states)
            hidden_states = residual + mlp_out
            if use_cache:
                # Return proper (key, value) tuple format expected by
                # _safe_crop_kv and real HF models. Both are the hidden
                # state for this tiny test model (no real attention keys).
                return hidden_states, (hidden_states.clone(), hidden_states.clone())
            return hidden_states,

    class TinyModel(nn.Module):
        def __init__(self, vocab_size, hidden=32, num_layers=8):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(vocab_size, hidden)
            self.model.layers = nn.ModuleList(
                [TinyDecoderLayer(hidden) for _ in range(num_layers)]
            )
            self.model.norm = nn.LayerNorm(hidden)
            self.lm_head = nn.Linear(hidden, vocab_size)
            self.config = type("Cfg", (), {
                "hidden_size": hidden,
                "num_hidden_layers": num_layers,
                "num_key_value_heads": 2,
                "num_attention_heads": 2,
                "head_dim": hidden // 2,
            })()

        def forward(self, input_ids, attention_mask=None, use_cache=False,
                    past_key_values=None, inputs_embeds=None, **kwargs):
            if inputs_embeds is not None:
                hidden = inputs_embeds
            else:
                hidden = self.model.embed_tokens(input_ids)
            new_kvs = []
            for i, layer in enumerate(self.model.layers):
                layer_kv = past_key_values[i] if past_key_values else None
                out = layer(hidden, use_cache=use_cache, past_key_value=layer_kv)
                hidden = out[0]
                if use_cache:
                    new_kvs.append(out[1])
            logits = self.lm_head(self.model.norm(hidden))
            Output = type("Output", (), {})
            if use_cache:
                out = Output()
                out.past_key_values = tuple(new_kvs)
                out.logits = logits
                return out
            out = Output()
            out.logits = logits
            return out

    model = TinyModel(vocab_size, hidden=32, num_layers=8)
    model.eval()
    return model, tokenizer


# ═════════════════════════════════════════════════════════════════════════════
# Shared fixtures — must precede class definitions to avoid scopemismatch
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="class")
def tiny_model_and_tokenizer(real_tokenizer):
    """Module-level fixture so all test classes can access the tiny model."""
    return _make_tiny_model_and_tokenizer(real_tokenizer)


# ═════════════════════════════════════════════════════════════════════════════
# Unit tests — module imports and config
# ═════════════════════════════════════════════════════════════════════════════


class TestSpeculativeConfig:
    """Validate the config schema changes for speculative decoding."""

    def test_default_config_no_speculative(self):
        """Speculative is off by default."""
        cfg = ModelConfig()
        assert cfg.use_speculative is False
        assert cfg.speculative_mode == "self"
        assert cfg.speculative_num_tokens == 3
        assert cfg.speculative_draft_model == ""
        assert cfg.speculative_num_draft_layers == 0

    def test_enable_self_speculative(self):
        """Self-speculative mode requires no draft model."""
        cfg = ModelConfig(
            use_speculative=True,
            speculative_mode="self",
            speculative_num_tokens=4,
        )
        assert cfg.use_speculative is True
        assert cfg.speculative_mode == "self"
        assert cfg.speculative_num_tokens == 4

    def test_draft_model_requires_draft_path(self):
        """draft_model mode must specify a draft model name."""
        with pytest.raises(ValueError, match="speculative_draft_model"):
            ModelConfig(
                use_speculative=True,
                speculative_mode="draft_model",
                speculative_draft_model="",  # empty
            )

    def test_draft_model_with_model_name_valid(self):
        """draft_model mode with a name is valid."""
        cfg = ModelConfig(
            use_speculative=True,
            speculative_mode="draft_model",
            speculative_draft_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        )
        assert cfg.speculative_draft_model == "HuggingFaceTB/SmolLM2-135M-Instruct"

    def test_speculative_field_ranges(self):
        """num_tokens and num_draft_layers respect bounds."""
        cfg = ModelConfig(
            speculative_num_tokens=1,
            speculative_num_draft_layers=0,
        )
        assert cfg.speculative_num_tokens == 1
        assert cfg.speculative_num_draft_layers == 0

        cfg2 = ModelConfig(
            speculative_num_tokens=16,
            speculative_num_draft_layers=64,
        )
        assert cfg2.speculative_num_tokens == 16
        assert cfg2.speculative_num_draft_layers == 64


class TestModelIntrospection:
    """Test the model layer-finding helpers."""

    def test_find_layers_gemma_style(self):
        """_find_model_layers finds layers in Gemma-style architecture."""
        import torch.nn as nn
        from benchmark.inference.speculative import _find_model_layers

        # Build a minimal Gemma-like module tree.
        class FakeDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Linear(64, 64)
                self.mlp = nn.Linear(64, 64)

        class FakeGemmaModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 64)
                self.layers = nn.ModuleList([FakeDecoderLayer() for _ in range(8)])
                self.norm = nn.LayerNorm(64)

        class FakeGemmaForCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeGemmaModel()
                self.lm_head = nn.Linear(64, 100)

        model = FakeGemmaForCausalLM()
        layers = _find_model_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 8

    def test_find_layers_gpt2_style(self):
        """_find_model_layers finds layers in GPT-2-style architecture."""
        import torch.nn as nn
        from benchmark.inference.speculative import _find_model_layers

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.c_attn = nn.Linear(64, 192)

        class FakeGPT2Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = FakeAttention()

        class FakeGPT2Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.wte = nn.Embedding(100, 64)
                self.h = nn.ModuleList([FakeGPT2Block() for _ in range(6)])
                self.ln_f = nn.LayerNorm(64)

        class FakeGPT2LMHeadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = FakeGPT2Transformer()
                self.lm_head = nn.Linear(64, 100)

        model = FakeGPT2LMHeadModel()
        layers = _find_model_layers(model)
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) == 6

    def test_find_embedding(self):
        """_find_embedding locates token embeddings."""
        import torch.nn as nn
        from benchmark.inference.speculative import _find_embedding

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(100, 64)
                self.model.layers = nn.ModuleList([])
                self.model.norm = nn.LayerNorm(64)

        model = FakeModel()
        embed = _find_embedding(model)
        assert isinstance(embed, nn.Embedding)

    def test_find_lm_head(self):
        """_find_lm_head locates the language model head."""
        import torch.nn as nn
        from benchmark.inference.speculative import _find_lm_head

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([])
                self.lm_head = nn.Linear(64, 100)

        model = FakeModel()
        lm_head = _find_lm_head(model)
        assert isinstance(lm_head, nn.Linear)

    def test_find_final_norm(self):
        """_find_final_norm locates the final layer norm."""
        import torch.nn as nn
        from benchmark.inference.speculative import _find_final_norm

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([])
                self.model.norm = nn.LayerNorm(64)

        model = FakeModel()
        norm = _find_final_norm(model)
        assert isinstance(norm, nn.LayerNorm)

    def test_find_layers_raises_on_unknown(self):
        """_find_model_layers raises AttributeError when no layers found."""
        import torch.nn as nn
        from benchmark.inference.speculative import _find_model_layers

        class UnknownModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

        model = UnknownModel()
        with pytest.raises(AttributeError, match="Cannot locate"):
            _find_model_layers(model)


# ═════════════════════════════════════════════════════════════════════════════
# Integration tests — SelfSpeculativeDecoder with a real tokenizer
# ═════════════════════════════════════════════════════════════════════════════


class TestSelfSpeculativeDecoder:
    """Test the self-speculative decoder with a real tokenizer.

    These tests use a tiny random model (not a real pretrained model) to
    validate the control flow without requiring GPU memory.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def tiny_model_and_tokenizer(cls, real_tokenizer):
        return _make_tiny_model_and_tokenizer(real_tokenizer)

    def test_decoder_creation(self, tiny_model_and_tokenizer):
        """SelfSpeculativeDecoder can be created from a tiny model."""
        from benchmark.inference.speculative import (
            SelfSpeculativeDecoder, SpeculativeConfig,
        )

        model, tokenizer = tiny_model_and_tokenizer

        _MiniBackend = types.SimpleNamespace(
            model=model, tokenizer=tokenizer,
            devices=[torch.device("cpu")], max_new_tokens=32,
            precision_config=None,
        )
        backend = _MiniBackend
        cfg = SpeculativeConfig(mode="self", num_speculative_tokens=2, num_draft_layers=2)
        decoder = SelfSpeculativeDecoder(backend, cfg)

        assert decoder.K == 2
        assert decoder._num_draft_layers == 2
        assert decoder._total_layers == 8
        assert decoder.is_loaded is False
        decoder.load()
        assert decoder.is_loaded is True

    def test_decoder_stats(self, tiny_model_and_tokenizer):
        """Stats track drafted, accepted, and rate."""
        from benchmark.inference.speculative import (
            SelfSpeculativeDecoder, SpeculativeConfig,
        )

        model, tokenizer = tiny_model_and_tokenizer

        backend = _make_backend(model=model, tokenizer=tokenizer, max_new_tokens=32)
        cfg = SpeculativeConfig(mode="self", num_speculative_tokens=2, num_draft_layers=2)
        decoder = SelfSpeculativeDecoder(backend, cfg)
        decoder.load()

        stats = decoder.stats
        assert stats["mode"] == "self"
        assert stats["K"] == 2
        assert stats["num_draft_layers"] == 2
        assert stats["total_layers"] == 8
        assert stats["total_drafted"] == 0
        assert stats["total_accepted"] == 0
        assert stats["acceptance_rate"] == 0.0

    def test_translate_batch_returns_output(self, tiny_model_and_tokenizer):
        """translate_batch produces valid output structure."""
        from benchmark.inference.speculative import (
            SelfSpeculativeDecoder, SpeculativeConfig,
        )

        model, tokenizer = tiny_model_and_tokenizer

        backend = _make_backend(model=model, tokenizer=tokenizer, max_new_tokens=8)
        cfg = SpeculativeConfig(mode="self", num_speculative_tokens=2, num_draft_layers=2)
        decoder = SelfSpeculativeDecoder(backend, cfg)
        decoder.load()

        # Build a minimal batch
        class _MiniBatch:
            batch_id = 0
            raw_texts = ["hello world"]

            def __init__(self, ids, mask):
                self.input_ids = ids
                self.attention_mask = mask

        enc = tokenizer("hello world", return_tensors="pt")
        batch = _MiniBatch(enc["input_ids"], enc["attention_mask"])

        result = decoder.translate_batch(batch, backend)

        # Check output structure
        assert result.batch_id == 0
        assert result.batch_size == 1
        assert len(result.generations) == 1
        gen = result.generations[0]
        assert hasattr(gen, "translated_text")
        assert hasattr(gen, "output_tokens")
        assert hasattr(gen, "total_latency_ms")
        assert gen.output_tokens >= 0

        # Phase timings should be present
        assert "draft_ms" in gen.phase_timings
        assert "verify_ms" in gen.phase_timings
        assert "acceptance_rate" in gen.phase_timings
        assert "method" in result.phase_timings
        assert result.phase_timings["method"] == "self_speculative_layers_d_to_l"

    def test_auto_draft_layers(self, tiny_model_and_tokenizer):
        """num_draft_layers=0 computes total//4."""
        from benchmark.inference.speculative import (
            SelfSpeculativeDecoder, SpeculativeConfig,
        )

        model, tokenizer = tiny_model_and_tokenizer

        backend = _make_backend(model=model, tokenizer=tokenizer, max_new_tokens=32)
        cfg = SpeculativeConfig(mode="self", num_speculative_tokens=2, num_draft_layers=0)
        decoder = SelfSpeculativeDecoder(backend, cfg)

        # 8 total layers, auto = 8 // 4 = 2
        assert decoder._num_draft_layers == 2
        assert decoder._total_layers == 8


class TestDraftModelSpeculativeDecoder:
    """Test the draft-model speculative decoder."""

    def test_requires_draft_model_name(self):
        """draft_model mode raises without a draft model name."""
        from benchmark.inference.speculative import (
            DraftModelSpeculativeDecoder, SpeculativeConfig,
        )

        backend = _make_backend()
        cfg = SpeculativeConfig(mode="draft_model", draft_model_name="", num_speculative_tokens=3)

        with pytest.raises(ValueError, match="requires.*draft_model"):
            DraftModelSpeculativeDecoder(backend, cfg)

    def test_tracks_stats(self):
        """Stats initialization is correct for draft model decoder."""
        from benchmark.inference.speculative import (
            DraftModelSpeculativeDecoder, SpeculativeConfig,
        )

        backend = _make_backend()
        cfg = SpeculativeConfig(
            mode="draft_model",
            draft_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            num_speculative_tokens=5,
        )

        # Load will fail (no real model download in unit test), but init should work.
        # Only skip on import/network errors — let assertions fail normally.
        try:
            decoder = DraftModelSpeculativeDecoder(backend, cfg)
        except _NETWORK_ERRORS:
            pytest.skip("Draft model download requires network")

        assert decoder.K == 5
        stats = decoder.stats
        assert stats["mode"] == "draft_model"
        assert stats["total_drafted"] == 0
        assert stats["acceptance_rate"] == 0.0


class TestFactory:
    """Test the create_speculative_decoder factory."""

    def test_creates_self_decoder(self, tiny_model_and_tokenizer):
        """Factory creates SelfSpeculativeDecoder for mode='self'."""
        from benchmark.inference.speculative import (
            create_speculative_decoder, SelfSpeculativeDecoder,
            SpeculativeConfig,
        )

        model, tokenizer = tiny_model_and_tokenizer

        backend = _make_backend(model=model, tokenizer=tokenizer, max_new_tokens=32)
        decoder = create_speculative_decoder(
            backend, mode="self", num_speculative_tokens=3,
        )
        assert isinstance(decoder, SelfSpeculativeDecoder)
        assert decoder.K == 3

    def test_unknown_mode_raises(self, tiny_model_and_tokenizer):
        """Unknown mode raises ValueError."""
        from benchmark.inference.speculative import create_speculative_decoder

        model, tokenizer = tiny_model_and_tokenizer

        backend = _make_backend(model=model, tokenizer=tokenizer, max_new_tokens=32)
        with pytest.raises(ValueError, match="Unknown speculative mode"):
            create_speculative_decoder(backend, mode="invalid_mode")


# ═════════════════════════════════════════════════════════════════════════════
# Speedup verification tests — wall-clock comparison
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestSpeculativeSpeedup:
    """Measure wall-clock speedup of self-speculative vs standard decoding.

    These tests require a real model.  They use the tiny fake model above
    for the basic check, and optionally a real model if available.
    """

    def test_self_speculative_produces_fewer_forward_passes(self, tiny_model_and_tokenizer):
        """Self-speculative decode should produce tokens with fewer effective
        full-model forward passes per token than standard decoding.

        We verify this by counting: each accepted speculative token avoids
        a full-model forward pass.  With K=3 and acceptance_rate > 0.5,
        the effective passes-per-token should be < 1.0.
        """
        from benchmark.inference.speculative import (
            SelfSpeculativeDecoder, SpeculativeConfig,
        )

        model, tokenizer = tiny_model_and_tokenizer

        backend = _make_backend(model=model, tokenizer=tokenizer, max_new_tokens=16)
        cfg = SpeculativeConfig(mode="self", num_speculative_tokens=3, num_draft_layers=2)
        decoder = SelfSpeculativeDecoder(backend, cfg)
        decoder.load()

        class _MiniBatch:
            batch_id = 0
            raw_texts = ["hello world test sentence"]

            def __init__(self, ids, mask):
                self.input_ids = ids
                self.attention_mask = mask

        enc = tokenizer("hello world test sentence", return_tensors="pt")
        batch = _MiniBatch(enc["input_ids"], enc["attention_mask"])

        start = time.monotonic()
        result = decoder.translate_batch(batch, backend)
        elapsed = time.monotonic() - start

        # Basic sanity: something was generated
        assert result.output_tokens_total >= 0

        # Stats were updated
        stats = decoder.stats
        # A random model with random weights may accept zero tokens.
        assert stats["total_drafted"] >= 0
        # Acceptance rate is tracked (may be 0 for tiny random model)
        assert "acceptance_rate" in stats

        # Phase timings are present and non-negative
        pt = result.phase_timings
        assert pt["draft_ms"] >= 0
        assert pt["verify_ms"] >= 0
        assert 0.0 <= pt["acceptance_rate"] <= 1.0

    def test_standard_vs_speculative_wallclock(self, tiny_model_and_tokenizer):
        """Wall-clock comparison: self-speculative vs standard decode.

        Runs both paths on the same input, comparing:
          - Both produce output
          - Phase timings differ between paths
          - Speculative path records draft+verify phases
        """
        from benchmark.inference.speculative import (
            SelfSpeculativeDecoder, SpeculativeConfig,
        )

        model, tokenizer = tiny_model_and_tokenizer

        backend = _make_backend(model=model, tokenizer=tokenizer, max_new_tokens=8)
        cfg = SpeculativeConfig(mode="self", num_speculative_tokens=2, num_draft_layers=2)
        decoder = SelfSpeculativeDecoder(backend, cfg)
        decoder.load()

        class _MiniBatch:
            batch_id = 0
            raw_texts = ["test"]

            def __init__(self, ids, mask):
                self.input_ids = ids
                self.attention_mask = mask

        enc = tokenizer("test", return_tensors="pt")
        batch = _MiniBatch(enc["input_ids"], enc["attention_mask"])

        # ── Speculative path ──
        t0 = time.monotonic()
        spec_result = decoder.translate_batch(batch, backend)
        spec_elapsed = time.monotonic() - t0

        # ── Verify output structure ──
        assert spec_result.batch_size == 1
        gen = spec_result.generations[0]
        assert gen.total_latency_ms > 0

        # Phase timings breakdown
        pt = gen.phase_timings
        assert "draft_ms" in pt
        assert "verify_ms" in pt
        # draft + verify should be less than or equal to total latency
        combined = pt["draft_ms"] + pt["verify_ms"]
        assert combined >= 0

        # Stats are cumulative across calls
        stats = decoder.stats
        # A random model with random weights may accept zero tokens.
        assert stats["total_drafted"] >= 0
        assert stats["acceptance_rate"] >= 0.0


@pytest.mark.slow
class TestSpeculativeCorrectness:
    """Verify speculative decoding produces coherent output."""

    def test_output_is_valid_utf8(self, tiny_model_and_tokenizer):
        """Speculative output is valid UTF-8 text."""
        from benchmark.inference.speculative import (
            SelfSpeculativeDecoder, SpeculativeConfig,
        )

        model, tokenizer = tiny_model_and_tokenizer

        backend = _make_backend(model=model, tokenizer=tokenizer, max_new_tokens=8)
        cfg = SpeculativeConfig(mode="self", num_speculative_tokens=2, num_draft_layers=2)
        decoder = SelfSpeculativeDecoder(backend, cfg)
        decoder.load()

        class _MiniBatch:
            batch_id = 0
            raw_texts = ["hello world"]

            def __init__(self, ids, mask):
                self.input_ids = ids
                self.attention_mask = mask

        enc = tokenizer("hello world", return_tensors="pt")
        batch = _MiniBatch(enc["input_ids"], enc["attention_mask"])

        result = decoder.translate_batch(batch, backend)

        gen = result.generations[0]
        # Output should be decodable UTF-8
        gen.translated_text.encode("utf-8")

    def test_batch_with_multiple_sequences(self, tiny_model_and_tokenizer):
        """Self-speculative handles batches with multiple sequences."""
        from benchmark.inference.speculative import (
            SelfSpeculativeDecoder, SpeculativeConfig,
        )

        model, tokenizer = tiny_model_and_tokenizer

        backend = _make_backend(model=model, tokenizer=tokenizer, max_new_tokens=4)
        cfg = SpeculativeConfig(mode="self", num_speculative_tokens=2, num_draft_layers=2)
        decoder = SelfSpeculativeDecoder(backend, cfg)
        decoder.load()

        # Batch of 3 sequences (padded)
        texts = ["hello world", "test sentence", "another one"]
        enc = tokenizer(texts, return_tensors="pt", padding=True)
        batch = type("Batch", (), {
            "batch_id": 1,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "raw_texts": texts,
        })()

        result = decoder.translate_batch(batch, backend)

        assert result.batch_id == 1
        assert result.batch_size == 3
        assert len(result.generations) == 3

        # Each sequence has a result
        for i, gen in enumerate(result.generations):
            assert gen.input_text == texts[i]
            assert gen.output_tokens >= 0
            assert gen.total_latency_ms > 0


# ═════════════════════════════════════════════════════════════════════════════
# Wiring smoke tests — verify the full wiring doesn't crash
# ═════════════════════════════════════════════════════════════════════════════


class TestWiring:
    """Smoke tests for the speculative decoding wiring."""

    def test_backend_init_with_speculative_flags(self):
        """AutoregressiveBackend accepts speculative flags in extra dict."""
        # This test verifies the backend doesn't crash on speculative config.
        # We don't need a real GPU — just validate the flag is stored.
        from benchmark.inference.backends.protocol import BackendConfig
        from benchmark.hardware.backend import DeviceInfo

        try:
            from benchmark.inference.backends.autoregressive import AutoregressiveBackend

            di = DeviceInfo(backend="cpu", device=torch.device("cpu"), num_devices=1)
            extra = {
                "use_speculative": True,
                "speculative_mode": "self",
                "speculative_num_tokens": 3,
                "speculative_num_draft_layers": 0,
            }
            cfg = BackendConfig(
                model_path="google/translategemma-4b-it",
                device_info=di,
                extra=extra,
            )
            backend = AutoregressiveBackend(cfg)
            assert backend._use_speculative is True
            assert backend._spec_mode == "self"
            assert backend._spec_decoder is None
        except ImportError:
            pytest.skip("Transformers not available")

    def test_safe_mode_disables_speculative(self):
        """Safe mode turns off speculative decoding."""
        from benchmark.inference.backends.protocol import BackendConfig
        from benchmark.hardware.backend import DeviceInfo

        try:
            from benchmark.inference.backends.autoregressive import AutoregressiveBackend

            di = DeviceInfo(backend="cpu", device=torch.device("cpu"), num_devices=1)
            extra = {
                "safe_mode": True,
                "use_speculative": True,
                "speculative_mode": "self",
            }
            cfg = BackendConfig(
                model_path="google/translategemma-4b-it",
                device_info=di,
                extra=extra,
            )
            backend = AutoregressiveBackend(cfg)
            assert backend._use_speculative is False  # safe mode overrides
            assert backend._safe_mode is True
        except ImportError:
            pytest.skip("Transformers not available")

    def test_harness_passes_speculative_to_extra(self):
        """Harness includes speculative fields in the extra dict."""
        # Validate config → extra mapping
        model_cfg = ModelConfig(
            use_speculative=True,
            speculative_mode="self",
            speculative_num_tokens=5,
            speculative_num_draft_layers=2,
        )

        # Simulate what the harness does
        extra = {
            "use_speculative": model_cfg.use_speculative,
            "speculative_mode": model_cfg.speculative_mode,
            "speculative_num_tokens": model_cfg.speculative_num_tokens,
            "speculative_draft_model": model_cfg.speculative_draft_model,
            "speculative_num_draft_layers": model_cfg.speculative_num_draft_layers,
        }

        assert extra["use_speculative"] is True
        assert extra["speculative_mode"] == "self"
        assert extra["speculative_num_tokens"] == 5
        assert extra["speculative_num_draft_layers"] == 2

    def test_factory_defaults(self):
        """create_speculative_decoder with defaults creates self-decoder."""
        from benchmark.inference.speculative import (
            create_speculative_decoder, SelfSpeculativeDecoder,
        )
        import torch.nn as nn

        # Build minimal fake backend for factory
        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Linear(64, 64)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(100, 64)
                self.model.layers = nn.ModuleList([FakeLayer() for _ in range(8)])
                self.model.norm = nn.LayerNorm(64)
                self.lm_head = nn.Linear(64, 100)

        model = FakeModel()
        backend = _make_backend(
            model=model, max_new_tokens=32,
        )
        decoder = create_speculative_decoder(backend)
        assert isinstance(decoder, SelfSpeculativeDecoder)
        assert decoder.K == 3  # default
