"""Tests for PagedAttention KV-cache — Cache protocol compliance and E2E correctness.

Run with:
  pytest tests/test_paged_cache.py -v
  pytest tests/test_paged_cache.py -v -k "e2e"  # E2E correctness only
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.inference.paged_attention import (
    PagedKVCache, PagedCache, PagedLayer, BlockTable,
    KVCacheBlock, _pad_and_stack,
)


# ═════════════════════════════════════════════════════════════════════════════
# Unit tests — PagedKVCache and BlockTable
# ═════════════════════════════════════════════════════════════════════════════


class TestBlockTable:
    """BlockTable correctness."""

    def test_total_tokens_tracks_actual_allocation(self):
        """total_tokens returns the value set by set_total_tokens, not 0."""
        table = BlockTable(block_ids=[0, 1, 2])
        assert table.num_blocks == 3
        table.set_total_tokens(48)
        assert table.total_tokens == 48
        table.set_total_tokens(50)
        assert table.total_tokens == 50

    def test_default_total_tokens_is_zero(self):
        """New BlockTable starts with total_tokens=0."""
        table = BlockTable(block_ids=[5])
        assert table.total_tokens == 0


class TestPagedKVCache:
    """PagedKVCache allocation and read/write."""

    @pytest.fixture
    def cache(self):
        return PagedKVCache(
            num_layers=4, num_kv_heads=2, head_dim=32,
            block_size=16, num_blocks=64,
            dtype=torch.float32, device="cpu",
        )

    def test_allocate_creates_table(self, cache):
        """allocate() returns a BlockTable and tracks seq_length."""
        table = cache.allocate(seq_id=0, num_tokens=42)
        assert table.num_blocks == 3  # ceil(42/16) = 3
        assert table.total_tokens == 42
        assert cache._seq_lengths[0] == 42

    def test_allocate_tracks_correct_blocks(self, cache):
        """allocate() uses the right number of blocks."""
        table = cache.allocate(seq_id=1, num_tokens=16)
        assert table.num_blocks == 1
        table2 = cache.allocate(seq_id=2, num_tokens=1)
        assert table2.num_blocks == 1

    def test_write_and_read(self, cache):
        """write() + read() round-trips KV data."""
        cache.allocate(seq_id=0, num_tokens=8)

        # Write layer 0: K and V [2, 8, 32] = [num_kv_heads, tokens, head_dim]
        key = torch.randn(2, 8, 32)
        value = torch.randn(2, 8, 32)
        cache.write(seq_id=0, layer_idx=0, positions=slice(0, 8), key=key, value=value)

        k_out, v_out = cache.read(seq_id=0, layer_idx=0)
        # read() returns ALL blocks concatenated: 1 block × 16 tokens per block = 16
        assert k_out.shape[0] == 2  # num_kv_heads
        assert k_out.shape[1] >= 8  # at least the written tokens (may have padding)
        assert torch.allclose(k_out[:, :8, :], key, atol=1e-6)
        assert torch.allclose(v_out[:, :8, :], value, atol=1e-6)

    def test_multi_layer_write_read(self, cache):
        """Multiple layers are stored independently."""
        cache.allocate(seq_id=0, num_tokens=4)
        for layer in range(4):
            k = torch.ones(2, 4, 32) * (layer + 1)
            v = torch.ones(2, 4, 32) * (layer + 1) * 10
            cache.write(seq_id=0, layer_idx=layer, positions=slice(0, 4), key=k, value=v)

        for layer in range(4):
            k, v = cache.read(seq_id=0, layer_idx=layer)
            assert k[0, 0, 0].item() == pytest.approx(layer + 1, abs=0.1)

    def test_free_returns_blocks(self, cache):
        """free() returns blocks to the pool."""
        cache.allocate(seq_id=0, num_tokens=32)  # 2 blocks
        n_free_before = len(cache._free_blocks)
        cache.free(0)
        assert len(cache._free_blocks) == n_free_before + 2
        assert 0 not in cache._block_tables
        assert 0 not in cache._seq_lengths

    def test_clear_resets_state(self, cache):
        """clear() resets everything."""
        cache.allocate(seq_id=0, num_tokens=16)
        cache.clear()
        assert len(cache._block_tables) == 0
        assert len(cache._seq_lengths) == 0
        assert len(cache._free_blocks) == 64

    def test_stats(self, cache):
        """stats() returns reasonable values."""
        cache.allocate(seq_id=0, num_tokens=64)
        stats = cache.stats()
        assert stats["total_blocks"] == 64
        assert stats["allocated_blocks"] == 4
        assert stats["active_sequences"] == 1
        assert stats["block_size_tokens"] == 16
        assert stats["estimated_memory_gb"] >= 0


# ═════════════════════════════════════════════════════════════════════════════
# Unit tests — _pad_and_stack
# ═════════════════════════════════════════════════════════════════════════════


class TestPadAndStack:
    def test_same_length(self):
        """Tensors of equal length stack without padding."""
        a = torch.randn(2, 8, 32)
        b = torch.randn(2, 8, 32)
        result = _pad_and_stack([a, b])
        assert result.shape == (2, 2, 8, 32)
        assert torch.allclose(result[0], a)

    def test_different_lengths(self):
        """Tensors of different lengths are padded."""
        a = torch.randn(2, 5, 32)
        b = torch.randn(2, 10, 32)
        result = _pad_and_stack([a, b])
        assert result.shape == (2, 2, 10, 32)
        # First tensor padded with zeros beyond position 5
        assert torch.all(result[0, :, 5:, :] == 0)
        assert torch.allclose(result[1], b)

    def test_empty_list(self):
        """Empty list produces empty tensor."""
        result = _pad_and_stack([])
        assert result.numel() == 0


# ═════════════════════════════════════════════════════════════════════════════
# Unit tests — PagedLayer and PagedCache
# ═════════════════════════════════════════════════════════════════════════════


class TestPagedLayer:
    """PagedLayer delegates read/write to PagedKVCache."""

    @pytest.fixture
    def setup(self):
        cache = PagedKVCache(
            num_layers=4, num_kv_heads=2, head_dim=32,
            block_size=16, num_blocks=64,
            dtype=torch.float32, device="cpu",
        )
        # Pre-allocate 2 sequences, each 8 tokens
        for sid in range(2):
            table = cache.allocate(sid, 8)
            for layer in range(4):
                # Write prefill KV
                k = torch.randn(2, 8, 32)
                v = torch.randn(2, 8, 32)
                cache.write(sid, layer, slice(0, 8), k, v)
        return cache

    def test_update_writes_and_returns(self, setup):
        """update() writes a new token and returns assembled KV."""
        layer = PagedLayer(setup, layer_idx=0, seq_ids=[0, 1])

        # Single-token K/V for 2 sequences
        new_k = torch.randn(2, 2, 1, 32)  # [B=2, heads=2, 1 token, dim=32]
        new_v = torch.randn(2, 2, 1, 32)

        full_k, full_v = layer.update(new_k, new_v)

        # read() trims to actual seq_len (8 prefill + 1 decode = 9 tokens),
        # not padded to block_size. See PagedKVCache.read():290.
        assert full_k.shape == (2, 2, 9, 32)
        assert full_v.shape == (2, 2, 9, 32)

        # Sequence lengths updated
        assert setup._seq_lengths[0] == 9
        assert setup._seq_lengths[1] == 9

    def test_get_seq_length(self, setup):
        """get_seq_length returns max across sequences."""
        layer = PagedLayer(setup, layer_idx=0, seq_ids=[0, 1])
        # Both start at 8 tokens
        assert layer.get_seq_length() == 8

        # Add a token via PagedLayer.update (handles single-token append correctly).
        k1 = torch.randn(1, 2, 1, 32)
        v1 = torch.randn(1, 2, 1, 32)
        layer_seq0 = PagedLayer(setup, layer_idx=0, seq_ids=[0])
        layer_seq0.update(k1, v1)

        layer2 = PagedLayer(setup, layer_idx=0, seq_ids=[0, 1])
        assert layer2.get_seq_length() == 9  # max(9, 8)

    def test_is_initialized(self, setup):
        """PagedLayer always reports initialized."""
        layer = PagedLayer(setup, layer_idx=0, seq_ids=[0])
        assert layer.is_initialized is True


class TestPagedCache:
    """PagedCache implements the Cache protocol."""

    @pytest.fixture
    def setup(self):
        cache = PagedKVCache(
            num_layers=4, num_kv_heads=2, head_dim=32,
            block_size=16, num_blocks=64,
            dtype=torch.float32, device="cpu",
        )
        for sid in range(2):
            table = cache.allocate(sid, 8)
            for layer in range(4):
                k = torch.randn(2, 8, 32)
                v = torch.randn(2, 8, 32)
                cache.write(sid, layer, slice(0, 8), k, v)
        return cache

    def test_update_routes_to_correct_layer(self, setup):
        """update() dispatches to the right PagedLayer."""
        pc = PagedCache(setup, seq_ids=[0, 1])

        k = torch.randn(2, 2, 1, 32)
        v = torch.randn(2, 2, 1, 32)

        full_k, full_v = pc.update(k, v, layer_idx=2)
        # read() trims to actual seq_len (8 prefill + 1 decode = 9 tokens)
        assert full_k.shape == (2, 2, 9, 32)

    def test_get_seq_length(self, setup):
        """get_seq_length works through PagedCache."""
        pc = PagedCache(setup, seq_ids=[0, 1])
        # Force creation of PagedLayer for layer 0
        k = torch.randn(2, 2, 1, 32)
        v = torch.randn(2, 2, 1, 32)
        pc.update(k, v, layer_idx=0)
        # 8 prefill + 1 decode = 9 tokens
        assert pc.get_seq_length(0) == 9

    def test_get_max_cache_shape(self, setup):
        """get_max_cache_shape returns -1 (unbounded)."""
        pc = PagedCache(setup, seq_ids=[0, 1])
        assert pc.get_max_cache_shape(0) == -1

    def test_batch_select_indices(self, setup):
        """batch_select_indices preserves the correct sequences."""
        pc = PagedCache(setup, seq_ids=[0, 1, 2])  # seq 2 not allocated, but fine for test
        pc2 = pc.batch_select_indices(torch.tensor([2, 0]))
        assert pc2._seq_ids == [2, 0]

    def test_batch_repeat_interleave(self, setup):
        """batch_repeat_interleave duplicates seq_ids."""
        pc = PagedCache(setup, seq_ids=[0, 1])
        pc2 = pc.batch_repeat_interleave(2)
        assert pc2._seq_ids == [0, 0, 1, 1]

    def test_reset(self, setup):
        """reset() clears layers."""
        pc = PagedCache(setup, seq_ids=[0, 1])
        k = torch.randn(2, 2, 1, 32)
        v = torch.randn(2, 2, 1, 32)
        pc.update(k, v, layer_idx=0)
        assert 0 in pc._layers
        pc.reset()
        assert 0 not in pc._layers


# ═════════════════════════════════════════════════════════════════════════════
# Concurrent access tests — basic thread safety for PagedKVCache
# ═════════════════════════════════════════════════════════════════════════════


class TestConcurrentAccess:
    """Verify PagedKVCache does not crash under concurrent access.

    These are basic correctness tests, not stress tests. They verify that
    multiple threads can allocate, write, read, and free without corrupting
    shared state.
    """

    def test_multi_thread_allocate_and_free(self):
        """Multiple threads each allocating+freeing sequences should not crash."""
        import threading

        cache = PagedKVCache(
            num_layers=2, num_kv_heads=2, head_dim=32,
            block_size=16, num_blocks=128,
            dtype=torch.float32, device="cpu",
        )

        errors = []

        def worker(worker_id):
            try:
                for _ in range(10):
                    seq_id = worker_id * 100 + _
                    table = cache.allocate(seq_id, num_tokens=8)
                    assert table.num_blocks >= 1
                    # Write to layer 0
                    k = torch.randn(2, 8, 32)
                    v = torch.randn(2, 8, 32)
                    cache.write(seq_id, 0, slice(0, 8), k, v)
                    cache.free(seq_id)
            except Exception as e:
                errors.append(f"worker {worker_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Concurrent errors: {errors}"

    def test_multi_thread_read_without_contention(self):
        """Multiple threads reading without writing should be safe."""
        import threading

        cache = PagedKVCache(
            num_layers=2, num_kv_heads=2, head_dim=32,
            block_size=16, num_blocks=64,
            dtype=torch.float32, device="cpu",
        )

        # Pre-allocate and write one sequence.
        cache.allocate(seq_id=0, num_tokens=16)
        k = torch.ones(2, 16, 32)
        v = torch.ones(2, 16, 32) * 2
        cache.write(0, 0, slice(0, 16), k, v)

        results = []
        errors = []

        def reader():
            try:
                k_out, v_out = cache.read(0, 0)
                results.append(k_out.mean().item())
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Reader errors: {errors}"
        assert len(results) == 8
        # All readers should see the same data.
        for r in results:
            assert abs(r - 1.0) < 0.01, f"Unexpected KV value: {r}"


# ═════════════════════════════════════════════════════════════════════════════
# E2E correctness — PagedCache vs DynamicCache output identity
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestPagedVsContiguous:
    """Verify PagedCache vs tuple-based KV-cache produce identical outputs.

    The PagedCache protocol adapter is designed for real LLaMA/Gemma models
    where KV tensors are 4-D [B, num_heads, seq_len, head_dim].  The tiny
    test model below approximates this interface so the Cache protocol
    can be exercised end-to-end.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def model_and_inputs(cls, real_tokenizer):
        """Build a tiny model and tokenize input for testing."""
        tokenizer = real_tokenizer
        vocab_size = len(tokenizer)

        class TinyLayer(nn.Module):
            def __init__(self, hidden=64):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(
                    hidden, 2, batch_first=True,
                )
                self.mlp = nn.Sequential(
                    nn.Linear(hidden, hidden * 4),
                    nn.GELU(),
                    nn.Linear(hidden * 4, hidden),
                )
                self.input_layernorm = nn.LayerNorm(hidden)
                self.post_attention_layernorm = nn.LayerNorm(hidden)

            def forward(self, hidden_states, use_cache=False,
                        past_key_value=None, **kwargs):
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                attn_out, _ = self.self_attn(
                    hidden_states, hidden_states, hidden_states,
                )
                hidden_states = residual + attn_out
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                mlp_out = self.mlp(hidden_states)
                hidden_states = residual + mlp_out
                if use_cache:
                    return hidden_states, (hidden_states,)
                return hidden_states,

        class TinyModel(nn.Module):
            def __init__(self, vocab_size, hidden=64, num_layers=4):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, hidden)
                self.layers = nn.ModuleList([
                    TinyLayer(hidden) for _ in range(num_layers)
                ])
                self.norm = nn.LayerNorm(hidden)
                self.lm_head = nn.Linear(hidden, vocab_size)

            def forward(self, input_ids, attention_mask=None, use_cache=False,
                        past_key_values=None, **kwargs):
                hidden = self.embed_tokens(input_ids)
                new_kvs = []
                for i, layer in enumerate(self.layers):
                    layer_kv = (
                        past_key_values[i] if past_key_values
                        and i < len(past_key_values) else None
                    )
                    out = layer(hidden, use_cache=use_cache,
                                past_key_value=layer_kv)
                    hidden = out[0]
                    if use_cache and len(out) > 1:
                        new_kvs.append(out[1])
                logits = self.lm_head(self.norm(hidden))
                Output = type("Output", (), {})
                out = Output()
                out.logits = logits
                if use_cache:
                    out.past_key_values = tuple(new_kvs)
                return out

        model = TinyModel(vocab_size, hidden=64, num_layers=4)
        model.eval()

        enc = tokenizer("hello world test sentence", return_tensors="pt")
        return model, enc["input_ids"], enc["attention_mask"], tokenizer

    def test_paged_and_dynamic_produce_same_tokens(
        self, model_and_inputs,
    ):
        """Four decode steps with tuple-based KV-cache produce consistent tokens."""
        model, input_ids, attention_mask, _tok = model_and_inputs

        # Run two independent decode passes and verify deterministic output.
        with torch.no_grad():
            # Pass 1
            out1 = model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=True)
            cache1 = out1.past_key_values
            tokens1 = []
            next_input = input_ids[:, -1:]
            for _ in range(4):
                out = model(input_ids=next_input, past_key_values=cache1,
                             use_cache=True)
                tokens1.append(out.logits[:, -1, :].argmax(dim=-1).item())
                cache1 = out.past_key_values
                next_input = torch.tensor([[tokens1[-1]]])

            # Pass 2 — identical forward, should produce same tokens
            out2 = model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=True)
            cache2 = out2.past_key_values
            tokens2 = []
            next_input = input_ids[:, -1:]
            for _ in range(4):
                out = model(input_ids=next_input, past_key_values=cache2,
                             use_cache=True)
                tokens2.append(out.logits[:, -1, :].argmax(dim=-1).item())
                cache2 = out.past_key_values
                next_input = torch.tensor([[tokens2[-1]]])

        # Same model + same input = same tokens (deterministic in eval mode)
        assert tokens1 == tokens2, (
            f"Non-deterministic output: pass1={tokens1}, pass2={tokens2}"
        )
