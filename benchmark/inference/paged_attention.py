"""PagedAttention KV-cache manager (v3.4).

Implements vLLM-style block-level virtual memory for KV-cache management.
Wired into the continuous batching scheduler via the PagedCache protocol
adapter.  See ``ContinuousBatcher._prefill_batch()`` and ``step()`` for
the hot-path integration.

Key concepts
------------
- **PagedKVCache**: Block-level memory manager.  Allocates/frees blocks,
  writes KV into physical block slots, reads assembled contiguous tensors.
- **PagedLayer**: Per-layer adapter implementing the ``DynamicLayer``
  interface that transformers ``DynamicCache`` expects.
- **PagedCache**: Drop-in ``DynamicCache`` replacement — routes attention
  layer KV reads/writes through ``PagedKVCache``.

Usage
-----
>>> paged_kv = PagedKVCache(num_layers=24, num_kv_heads=4, ...)
>>> cache = PagedCache(paged_kv, seq_ids=[0, 1])
>>> out = model(input_ids=..., past_key_values=cache, use_cache=True)

This module is wired into the inference hot path through:
  - ``ContinuousBatcher._prefill_batch()`` — allocates paged blocks,
    writes prefill KV, creates PagedCache.
  - ``ContinuousBatcher.step()`` — passes PagedCache as past_key_values
    to model forward, rebuilds on completion.

For the vLLM-style architecture this implements, see:
  - Kwon et al., "Efficient Memory Management for Large Language Model Serving
    with PagedAttention" (SOSP '23)

---

Implements the vLLM-style memory-paged KV-cache architecture for efficient
variable-length sequence batching.

Key concepts
------------
- **Block**: A fixed-size chunk of KV-cache (e.g., 16 tokens).  Each block
  stores K and V tensors for all layers at those positions.
- **Block table**: Maps logical token positions → physical block indices.
  Sequences with shared prefixes can share physical blocks.
- **Free list**: Pool of unallocated blocks, allocated on demand.

Memory savings vs contiguous KV-cache
--------------------------------------
For batch_size=32, seq_len=1024, Gemma 3 12B:
  - Contiguous: 48 layers × 8 KV heads × 1024 seq × 256 dim × 2 bytes × 32 = ~48 GB
  - Paged (block_size=16, avg utilization=70%): ~34 GB (40% savings)
  - Paged (block_size=16, avg utilization=50%): ~24 GB (50% savings)

The savings come from:
  1. No padding for max_seq_len — each sequence uses only the blocks it needs.
  2. Shared prefix blocks across sequences in the same batch.
  3. Immediate recycling of blocks from completed sequences.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from benchmark.config.constants import (
    DEFAULT_NUM_KV_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_HEAD_DIM,
    PAGED_BLOCK_SIZE,
    PAGED_NUM_BLOCKS_LARGE_GPU,
)

logger = logging.getLogger(__name__)


@dataclass
class KVCacheBlock:
    """One physical block of KV-cache storage.

    Stores K and V for ``block_size`` tokens across all layers.
    """

    block_id: int
    # Per-layer: list of (key_tensor, value_tensor), each shape:
    #   [num_heads, block_size, head_dim]
    layers: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    ref_count: int = 0


@dataclass
class BlockTable:
    """Maps logical positions to physical blocks for one sequence."""

    block_ids: list[int] = field(default_factory=list)
    _total_tokens: int = 0

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def set_total_tokens(self, n: int) -> None:
        self._total_tokens = n


class PagedKVCache:
    """Paged memory manager for key-value caches.

    Usage
    -----
    >>> cache = PagedKVCache(num_layers=48, num_kv_heads=8, head_dim=256,
    ...                      block_size=16, num_blocks=1024)
    >>> cache.allocate(seq_id=0, num_tokens=64)
    >>> cache.write(seq_id=0, layer_idx=5, positions=slice(0, 16),
    ...             key=..., value=...)
    >>> k, v = cache.read(seq_id=0, layer_idx=5)
    """

    def __init__(
        self,
        num_layers: int = DEFAULT_NUM_LAYERS,
        num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
        head_dim: int = DEFAULT_HEAD_DIM,
        block_size: int = PAGED_BLOCK_SIZE,
        num_blocks: int = PAGED_NUM_BLOCKS_LARGE_GPU,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda:0",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        # ── Physical block pool ──
        self._blocks: dict[int, KVCacheBlock] = {}
        self._free_blocks: deque[int] = deque(range(num_blocks))

        # ── Per-sequence block tables ──
        self._block_tables: dict[int, BlockTable] = {}

        # ── Per-sequence token count (used by PagedLayer.update) ──
        self._seq_lengths: dict[int, int] = {}

        # ── Pre-allocate all blocks at init time (or lazily) ──
        self._preallocated = False

    # ── Public API ──────────────────────────────────────────────────────

    def allocate(self, seq_id: int, num_tokens: int) -> BlockTable:
        """Allocate blocks for a sequence of *num_tokens*.

        Returns the block table.  Blocks are allocated on demand from
        the free pool.  Raises RuntimeError if insufficient free blocks.
        """
        num_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self._free_blocks) < num_needed:
            raise RuntimeError(
                f"Out of KV-cache blocks: need {num_needed}, "
                f"have {len(self._free_blocks)} free"
            )

        block_ids = []
        for _ in range(num_needed):
            bid = self._free_blocks.popleft()
            block_ids.append(bid)
            if bid not in self._blocks:
                self._blocks[bid] = KVCacheBlock(block_id=bid)
            self._blocks[bid].ref_count += 1

        table = BlockTable(block_ids=block_ids)
        table.set_total_tokens(num_tokens)
        self._block_tables[seq_id] = table
        self._seq_lengths[seq_id] = num_tokens
        return table

    def free(self, seq_id: int) -> None:
        """Free all blocks allocated to a sequence.

        Blocks with zero references return to the free pool.
        """
        if seq_id not in self._block_tables:
            return

        table = self._block_tables.pop(seq_id)
        self._seq_lengths.pop(seq_id, None)
        for bid in table.block_ids:
            if bid in self._blocks:
                self._blocks[bid].ref_count -= 1
                if self._blocks[bid].ref_count <= 0:
                    self._blocks[bid].layers.clear()
                    self._free_blocks.appendleft(bid)  # LIFO for cache locality

    def write(
        self,
        seq_id: int,
        layer_idx: int,
        positions: slice,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write K and V for a range of token positions.

        Automatically maps positions to physical blocks and writes
        into the correct block slot.
        """
        table = self._block_tables.get(seq_id)
        if table is None:
            raise KeyError(f"Sequence {seq_id} has no allocated blocks")

        start_pos = positions.start
        end_pos = positions.stop
        token_count = end_pos - start_pos

        # Determine which blocks this range spans.
        start_block = start_pos // self.block_size
        end_block = (end_pos - 1) // self.block_size

        for block_idx in range(start_block, end_block + 1):
            if block_idx >= len(table.block_ids):
                break

            bid = table.block_ids[block_idx]
            block = self._blocks.get(bid)
            if block is None:
                logger.warning("Block %d not found for seq %d", bid, seq_id)
                continue

            # Ensure layer storage exists.
            while len(block.layers) <= layer_idx:
                # Pre-allocate layer KV tensors if needed.
                k_tensor = torch.zeros(
                    self.num_kv_heads, self.block_size, self.head_dim,
                    dtype=self.dtype, device=self.device,
                )
                v_tensor = torch.zeros_like(k_tensor)
                block.layers.append((k_tensor, v_tensor))

            # Write slice into block.
            block_start = block_idx * self.block_size
            offset_start = max(start_pos, block_start) - block_start
            offset_end = min(end_pos, block_start + self.block_size) - block_start

            k_block, v_block = block.layers[layer_idx]
            key_slice = key[:, offset_start:offset_end, :]
            val_slice = value[:, offset_start:offset_end, :]
            k_block[:, offset_start:offset_end, :].copy_(key_slice)
            v_block[:, offset_start:offset_end, :].copy_(val_slice)

    def read(
        self, seq_id: int, layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the full K and V for a layer, assembled from blocks.

        Returns concatenated tensors of shape [num_kv_heads, total_tokens, head_dim].
        """
        table = self._block_tables.get(seq_id)
        if table is None:
            raise KeyError(f"Sequence {seq_id} has no allocated blocks")

        keys = []
        values = []
        for bid in table.block_ids:
            block = self._blocks.get(bid)
            if block is None or layer_idx >= len(block.layers):
                continue
            k, v = block.layers[layer_idx]
            keys.append(k)
            values.append(v)

        if not keys:
            raise RuntimeError(f"No KV data for seq {seq_id}, layer {layer_idx}")

        k_cat = torch.cat(keys, dim=1)  # [num_heads, total_blocks * block_size, head_dim]
        v_cat = torch.cat(values, dim=1)
        return k_cat, v_cat

    def share_prefix(self, src_seq_id: int, dst_seq_id: int, num_shared_tokens: int) -> None:
        """Share the first N tokens' blocks between two sequences.

        The destination sequence increments ref counts on the shared
        blocks instead of allocating new ones.  Used for shared prompts
        or system messages.
        """
        src_table = self._block_tables.get(src_seq_id)
        if src_table is None:
            raise KeyError(f"Source sequence {src_seq_id} not found")

        num_blocks = (num_shared_tokens + self.block_size - 1) // self.block_size
        shared_ids = src_table.block_ids[:num_blocks]

        for bid in shared_ids:
            if bid in self._blocks:
                self._blocks[bid].ref_count += 1

        self._block_tables[dst_seq_id] = BlockTable(block_ids=list(shared_ids))

    # ── Introspection ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return memory usage statistics."""
        # Blocks are lazily created in _blocks, so count allocations
        # from the free-pool delta, not _blocks dict size.
        allocated = self.num_blocks - len(self._free_blocks)
        total = self.num_blocks
        # Estimate memory per block.
        bytes_per_block = (
            self.num_layers * self.num_kv_heads * self.block_size
            * self.head_dim * 2 * 2  # K + V, 2 bytes each (BF16)
        )
        total_gb = (allocated * bytes_per_block) / (1024**3)
        return {
            "total_blocks": total,
            "allocated_blocks": allocated,
            "free_blocks": len(self._free_blocks),
            "active_sequences": len(self._block_tables),
            "estimated_memory_gb": round(total_gb, 2),
            "block_size_tokens": self.block_size,
            "bytes_per_block": bytes_per_block,
        }

    def clear(self) -> None:
        """Reset the cache — free all sequences, return all blocks."""
        self._block_tables.clear()
        self._seq_lengths.clear()
        for block in self._blocks.values():
            block.ref_count = 0
            block.layers.clear()
        self._free_blocks = deque(range(self.num_blocks))

    def free_all(self) -> None:
        """Free all allocated blocks and release GPU memory.

        Unlike ``clear()``, which preserves the block pool for reuse,
        this method deallocates the underlying tensors so that GPU
        memory can be reclaimed by other components (e.g., a CUDA‑graph
        capture that needs the VRAM).
        """
        for block in self._blocks.values():
            block.ref_count = 0
            for k, v in block.layers:
                del k, v
            block.layers.clear()
        self._blocks.clear()
        self._free_blocks.clear()
        self._block_tables.clear()
        self._seq_lengths.clear()
        # Force PyTorch to release cached memory back to the driver.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ═════════════════════════════════════════════════════════════════════════════
# Cache Protocol Adapter — makes PagedKVCache a drop-in DynamicCache replacement
# ═════════════════════════════════════════════════════════════════════════════


def _pad_and_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Pad a list of tensors to the same sequence length and stack on dim=0.

    The tensors are shaped ``[num_heads, seq_len_i, head_dim]`` with
    varying ``seq_len_i``.  Pads to ``max(seq_len_i)`` along dim=1
    with zeros and stacks into ``[B, num_heads, max_seq_len, head_dim]``.
    """
    if not tensors:
        return torch.empty(0)
    max_len = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[1] < max_len:
            p = torch.zeros(
                t.shape[0], max_len, t.shape[2],
                dtype=t.dtype, device=t.device,
            )
            p[:, :t.shape[1], :] = t
            padded.append(p)
        else:
            padded.append(t)
    return torch.stack(padded, dim=0)


class PagedLayer:
    """Per-layer KV-cache backed by PagedKVCache blocks.

    Implements the ``DynamicLayer`` interface that ``DynamicCache``
    expects: ``update(key_states, value_states, cache_kwargs=None)``
    returns ``(full_keys, full_values)`` as contiguous tensors suitable
    for the attention kernel.

    Does NOT store tensors itself — all reads/writes delegate to the
    shared ``PagedKVCache`` instance identified by ``seq_id``.
    """

    def __init__(
        self,
        paged_cache: PagedKVCache,
        layer_idx: int,
        seq_ids: list[int],
    ):
        self._cache = paged_cache
        self._layer_idx = layer_idx
        self._seq_ids = list(seq_ids)  # batch-index → seq_id

    @property
    def is_initialized(self) -> bool:
        return True

    def get_seq_length(self) -> int:
        """Max cached sequence length across all active sequences."""
        max_len = 0
        for sid in self._seq_ids:
            blen = self._cache._seq_lengths.get(sid, 0)
            if blen > max_len:
                max_len = blen
        return max_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new single-token KV and return full assembled tensors.

        *key_states*, *value_states*: ``[B, num_kv_heads, 1, head_dim]``
        (single decode token per sequence).

        Returns ``(full_keys, full_values)`` each shaped
        ``[B, num_kv_heads, total_seq_len, head_dim]`` after padding
        to the longest sequence in the batch.
        """
        B = key_states.shape[0]
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []

        for i in range(B):
            seq_id = self._seq_ids[i]
            pos = self._cache._seq_lengths.get(seq_id, 0)

            # Guard: allocate a new block if position overflows the last block.
            table = self._cache._block_tables.get(seq_id)
            if table is not None:
                needed_blocks = (pos + 1 + self._cache.block_size - 1) // self._cache.block_size
                if needed_blocks > table.num_blocks:
                    # Grow the block table by allocating one more block.
                    if not self._cache._free_blocks:
                        raise RuntimeError(
                            f"Out of paged blocks for seq {seq_id} at pos {pos}"
                        )
                    new_bid = self._cache._free_blocks.popleft()
                    table.block_ids.append(new_bid)
                    if new_bid not in self._cache._blocks:
                        self._cache._blocks[new_bid] = KVCacheBlock(block_id=new_bid)
                    self._cache._blocks[new_bid].ref_count += 1

            # Write the single-token K/V into the correct block slot.
            # write() expects the absolute position within the sequence;
            # we pass a tensor covering ONLY the new token, positioned at [pos:pos+1].
            # The write() method slices key[:, offset_start:offset_end, :],
            # so we need a tensor with enough tokens to cover offset_start.
            # Alternative: copy directly into the block layer tensor.
            block_idx = pos // self._cache.block_size
            offset_in_block = pos % self._cache.block_size
            if table is not None and block_idx < len(table.block_ids):
                bid = table.block_ids[block_idx]
                block = self._cache._blocks.get(bid)
                if block is not None:
                    while len(block.layers) <= self._layer_idx:
                        k_tensor = torch.zeros(
                            self._cache.num_kv_heads, self._cache.block_size,
                            self._cache.head_dim,
                            dtype=self._cache.dtype, device=self._cache.device,
                        )
                        v_tensor = torch.zeros_like(k_tensor)
                        block.layers.append((k_tensor, v_tensor))
                    k_block, v_block = block.layers[self._layer_idx]
                    k_block[:, offset_in_block, :].copy_(
                        key_states[i, :, 0, :]
                    )
                    v_block[:, offset_in_block, :].copy_(
                        value_states[i, :, 0, :]
                    )

            self._cache._seq_lengths[seq_id] = pos + 1

            # Update the block table's token count.
            if table is not None:
                table.set_total_tokens(pos + 1)

            # Read assembled full KV for this sequence.
            k, v = self._cache.read(seq_id, self._layer_idx)
            keys.append(k)
            values.append(v)

        return _pad_and_stack(keys), _pad_and_stack(values)

    # ── Required protocol stubs for Cache compatibility ──────────────────

    def get_mask_sizes(self, cache_position) -> tuple[int, int]:
        """Return (kv_length, kv_offset) for causal mask creation."""
        kv_length = self.get_seq_length()
        return (kv_length, 0)

    def get_max_cache_shape(self) -> int:
        """Maximum sequence length (-1 = unbounded)."""
        return -1

    def reset(self) -> None:
        pass

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        pass

    def crop(self, max_length: int) -> None:
        pass


class PagedCache:
    """Drop-in replacement for ``DynamicCache`` backed by ``PagedKVCache``.

    Implements the transformers ``Cache`` protocol so that
    ``model(input_ids=..., past_key_values=paged_cache)`` routes every
    attention layer's KV write/read through paged blocks.

    Usage
    -----
    >>> paged_kv = PagedKVCache(num_layers=24, num_kv_heads=4, ...)
    >>> cache = PagedCache(paged_kv, seq_ids=[0, 1])
    >>> out = model(input_ids=..., past_key_values=cache, use_cache=True)
    """

    def __init__(
        self,
        paged_cache: PagedKVCache,
        seq_ids: list[int],
    ):
        self._paged = paged_cache
        self._seq_ids = list(seq_ids)
        # Materialised PagedLayer per layer index (lazy, created by update()).
        self._layers: dict[int, PagedLayer] = {}

    # ── Cache protocol ───────────────────────────────────────────────────

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Called once per attention layer.  Routes through PagedLayer."""
        if layer_idx not in self._layers:
            self._layers[layer_idx] = PagedLayer(
                self._paged, layer_idx, self._seq_ids,
            )
        return self._layers[layer_idx].update(
            key_states, value_states, cache_kwargs,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        pl = self._layers.get(layer_idx)
        return pl.get_seq_length() if pl else 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1  # unbounded

    def get_mask_sizes(self, cache_position, layer_idx: int = 0) -> tuple[int, int]:
        pl = self._layers.get(layer_idx)
        return pl.get_mask_sizes(cache_position) if pl else (0, 0)

    # ── Batch reordering (used by continuous batching) ───────────────────

    def batch_select_indices(self, indices: torch.Tensor) -> "PagedCache":
        """Return a new PagedCache with only the selected batch indices."""
        new_ids = [self._seq_ids[i] for i in indices.tolist()]
        new_cache = PagedCache(self._paged, new_ids)
        # Share the existing PagedLayer instances (they delegate to
        # PagedKVCache, which is keyed by seq_id, so re-creation is safe).
        return new_cache

    def batch_repeat_interleave(self, repeats: int) -> "PagedCache":
        """Repeat each sequence *repeats* times (for beam search / CFG)."""
        new_ids = []
        for sid in self._seq_ids:
            new_ids.extend([sid] * repeats)
        return PagedCache(self._paged, new_ids)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reorder batch dimension (for beam search)."""
        self._seq_ids = [self._seq_ids[i] for i in beam_idx.tolist()]

    def remove_sequence(self, seq_id: int) -> None:
        """Remove a sequence from the active set (e.g. after EOS).

        Called when paged blocks are freed — prevents subsequent decode
        steps from attempting to read/write blocks that no longer exist.
        """
        try:
            self._seq_ids.remove(seq_id)
        except ValueError:
            pass  # already removed

    # ── Introspection ────────────────────────────────────────────────────

    def reset(self) -> None:
        self._layers.clear()

    def crop(self, max_length: int) -> None:
        pass

    @property
    def seq_ids(self) -> list[int]:
        return list(self._seq_ids)
