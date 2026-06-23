"""Continuous batching scheduler — PagedAttention-powered (v3.5).

Eliminates the idle bubble between static batches by maintaining a
continuously-running pool of sequences at different decode stages.
Powered by PagedAttention block-level KV-cache — completed sequences
release their blocks immediately and waiting sequences claim them.

Architecture
------------
  ┌─────────────────────────────────────────────┐
  │  Waiting queue (pre-tokenised chunks)        │
  │  [chunk_1] [chunk_2] [chunk_3] ...          │
  └──────────────────┬──────────────────────────┘
                     │ schedule()
                     ▼
  ┌─────────────────────────────────────────────┐
  │  Running pool (active sequences)             │
  │  seq_A (step 12)  seq_B (step 5)            │
  │  seq_C (step 3)   seq_D (step 18)           │
  └──────────────────┬──────────────────────────┘
                     │ step() → 1 token each
                     ▼
              ┌──────┴──────┐
              │  EOS / max?  │
              └──────┬──────┘
              Yes     No
              │       │
              ▼       └──▶ back to Running pool
          Output

PagedAttention integration
--------------------------
- **Prefill**: allocates PagedKVCache blocks per new sequence, writes
  prompt KV into blocks, creates PagedCache for model forward.
- **Decode**: PagedCache.update() is called by every attention layer,
  writing new K/V into paged blocks and returning assembled contiguous
  tensors for the attention kernel.
- **Completion**: paged_kv.free(seq_id) returns blocks to pool instantly.

Activation gate
---------------
Only activates when ALL of:
  - Backend is CUDA (not MPS, not CPU).
  - Batch size >= 8 (small batches don't benefit from dynamic scheduling).
  - PagedAttention is enabled (--paged-attention flag).
  - TR_CONTINUOUS_BATCHING_MIN_BATCH env var can lower the threshold.

Key invariants
--------------
- The batch is never empty while there are waiting or running sequences.
- New sequences join at any decode step (prefill then join).
- Completed sequences are immediately replaced.
- Paged blocks are freed as soon as a sequence completes.
- _active_order list tracks batch dimension → seq_id mapping,
  eliminating the dict-iteration-order correctness bug.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from benchmark.config.constants import END_OF_TURN_TOKEN_ID

logger = logging.getLogger(__name__)

# ── Activation thresholds ────────────────────────────────────────────────────
# Continuous batching activates at batch >= 2 so small workloads benefit from
# dynamic scheduling.  Chunked prefill (batched padding) is only used at batch >= 4;
# below that, single-sequence prefill avoids padding waste.
MIN_BATCH_SIZE_FOR_CONTINUOUS = 2  # activate continuous batching at 2
MIN_BATCH_FOR_CHUNKED_PREFILL = 4  # only batch-pad prefills at 4+ sequences


@dataclass
class SequenceState:
    """State for one actively-decoding sequence."""

    seq_id: int
    input_ids: list[int] = field(default_factory=list)        # prompt tokens
    generated_ids: list[int] = field(default_factory=list)     # output tokens so far
    current_position: int = 0
    max_new_tokens: int = 512
    eos_token_id: int = 1
    end_of_turn_token_id: int = END_OF_TURN_TOKEN_ID
    done: bool = False
    raw_text: str = ""

    @property
    def total_tokens(self) -> int:
        return self.current_position + len(self.generated_ids)


class ContinuousBatcher:
    """PagedAttention-powered continuous batching scheduler.

    Keeps the GPU fed by dynamically adding and removing sequences
    from the active batch at every decode step.  Uses ``PagedKVCache``
    for zero-fragmentation KV-cache management.

    CUDA-only — MPS and CPU do not benefit from dynamic scheduling
    because their batch sizes are too small to amortize the overhead.
    """

    def __init__(
        self,
        engine,                     # InferenceEngine
        paged_kv,                   # PagedKVCache
        max_batch_size: int = 64,
        pad_token_id: int = 0,
    ):
        self.engine = engine
        self._paged_kv = paged_kv
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id
        self.device = engine.devices[0]
        self.tokenizer = engine.tokenizer

        # ── Queues ──
        self._waiting: list[SequenceState] = []
        self._running: dict[int, SequenceState] = {}
        self._completed: list[SequenceState] = []

        # ── Batch order: seq IDs in the order they appear in the batch ──
        # This eliminates the dict-iteration-order bug — we always build
        # batch tensors from this list, not from _running.values().
        #
        # Set-based companion for O(1) membership and atomic rollback.
        # The list is the authoritative ordering; the set is a fast
        # membership check used during removal and rollback.
        self._active_order: list[int] = []
        self._active_set: set[int] = set()

        # ── Sequence ID counter ──
        self._next_seq_id: int = 0

        # ── PagedCache for the current decode step ──
        self._paged_cache: Any = None  # PagedCache

        # ── Stats ──
        self.total_sequences_completed: int = 0
        self.total_tokens_generated: int = 0
        self.total_prefill_ms: float = 0.0
        self.total_decode_ms: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────

    def submit(self, input_ids: torch.Tensor, raw_text: str) -> int:
        """Submit a pre-tokenised chunk for translation.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[1, seq_len]`` (single sequence).
        raw_text : str
            Original text for logging/reporting.

        Returns
        -------
        int
            Assigned sequence ID.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        state = SequenceState(
            seq_id=seq_id,
            input_ids=input_ids.squeeze(0).tolist(),
            max_new_tokens=self.engine.decoding_params.max_new_tokens
            if hasattr(self.engine, 'decoding_params') else 512,
            eos_token_id=self.tokenizer.eos_token_id,
            end_of_turn_token_id=END_OF_TURN_TOKEN_ID,
            raw_text=raw_text,
        )
        self._waiting.append(state)
        return seq_id

    def step(self) -> list[SequenceState]:
        """Execute one decode iteration.

        1. Prefill waiting sequences (allocate paged blocks, write prompt KV).
        2. Add prefilled sequences to the running pool.
        3. Run one decode step for all running sequences through PagedCache.
        4. Remove completed sequences, free their paged blocks.

        Returns
        -------
        list[SequenceState]
            Sequences that completed this step.
        """
        newly_completed: list[SequenceState] = []
        new_seqs: list[SequenceState] = []

        # ── 1. Fill batch with waiting sequences (paged prefill) ──
        capacity = self.max_batch_size - len(self._running)
        if capacity > 0 and self._waiting:
            while len(new_seqs) < capacity and self._waiting:
                seq = self._waiting.pop(0)
                new_seqs.append(seq)

            t0 = time.monotonic()
            try:
                self._prefill_batch(new_seqs)
                self.total_prefill_ms += (time.monotonic() - t0) * 1000.0
            except Exception:
                # ── Atomic rollback on prefill failure ──
                # Free any KV blocks that were allocated for sequences
                # that succeeded before the failure.  Re-enqueue all
                # new_seqs to _waiting so they can be retried later.
                for seq in new_seqs:
                    try:
                        self._paged_kv.free(seq.seq_id)
                    except KeyError:
                        pass  # block was never allocated — fine
                    seq.current_position = 0
                    self._waiting.insert(0, seq)
                raise

            for seq in new_seqs:
                self._running[seq.seq_id] = seq
                self._active_order.append(seq.seq_id)
                self._active_set.add(seq.seq_id)

        if not self._running:
            return []

        # ── 2. Build batch from active sequences ──
        batch_input_ids, batch_positions = self._assemble_batch()

        # ── 3. Decode one token through PagedCache ──
        t0 = time.monotonic()
        with torch.no_grad():
            outputs = self.engine.model(
                input_ids=batch_input_ids,
                past_key_values=self._paged_cache,
                use_cache=True,
            )

        # PagedCache was updated in-place by attention layers.
        # outputs.past_key_values is the same PagedCache object.
        logits = outputs.logits  # [batch, 1, vocab_size]
        next_tokens = logits[:, -1, :].argmax(dim=-1)  # [batch]

        decode_ms = (time.monotonic() - t0) * 1000.0
        self.total_decode_ms += decode_ms

        # ── 4. Update sequences, evict completed ones ──
        completed_ids: list[int] = []
        for batch_idx, seq_id in enumerate(self._active_order):
            seq = self._running.get(seq_id)
            if seq is None or seq.done:
                continue

            tok = next_tokens[batch_idx].item()
            seq.generated_ids.append(tok)
            seq.current_position += 1
            self.total_tokens_generated += 1

            if tok in (seq.eos_token_id, seq.end_of_turn_token_id) or \
               len(seq.generated_ids) >= seq.max_new_tokens:
                seq.done = True
                completed_ids.append(seq_id)
                newly_completed.append(seq)
                self.total_sequences_completed += 1

        # ── 4a. Remove completed sequences ──
        for seq_id in completed_ids:
            self._running.pop(seq_id, None)
            self._active_set.discard(seq_id)
            # Free paged blocks immediately.
            self._paged_kv.free(seq_id)
            # Incrementally drop from PagedCache (avoids full rebuild).
            self._paged_cache.remove_sequence(seq_id)
        if completed_ids:
            # Rebuild _active_order without the completed seq_ids (list
            # comprehension preserves order of survivors for determinism).
            completed_set = set(completed_ids)
            self._active_order = [sid for sid in self._active_order
                                  if sid not in completed_set]

        return newly_completed

    def run_to_completion(self, input_batches: list) -> list[SequenceState]:
        """Process all input batches to completion using continuous batching.

        This is the main entry point — replaces the static translate_batch
        loop with a dynamic decode scheduler.

        Parameters
        ----------
        input_batches : list
            Pre-tokenised PipelineBatch objects from the data pipeline.

        Returns
        -------
        list[SequenceState]
            All completed sequences with their generated tokens.
        """
        # Submit all input chunks.
        for batch in input_batches:
            if not hasattr(batch, 'input_ids'):
                continue
            for i in range(batch.input_ids.shape[0]):
                raw = (
                    batch.raw_texts[i]
                    if hasattr(batch, 'raw_texts') and i < len(batch.raw_texts)
                    else ""
                )
                self.submit(batch.input_ids[i:i + 1], raw)

        # Run decode loop until all sequences complete.
        max_steps = 0
        while self._waiting or self._running:
            completed = self.step()
            max_steps += 1

            if completed:
                self._completed.extend(completed)
                logger.debug(
                    "Completed %d sequences (running=%d, waiting=%d, step=%d)",
                    len(completed), len(self._running),
                    len(self._waiting), max_steps,
                )

            # Safety: prevent infinite loop if a sequence never finishes.
            if max_steps > 100_000:
                logger.error(
                    "ContinuousBatcher hit max_steps=100k — aborting. "
                    "running=%d waiting=%d",
                    len(self._running), len(self._waiting),
                )
                # Force-complete remaining sequences.
                for seq in list(self._running.values()):
                    if not seq.done:
                        seq.done = True
                        self._completed.append(seq)
                        self._paged_kv.free(seq.seq_id)
                break

        return self._completed

    def is_idle(self) -> bool:
        """True when no sequences are waiting or running."""
        return not self._waiting and not self._running

    def running_count(self) -> int:
        return len(self._running)

    def waiting_count(self) -> int:
        return len(self._waiting)

    def active_batch_size(self) -> int:
        return len(self._active_order)

    # ── Internal ──────────────────────────────────────────────────────────

    def _prefill_batch(self, sequences: list[SequenceState]) -> None:
        """Paged prefill: allocate blocks, run model forward, write KV into blocks.

        With PagedKVCache, each sequence gets its own blocks — no padding
        to a shared max length, no KV concatenation.  The PagedCache handles
        variable-length sequences natively.

        Raises on any failure — the caller (``step()``) is responsible for
        atomic rollout of ``_running`` / ``_active_order`` / ``_active_set``.
        """
        from benchmark.inference.paged_attention import PagedCache

        if not sequences:
            return

        n_new = len(sequences)

        # ── Prefill each sequence individually ──
        # Batching prefills would require padding prompts to the same length,
        # which defeats the purpose of paged memory.  Individual prefill is
        # slightly slower for latency but saves memory and simplifies block
        # management.  Batched padding only above MIN_BATCH_FOR_CHUNKED_PREFILL.
        if n_new >= MIN_BATCH_FOR_CHUNKED_PREFILL:
            self._prefill_batch_padded(sequences)
        else:
            for seq in sequences:
                self._prefill_sequence(seq)

        # ── Build PagedCache for the full running pool ──
        if self._active_order:
            all_seq_ids = list(self._active_order)
            for seq in sequences:
                all_seq_ids.append(seq.seq_id)
        else:
            all_seq_ids = [s.seq_id for s in sequences]

        self._paged_cache = PagedCache(self._paged_kv, seq_ids=all_seq_ids)

    def _prefill_batch_padded(self, sequences: list[SequenceState]) -> None:
        """Batched prefill for 4+ sequences — pads to max prompt length."""
        device = self.device

        # Pad to max length in the group.
        max_len = max(len(s.input_ids) for s in sequences)
        padded = []
        attn_mask = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            prompt_len = len(seq.input_ids)
            row = seq.input_ids + [self.pad_token_id] * (max_len - prompt_len)
            padded.append(row)
            attn_mask[i, :prompt_len] = 1

        prompt_batch = torch.tensor(padded, dtype=torch.long, device=device)

        with torch.no_grad():
            prefill_out = self.engine.model(
                input_ids=prompt_batch,
                attention_mask=attn_mask,
                use_cache=True,
            )
            prefill_kv = prefill_out.past_key_values

        # Allocate paged blocks and write prefill KV for each sequence.
        seq_lengths = attn_mask.sum(dim=-1).int().tolist()
        # Track allocated blocks for rollback on failure.
        allocated_ids: list[int] = []
        try:
            for i, seq in enumerate(sequences):
                # Guard: skip if already allocated (prevents double-prefill overwrite).
                try:
                    self._paged_kv._block_tables[seq.seq_id]
                    continue  # already allocated — skip
                except KeyError:
                    pass

                slen = seq_lengths[i]
                self._paged_kv.allocate(seq.seq_id, slen)
                allocated_ids.append(seq.seq_id)
                seq.current_position = slen

                # Access KV tensors via len() + __getitem__ — the only
                # version-safe pattern across all transformers 4.x and 5.x.
                # DynamicCache.__getitem__(idx) → (k, v) tuple is stable.
                # DynamicCache.key_cache requires transformers >= 4.58.0.
                # enumerate(DynamicCache) yields version-dependent values.
                for layer_idx in range(len(prefill_kv)):
                    k, v = prefill_kv[layer_idx]
                    self._paged_kv.write(
                        seq.seq_id, layer_idx,
                        slice(0, slen),
                        k[i, :, :slen, :],
                        v[i, :, :slen, :],
                    )
        except Exception:
            # Prefill failure — free blocks allocated so far in this batch
            # to prevent paged-block leak, then re-raise so the caller
            # (step()) can atomically roll back _running / _active_order.
            for sid in allocated_ids:
                self._paged_kv.free(sid)
            raise

    def _prefill_sequence(self, seq: SequenceState) -> None:
        """Single-sequence prefill — no padding, direct block allocation."""
        device = self.device
        prompt = torch.tensor([seq.input_ids], dtype=torch.long, device=device)
        prompt_len = len(seq.input_ids)

        with torch.no_grad():
            prefill_out = self.engine.model(
                input_ids=prompt,
                use_cache=True,
            )
            prefill_kv = prefill_out.past_key_values

        # Allocate paged blocks for this sequence (with rollback on failure).
        allocated = False
        try:
            self._paged_kv.allocate(seq.seq_id, prompt_len)
            allocated = True
            seq.current_position = prompt_len

            # Write prefill KV into paged blocks.
            # len(prefill_kv) + prefill_kv[idx] is the only version-safe
            # access pattern across all transformers 4.x and 5.x.
            # DynamicCache.__getitem__(idx) → (k, v) tuple is stable.
            for layer_idx in range(len(prefill_kv)):
                k, v = prefill_kv[layer_idx]
                self._paged_kv.write(
                    seq.seq_id, layer_idx,
                    slice(0, prompt_len),
                    k[0, :, :prompt_len, :],
                    v[0, :, :prompt_len, :],
                )
        except Exception:
            if allocated:
                self._paged_kv.free(seq.seq_id)
            raise

    def _assemble_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the decode-step input tensors from active sequences.

        Returns (input_ids, position_ids) for the model forward.
        The attention mask is NOT needed — PagedCache returns assembled
        full-length KV tensors that already encode the causal mask.

        Uses ``_active_order`` (NOT ``_running.values()``) for deterministic
        batch-index → seq_id mapping.

        Raises
        ------
        RuntimeError
            If ``_active_order`` and ``_running`` are inconsistent — a seq_id
            in one but not the other indicates a scheduler bug that would
            produce silent data corruption.
        """
        device = self.device

        # ── Validate consistency between _active_order and _running ──
        # Use the pre-maintained _active_set instead of building on every call.
        active_set = self._active_set
        running_set = set(self._running.keys())
        only_in_active = active_set - running_set
        only_in_running = running_set - active_set
        # Defensive: _active_set and _active_order must agree on membership.
        assert len(active_set) == len(self._active_order), (
            f"_active_set ({len(active_set)}) and _active_order "
            f"({len(self._active_order)}) diverged — scheduler bug"
        )
        if only_in_active:
            raise RuntimeError(
                f"ContinuousBatcher state corruption: seq_ids {only_in_active} "
                f"are in _active_order but not in _running.  _assemble_batch "
                f"cannot safely construct the batch.  This is a scheduler bug."
            )
        if only_in_running:
            raise RuntimeError(
                f"ContinuousBatcher state corruption: seq_ids {only_in_running} "
                f"are in _running but not in _active_order.  These sequences "
                f"will be silently dropped from the decode batch.  This is a "
                f"scheduler bug."
            )

        # Build input IDs and position IDs in a SINGLE pass to prevent
        # desync between batch_input_ids and positions.
        next_tokens_list = []
        position_list = []
        for seq_id in self._active_order:
            seq = self._running[seq_id]  # guaranteed present by validation above
            if seq.generated_ids:
                next_tokens_list.append(seq.generated_ids[-1])
            else:
                next_tokens_list.append(seq.input_ids[-1])
            position_list.append(seq.current_position)

        batch_input_ids = torch.tensor(
            next_tokens_list, dtype=torch.long, device=device,
        ).unsqueeze(-1)  # [batch, 1]

        # Position IDs track cumulative token count per sequence.
        positions = torch.tensor(
            position_list,
            dtype=torch.long, device=device,
        ).unsqueeze(-1)  # [batch, 1]

        # ── Assertion: input IDs and position IDs must agree on batch size ──
        # A length mismatch means position_list or next_tokens_list desynced
        # from _active_order, which produces silent data corruption.
        assert batch_input_ids.shape[0] == positions.shape[0] == len(self._active_order), (
            f"Batch assembly size mismatch: batch_input_ids={batch_input_ids.shape[0]}, "
            f"positions={positions.shape[0]}, _active_order={len(self._active_order)}"
        )

        return batch_input_ids, positions

    def drain_running(self) -> list[SequenceState]:
        """Drain all remaining running sequences — free paged blocks, return states.

        Use this to recover from a stuck batch or to gracefully shut down
        without leaking paged KV-cache blocks.  All drained sequences are
        marked as ``done`` and their blocks are returned to the free pool.

        Returns
        -------
        list[SequenceState]
            All sequences that were in the running pool.
        """
        drained = []
        for seq_id in list(self._active_order):
            seq = self._running.pop(seq_id, None)
            if seq is not None:
                seq.done = True
                drained.append(seq)
                self._paged_kv.free(seq_id)
        self._active_order.clear()
        self._active_set.clear()
        self._paged_cache = None
        return drained

    def flush_completed(self) -> list[SequenceState]:
        """Drain all remaining running sequences (called at shutdown).

        Deprecated: use ``drain_running()`` instead.  Kept for backward
        compatibility.
        """
        return self.drain_running()


# ── Integration helper ───────────────────────────────────────────────────────


def should_use_continuous_batching(
    backend_name: str,
    batch_size: int,
    use_paged_attention: bool = False,
) -> bool:
    """Return True if continuous batching should be activated.

    Conditions:
      - CUDA backend (the only backend where batch sizes are large enough).
      - Batch size >= MIN_BATCH_SIZE_FOR_CONTINUOUS (default 8).
      - PagedAttention is enabled (required for zero-fragmentation KV cache).
      - TR_CONTINUOUS_BATCHING_MIN_BATCH env var can lower the threshold.
    """
    import os
    min_bs = int(os.environ.get(
        "TR_CONTINUOUS_BATCHING_MIN_BATCH",
        str(MIN_BATCH_SIZE_FOR_CONTINUOUS),
    ))

    if backend_name != "cuda":
        return False
    if batch_size < min_bs:
        logger.info(
            "Continuous batching skipped: batch_size=%d < min=%d",
            batch_size, min_bs,
        )
        return False
    if not use_paged_attention:
        logger.info(
            "Continuous batching requires --paged-attention. "
            "Skipping."
        )
        return False

    return True
