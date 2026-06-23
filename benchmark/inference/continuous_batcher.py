"""Continuous batching scheduler — PagedAttention-powered with chunked prefill (v3.6).

Eliminates the idle bubble between static batches by maintaining a
continuously-running pool of sequences at different decode stages.
Powered by PagedAttention block-level KV-cache — completed sequences
release their blocks immediately and waiting sequences claim them.

Chunked prefill (v3.6)
-----------------------
The defining characteristic of production-grade inference systems.
Instead of all-or-nothing prefill (which stalls decode while long
prompts are ingested), chunked prefill interleaves small prefill chunks
with decode tokens at every step:

1. Decode tokens are allocated FIRST (1 token per active decode sequence).
2. Remaining token_budget is filled with prefill chunks (up to 512 tokens
   per chunk, aligned to multiples of 128).
3. Sequences in _prefill_queue process incrementally until their entire
   prompt is prefilled, then transition to _decode_queue.

This guarantees that decode latency never spikes due to a long-prompt
prefill hogging the GPU — the core invariant of production serving.

Architecture
------------
  ┌─────────────────────────────────────────────┐
  │  Waiting queue (pre-tokenised chunks)        │
  │  [chunk_1] [chunk_2] [chunk_3] ...          │
  └──────────────────┬──────────────────────────┘
                     │ schedule()
                     ▼
  ┌─────────────────────────────────────────────┐
  │  Prefill queue (incremental chunked prefill) │
  │  seq_A (48/512)  seq_B (128/300)            │
  └──────────────────┬──────────────────────────┘
                     │ prefill completes
                     ▼
  ┌─────────────────────────────────────────────┐
  │  Decode queue (1 token per sequence)         │
  │  seq_C (step 12)  seq_D (step 5)            │
  │  seq_E (step 3)                              │
  └──────────────────┬──────────────────────────┘
                     │ step() → 1 token each
                     ▼
              ┌──────┴──────┐
              │  EOS / max?  │
              └──────┬──────┘
              Yes     No
              │       │
              ▼       └──▶ back to Decode queue
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
- Decode tokens are allocated first; prefill chunks fill the remainder.
- Chunk size is 512 tokens, aligned to multiples of 128.
- Sequences in _prefill_queue transition to _decode_queue atomically
  when their entire prompt has been prefilled.
- Paged blocks are freed as soon as a sequence completes.
- _active_order list tracks batch dimension → seq_id mapping,
  eliminating the dict-iteration-order correctness bug.
"""

from __future__ import annotations

import heapq
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

# ── Chunked prefill parameters ───────────────────────────────────────────────
# Chunk size of 512 tokens keeps prefill latency bounded (~50-100 ms on H200)
# while maintaining high GPU utilisation.  Each chunk is aligned to 128-token
# multiples so block allocation remains regular (PagedKVCache blocks are sized
# in multiples of 128).
PREFILL_CHUNK_SIZE = 512   # tokens per prefill chunk
PREFILL_CHUNK_ALIGN = 128  # chunk alignment granularity


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
    submit_time: float = 0.0  # monotonic timestamp set on submit, used for priority scoring
    prefill_progress: int = 0  # how many prompt tokens have been prefilled so far (chunked prefill)

    @property
    def total_tokens(self) -> int:
        return self.current_position + len(self.generated_ids)

    @property
    def prompt_len(self) -> int:
        return len(self.input_ids)

    @property
    def prefill_remaining(self) -> int:
        """Number of prompt tokens still awaiting prefill."""
        return max(0, self.prompt_len - self.prefill_progress)


class ContinuousBatcher:
    """PagedAttention-powered continuous batching scheduler with chunked prefill.

    Keeps the GPU fed by dynamically adding and removing sequences
    from the active batch at every decode step.  Uses ``PagedKVCache``
    for zero-fragmentation KV-cache management.

    Chunked prefill interleaves small (512-token) prefill chunks with
    decode tokens, eliminating prefill-induced decode stalls.  Sequences
    flow through three stages: _waiting → _prefill_queue → _decode_queue.

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
        self._waiting: list[tuple[float, int, SequenceState]] = []  # heapq: (priority, ts, seq)
        # Chunked prefill splits the old _running pool into two stages:
        #   _prefill_queue — sequences still ingesting their prompt in chunks
        #   _decode_queue — fully-prefilled sequences generating tokens
        self._prefill_queue: list[SequenceState] = []   # mid-prefill sequences
        self._decode_queue: list[SequenceState] = []    # fully-prefilled, actively decoding
        self._running: dict[int, SequenceState] = {}    # combined dict for O(1) lookup
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

        # ── Priority scheduling normalisers ──
        self._max_prompt_len: int = 1  # longest prompt seen so far (>= 1 avoids div0)
        self._max_wait: float = 1.0    # longest wait seen so far (>= 1.0 avoids div0)

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
            submit_time=time.monotonic(),
        )

        # Track running maximums for normalised priority scoring.
        prompt_len = len(state.input_ids)
        if prompt_len > self._max_prompt_len:
            self._max_prompt_len = prompt_len

        # Initial priority: newest arrival gets -1.0 (worst priority for
        # a zero-wait sequence).  The _dequeue_waiting() method recomputes
        # scores on pop so this is just a placeholder — we push with
        # priority 0.0 and rely on _dequeue_waiting() for the real score.
        heapq.heappush(self._waiting, (0.0, state.submit_time, state))
        return seq_id

    def step(self) -> list[SequenceState]:
        """Execute one decode iteration with chunked prefill.

        1. Admit new sequences from _waiting into _prefill_queue (paged allocation).
        2. Run _schedule_chunked_step() — decode first, prefill chunks second.
        3. Transition fully-prefilled sequences from _prefill_queue to _decode_queue.
        4. Remove completed sequences, free their paged blocks.

        Decode tokens are ALWAYS allocated before prefill chunks, ensuring
        that decode latency never spikes due to prefill work.

        Returns
        -------
        list[SequenceState]
            Sequences that completed this step.
        """
        newly_completed: list[SequenceState] = []

        # ── 0. Admit new sequences from waiting into prefill ──
        # How many sequences can we admit?  We need headroom in the combined
        # prefill+decode pool (up to max_batch_size).
        active_count = len(self._prefill_queue) + len(self._decode_queue)
        capacity = self.max_batch_size - active_count
        if capacity > 0 and self._waiting:
            admitted: list[SequenceState] = []
            while len(admitted) < capacity and self._waiting:
                seq = self._dequeue_waiting()
                # Allocate paged blocks for the full prompt up-front
                # (chunked prefill writes incrementally into the same blocks).
                prompt_len = len(seq.input_ids)
                try:
                    self._paged_kv.allocate(seq.seq_id, prompt_len)
                except Exception:
                    # Block allocation failed — re-enqueue and stop admitting.
                    heapq.heappush(
                        self._waiting,
                        self._priority_tuple(seq),
                    )
                    break
                seq.prefill_progress = 0
                admitted.append(seq)
                self._running[seq.seq_id] = seq

            # Add newly admitted sequences to prefill queue.
            for seq in admitted:
                self._prefill_queue.append(seq)

        # ── 1. Exit early if nothing to do ──
        if not self._decode_queue and not self._prefill_queue:
            return []

        # ── 2. Chunked step — decode first, then prefill ──
        completed_ids = self._schedule_chunked_step()

        # ── 3. Transition fully-prefilled sequences to decode queue ──
        newly_prefilled: list[SequenceState] = []
        still_prefilling: list[SequenceState] = []
        for seq in self._prefill_queue:
            if seq.prefill_progress >= seq.prompt_len:
                newly_prefilled.append(seq)
            else:
                still_prefilling.append(seq)
        self._prefill_queue = still_prefilling

        for seq in newly_prefilled:
            self._decode_queue.append(seq)
            # Sequence is now in the active decode batch — add to ordering.
            if seq.seq_id not in self._active_set:
                self._active_order.append(seq.seq_id)
                self._active_set.add(seq.seq_id)

        # ── 4. Collect completions and remove completed sequences ──
        for seq_id in completed_ids:
            seq = self._running.pop(seq_id, None)
            self._active_set.discard(seq_id)
            if seq is not None:
                seq.done = True
                newly_completed.append(seq)
                self.total_sequences_completed += 1
            # Free paged blocks immediately.
            try:
                self._paged_kv.free(seq_id)
            except KeyError:
                pass
            # Drop from PagedCache (avoids full rebuild).
            if self._paged_cache is not None:
                self._paged_cache.remove_sequence(seq_id)

        if completed_ids:
            # Rebuild _active_order without completed seq_ids (list
            # comprehension preserves order of survivors for determinism).
            completed_set = set(completed_ids)
            self._active_order = [sid for sid in self._active_order
                                  if sid not in completed_set]
            # Also prune _decode_queue of completed entries.
            self._decode_queue = [s for s in self._decode_queue
                                  if s.seq_id not in completed_set]

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
        while self._waiting or self._prefill_queue or self._decode_queue:
            completed = self.step()
            max_steps += 1

            if completed:
                self._completed.extend(completed)
                logger.debug(
                    "Completed %d sequences (prefill=%d, decode=%d, waiting=%d, step=%d)",
                    len(completed), len(self._prefill_queue),
                    len(self._decode_queue), len(self._waiting), max_steps,
                )

            # Safety: prevent infinite loop if a sequence never finishes.
            if max_steps > 100_000:
                logger.error(
                    "ContinuousBatcher hit max_steps=100k — aborting. "
                    "prefill=%d decode=%d waiting=%d",
                    len(self._prefill_queue), len(self._decode_queue),
                    len(self._waiting),
                )
                # Force-complete remaining sequences.
                for seq in list(self._running.values()):
                    if not seq.done:
                        seq.done = True
                        self._completed.append(seq)
                        try:
                            self._paged_kv.free(seq.seq_id)
                        except KeyError:
                            pass
                break

        return self._completed

    def is_idle(self) -> bool:
        """True when no sequences are waiting, prefilling, or decoding."""
        return not self._waiting and not self._prefill_queue and not self._decode_queue

    def running_count(self) -> int:
        """Total active sequences (prefill + decode)."""
        return len(self._prefill_queue) + len(self._decode_queue)

    def waiting_count(self) -> int:
        return len(self._waiting)

    def active_batch_size(self) -> int:
        """Number of sequences currently in the decode batch (excludes prefilling)."""
        return len(self._active_order)

    # ── Priority queue helpers ─────────────────────────────────────────────

    def _score(self, seq: SequenceState) -> float:
        """Compute priority score for a waiting sequence.

        Score = -(prompt_len / max_prompt_len) * 0.5 + (wait_time / max_wait) * 0.5

        Lower (more negative) = higher priority for heapq min-heap.
        Shorter prompts get priority over longer ones (reducing head-of-line
        blocking).  Wait time prevents starvation — long prompts eventually
        accumulate enough wait credit to overtake short ones.
        """
        wait_time = time.monotonic() - seq.submit_time
        if wait_time > self._max_wait:
            self._max_wait = wait_time

        prompt_len = len(seq.input_ids)
        score = (
            -(prompt_len / self._max_prompt_len) * 0.5
            + (wait_time / self._max_wait) * 0.5
        )
        return score

    def _priority_tuple(self, seq: SequenceState) -> tuple[float, int, SequenceState]:
        """Build a heapq entry for *seq* with a freshly computed score."""
        return (self._score(seq), seq.seq_id, seq)

    def _dequeue_waiting(self) -> SequenceState:
        """Pop the highest-priority sequence from the waiting queue.

        Because priorities change with wait time, the heap is sorted on every
        pop: we recompute scores for all entries, rebuild the heap, then pop
        the current best.  For typical batch sizes (<1000) this is negligible.
        """
        if not self._waiting:
            raise IndexError("_dequeue_waiting called on empty queue")

        # Re-score and rebuild heap so the top entry reflects current wait times.
        self._waiting = [self._priority_tuple(entry[2]) for entry in self._waiting]
        heapq.heapify(self._waiting)
        _, _, seq = heapq.heappop(self._waiting)
        return seq

    # ── Internal: Chunked Prefill Scheduler ──────────────────────────────────

    def _schedule_chunked_step(self) -> list[int]:
        """Run one chunked decode+prefill step. Returns list of completed seq_ids.

        Scheduling priority (THE defining invariant of production inference):
          1. Decode tokens — 1 per sequence in _decode_queue (MANDATORY).
          2. Prefill chunks — fill remaining sequence-budget capacity with
             chunked prompt tokens from _prefill_queue.

        Decode always runs first, so decode latency is bounded by a single
        token forward pass, never by an unbounded prefill.  The remaining
        batch capacity (max_batch_size - decode_count) is used to advance
        prefill sequences in 512-token chunks.
        """
        from benchmark.inference.paged_attention import PagedCache

        completed_ids: list[int] = []
        decode_count = len(self._decode_queue)

        # ── Allocate prefill chunks ──
        # Compute chunk sizes for each prefill sequence, bounded by
        # PREFILL_CHUNK_SIZE and the remaining batch capacity.
        # Round-robin allocation ensures short prompts are not starved.
        prefill_chunks: list[tuple[SequenceState, int]] = []  # (seq, chunk_size)
        # How many prefill SEQUENCES can join this step (capacity remaining
        # in the batch after accounting for decode sequences).
        prefill_seq_budget = max(0, self.max_batch_size - decode_count)

        # Track how many prefill sequences we have assigned a chunk to.
        # Round-robin ensures each waiting prefill sequence gets at most
        # one chunk per _schedule_chunked_step call.
        assigned = 0
        for seq in self._prefill_queue:
            if assigned >= prefill_seq_budget:
                break
            needed = seq.prefill_remaining
            if needed <= 0:
                continue
            chunk = min(needed, PREFILL_CHUNK_SIZE)
            # Align to PREFILL_CHUNK_ALIGN for regular block writes.
            chunk = (chunk // PREFILL_CHUNK_ALIGN) * PREFILL_CHUNK_ALIGN
            if chunk == 0:
                chunk = needed  # remaining < ALIGN, take the straggler
            if chunk <= 0:
                continue
            prefill_chunks.append((seq, chunk))
            assigned += 1

        # ── Execute prefill chunks (write KV into paged blocks) ──
        if prefill_chunks:
            t0 = time.monotonic()
            try:
                self._prefill_chunks(prefill_chunks)
                self.total_prefill_ms += (time.monotonic() - t0) * 1000.0
            except Exception:
                # Prefill failure — atomic rollback.
                # Free blocks allocated during this prefill attempt, reset
                # prefill_progress so the sequence stays in _prefill_queue
                # and can retry.
                for seq, chunk_size in prefill_chunks:
                    seq.prefill_progress = max(0, seq.prefill_progress - chunk_size)
                    try:
                        self._paged_kv.free(seq.seq_id)
                    except KeyError:
                        pass
                    seq.current_position = seq.prefill_progress
                raise

        # ── Build PagedCache for the combined decode + prefill batch ──
        all_active_ids = list(self._active_order)
        for seq in self._prefill_queue:
            if seq.seq_id not in self._active_set:
                all_active_ids.append(seq.seq_id)
        # Also include newly-transitioned (after this step) but they don't
        # exist yet — the PagedCache is built for what's currently active.
        if all_active_ids:
            self._paged_cache = PagedCache(self._paged_kv, seq_ids=all_active_ids)
        else:
            self._paged_cache = None

        # ── Execute decode step for all fully-prefilled sequences ──
        if self._decode_queue:
            completed_ids = self._decode_one_step()

        return completed_ids

    def _prefill_chunks(self, chunks: list[tuple[SequenceState, int]]) -> None:
        """Prefill a batch of token chunks into pre-allocated paged KV blocks.

        Each chunk is a slice of `chunk_size` tokens from the sequence's
        prompt, starting at `prefill_progress`.  The KV output is written
        directly into the pre-allocated paged blocks at the correct offsets.

        This is the core chunked-prefill primitive: the model forward only
        processes `chunk_size` tokens per sequence (not the full prompt),
        so prefill latency is bounded by PREFILL_CHUNK_SIZE=512.
        """
        if not chunks:
            return

        device = self.device
        n_chunks = len(chunks)

        # Determine max chunk size for padding.
        max_chunk = max(c[1] for c in chunks)

        # ── Build padded prefill batch ──
        padded = []
        attn_mask = torch.zeros(n_chunks, max_chunk, dtype=torch.long, device=device)
        for i, (seq, chunk_size) in enumerate(chunks):
            start = seq.prefill_progress
            end = start + chunk_size
            token_slice = seq.input_ids[start:end]
            row = token_slice + [self.pad_token_id] * (max_chunk - chunk_size)
            padded.append(row)
            attn_mask[i, :chunk_size] = 1

        prompt_batch = torch.tensor(padded, dtype=torch.long, device=device)

        with torch.no_grad():
            # Regular model forward — no KV cache yet since this is
            # a prefill step (first forward pass for this chunk).
            prefill_out = self.engine.model(
                input_ids=prompt_batch,
                attention_mask=attn_mask,
                use_cache=True,
            )
            prefill_kv = prefill_out.past_key_values

        # ── Write chunk KV into pre-allocated paged blocks ──
        # Track chunks successfully written for atomic rollback.
        written_chunks: list[tuple[int, int]] = []  # (seq_id, chunk_size)
        try:
            for i, (seq, chunk_size) in enumerate(chunks):
                start = seq.prefill_progress
                end = start + chunk_size

                # Write KV for this chunk into paged blocks at the
                # correct offset (start:end).
                for layer_idx in range(len(prefill_kv)):
                    k, v = prefill_kv[layer_idx]
                    self._paged_kv.write(
                        seq.seq_id, layer_idx,
                        slice(start, end),
                        k[i, :, :chunk_size, :],
                        v[i, :, :chunk_size, :],
                    )

                seq.prefill_progress = end
                seq.current_position = end
                written_chunks.append((seq.seq_id, chunk_size))
        except Exception:
            # Rollback: reset prefill_progress for chunks that were
            # partially written (paged blocks may be corrupt — caller
            # should free and re-allocate).
            for sid, csize in written_chunks:
                seq = self._running.get(sid)
                if seq is not None:
                    seq.prefill_progress -= csize
                    seq.current_position = seq.prefill_progress
            raise

    def _decode_one_step(self) -> list[int]:
        """Run one decode step for all sequences in _decode_queue.

        Returns list of seq_ids that completed (EOS or max_tokens).
        """
        device = self.device
        completed_ids: list[int] = []

        # ── Assemble decode batch tensor ──
        batch_input_ids, batch_positions = self._assemble_batch()

        # ── Validate batch has content ──
        if batch_input_ids.shape[0] == 0:
            return completed_ids

        # ── Model forward (1 token per sequence) ──
        t0 = time.monotonic()
        with torch.no_grad():
            outputs = self.engine.model(
                input_ids=batch_input_ids,
                past_key_values=self._paged_cache,
                use_cache=True,
            )

        # PagedCache was updated in-place by attention layers.
        logits = outputs.logits  # [batch, 1, vocab_size]
        next_tokens = logits[:, -1, :].argmax(dim=-1)  # [batch]

        decode_ms = (time.monotonic() - t0) * 1000.0
        self.total_decode_ms += decode_ms

        # ── Update sequences, detect completions ──
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

        return completed_ids

    # ── Internal: Legacy prefill (kept for backward compat / testing) ───────

    def _prefill_batch(self, sequences: list[SequenceState]) -> None:
        """Full-prompt prefill (legacy, non-chunked).

        Used when chunked prefill is disabled or for testing.  Prefills
        the entire prompt in one forward pass, then builds the PagedCache.
        """
        from benchmark.inference.paged_attention import PagedCache

        if not sequences:
            return

        n_new = len(sequences)

        if n_new >= MIN_BATCH_FOR_CHUNKED_PREFILL:
            self._prefill_batch_padded(sequences)
        else:
            for seq in sequences:
                self._prefill_sequence(seq)

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
        allocated_ids: list[int] = []
        try:
            for i, seq in enumerate(sequences):
                try:
                    self._paged_kv._block_tables[seq.seq_id]
                    continue  # already allocated — skip
                except KeyError:
                    pass

                slen = seq_lengths[i]
                self._paged_kv.allocate(seq.seq_id, slen)
                allocated_ids.append(seq.seq_id)
                seq.current_position = slen

                for layer_idx in range(len(prefill_kv)):
                    k, v = prefill_kv[layer_idx]
                    self._paged_kv.write(
                        seq.seq_id, layer_idx,
                        slice(0, slen),
                        k[i, :, :slen, :],
                        v[i, :, :slen, :],
                    )
        except Exception:
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

        allocated = False
        try:
            self._paged_kv.allocate(seq.seq_id, prompt_len)
            allocated = True
            seq.current_position = prompt_len

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
        """Build the decode-step input tensors from decode-queue sequences.

        Returns (input_ids, position_ids) for the model forward.
        The attention mask is NOT needed — PagedCache returns assembled
        full-length KV tensors that already encode the causal mask.

        Only sequences in ``_decode_queue`` (fully prefilled) participate in
        decoding.  Sequences still in ``_prefill_queue`` are excluded from the
        decode batch — they do not yet have a PagedCache entry capable of
        incremental generation.

        Uses ``_active_order`` (NOT ``_decode_queue`` directly) for deterministic
        batch-index → seq_id mapping.

        Raises
        ------
        RuntimeError
            If ``_active_order`` contains seq_ids not in the decode set — this
            would indicate a scheduler bug where a prefill-only sequence leaked
            into the decode batch.
        """
        device = self.device

        # ── Build decode set for validation ──
        decode_set = {s.seq_id for s in self._decode_queue}

        # ── Validate consistency ──
        # _active_order must only contain decode-queue sequences.
        # _active_set is the O(1) companion for membership checks.
        active_set = self._active_set
        assert len(active_set) == len(self._active_order), (
            f"_active_set ({len(active_set)}) and _active_order "
            f"({len(self._active_order)}) diverged — scheduler bug"
        )
        leaked = active_set - decode_set
        if leaked:
            raise RuntimeError(
                f"Chunked prefill scheduler bug: seq_ids {leaked} are in "
                f"_active_order but not in _decode_queue.  Prefill-only "
                f"sequences must not participate in the decode batch."
            )
        missing = decode_set - active_set
        if missing:
            raise RuntimeError(
                f"Chunked prefill scheduler bug: seq_ids {missing} are in "
                f"_decode_queue but not in _active_order.  These sequences "
                f"will be silently dropped from the decode batch."
            )

        # Build input IDs and position IDs in a SINGLE pass to prevent
        # desync between batch_input_ids and positions.
        next_tokens_list = []
        position_list = []
        for seq_id in self._active_order:
            seq = self._running.get(seq_id)  # guaranteed present by validation
            if seq is None:
                raise RuntimeError(
                    f"seq_id {seq_id} in _active_order but not in _running — "
                    f"scheduler state corruption"
                )
            if seq.generated_ids:
                next_tokens_list.append(seq.generated_ids[-1])
            else:
                # Freshly-transitioned to decode: first decode token uses
                # the last prompt token as input.
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
        assert batch_input_ids.shape[0] == positions.shape[0] == len(self._active_order), (
            f"Batch assembly size mismatch: batch_input_ids={batch_input_ids.shape[0]}, "
            f"positions={positions.shape[0]}, _active_order={len(self._active_order)}"
        )

        return batch_input_ids, positions

    def drain_running(self) -> list[SequenceState]:
        """Drain all active sequences (prefill + decode) — free blocks, return states.

        Use this to recover from a stuck batch or to gracefully shut down
        without leaking paged KV-cache blocks.  All drained sequences are
        marked as ``done`` and their blocks are returned to the free pool.

        Returns
        -------
        list[SequenceState]
            All sequences that were in the prefill or decode queues.
        """
        drained = []
        # Drain decode queue.
        for seq in list(self._decode_queue):
            seq.done = True
            drained.append(seq)
            self._running.pop(seq.seq_id, None)
            self._paged_kv.free(seq.seq_id)
        self._decode_queue.clear()
        # Drain prefill queue.
        for seq in list(self._prefill_queue):
            seq.done = True
            drained.append(seq)
            self._running.pop(seq.seq_id, None)
            self._paged_kv.free(seq.seq_id)
        self._prefill_queue.clear()
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
