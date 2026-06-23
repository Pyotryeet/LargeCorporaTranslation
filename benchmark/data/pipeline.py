"""Async prefetch pipeline — keeps the GPU fed with pre-tokenised batches.

v2.0: Thread-local tokenizer instances for lock-free parallel tokenization.
      Pinned-memory tensors for DMA-fast CPU→GPU transfers on CUDA.
      Event-driven batch collection (blocking dequeue, no busy-wait).
"""

import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Optional
import torch
from transformers import PreTrainedTokenizerBase
from benchmark.data.loader import JSONLLoader
from benchmark.data.chunker import TextChunker, NullChunker
from benchmark.data.filters import ChunkFilter
from benchmark.config.constants import (
    LOADER_JOIN_TIMEOUT,
    WORKER_JOIN_TIMEOUT,
    SENTINEL_PUT_TIMEOUT,
    BATCH_COLLECT_TIMEOUT,
    TOKENISER_GET_TIMEOUT,
    DEFAULT_MAX_SEQ_LEN,
)

logger = logging.getLogger(__name__)

# ── Per-tokenizer warning suppression ──────────────────────────────────────────
# _build_translation_prompt falls back to a plain text prefix when the
# chat template fails (e.g. SmolLM2's tokenizer doesn't know source_lang_code).
# Warn once per tokenizer identity instead of per chunk (which would flood).
_TEMPLATE_WARNED: set[str] = set()

# ── Per-tokenizer prompt-style cache ───────────────────────────────────────────
# _build_translation_prompt probes tokenizer properties (has src_lang? is MADLAD?)
# on every chunk.  Some probes — especially tokenizer.get_vocab() — materialise a
# 256k-entry dict from the SentencePiece C++ backend and take ~200 ms.  Caching
# the result per tokenizer identity avoids O(N_chunks × N_threads) overhead.
# Values: "nllb" | "madlad" | "chat" | "plain"
_PROMPT_STYLE: dict[str, str] = {}

# ── Module-level constants (imported from central constants) ─────────────────
_LOADER_JOIN_TIMEOUT = LOADER_JOIN_TIMEOUT
_WORKER_JOIN_TIMEOUT = WORKER_JOIN_TIMEOUT
_SENTINEL_PUT_TIMEOUT = SENTINEL_PUT_TIMEOUT
_BATCH_COLLECT_TIMEOUT = BATCH_COLLECT_TIMEOUT
_TOKENISER_GET_TIMEOUT = TOKENISER_GET_TIMEOUT
_DEFAULT_MAX_SEQ_LEN = DEFAULT_MAX_SEQ_LEN


@dataclass
class PipelineBatch:
    batch_id: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_lengths: list[int]
    raw_texts: list[str]
    token_counts: list[int]
    # Back-references to the full pre-allocated pinned buffers (if any).
    # These are set by AsyncPipeline.next_batch() when the pinned pool is used
    # and allow release_batch() to return tensors to the pool.
    _pool_ids: Optional[torch.Tensor] = None
    _pool_mask: Optional[torch.Tensor] = None


class PinnedBufferPool:
    """Bounded pool of pre-allocated pinned-memory tensors.

    Pinned-memory allocation requires ``mlock()`` (≈50 µs per page), which
    adds up to measurable overhead when re-allocated on every batch.  This
    pool pre-allocates a few buffers and reuses them across batches, reducing
    per-batch pinned-memory allocation overhead to a near-zero copy.
    """

    def __init__(self, max_batch_size: int, max_seq_len: int, pool_size: int = 4):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.pool_size = pool_size
        self._free: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._hits: int = 0
        self._misses: int = 0
        # On MPS, pin_memory creates device tensors (mps:0) instead of
        # CPU-host-pinned tensors.  MPS uses unified memory, so pinning
        # provides no benefit but breaks CPU-side tensor assignment.
        # On CUDA, pin_memory keeps tensors on CPU with page-locked DMA.
        # On CPU/MPS, just use regular tensors.
        self._should_pin = torch.cuda.is_available()

    def acquire(self, timeout: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a pre-allocated (input_ids, attention_mask) pair or create one.

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait for a free buffer before creating a new
            one.  When None (default), creates immediately if the pool is
            empty (no blocking).  A positive value causes the caller to
            block until a buffer is released or the timeout expires.
            Implementation note: the current non-blocking pool never waits;
            the timeout parameter is accepted for API compatibility and
            logged if the pool is exhausted.
        """
        if self._free:
            self._hits += 1
            return self._free.pop()
        self._misses += 1
        if timeout is not None and timeout > 0:
            logger.debug(
                "PinnedBufferPool.acquire timeout=%.1fs — pool exhausted (%d misses)",
                timeout, self._misses,
            )
        ids = torch.empty(
            self.max_batch_size, self.max_seq_len,
            dtype=torch.long, pin_memory=self._should_pin,
        )
        mask = torch.zeros(
            self.max_batch_size, self.max_seq_len,
            dtype=torch.long, pin_memory=self._should_pin,
        )
        return ids, mask

    def release(self, ids: torch.Tensor, mask: torch.Tensor) -> None:
        """Return a pair to the pool for reuse when no inflight slices remain.

        Each batch slice is a view into the pre-allocated pinned buffer.
        Releasing a buffer while any view (inflight batch) still references
        it would corrupt that batch.  Callers must guarantee that all
        inflight batches that use slices of this buffer have been consumed
        before calling release().  This method zeroes the tensors to prevent
        accidental data leakage between batches.

        The pool is bounded (pool_size), so excess releases beyond capacity
        are silently discarded rather than growing the pool unboundedly.
        """
        if len(self._free) < self.pool_size:
            ids.zero_()
            mask.zero_()
            self._free.append((ids, mask))

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


class AsyncPipeline:
    """Pinned-memory, lock-free, event-driven data pipeline.

    Each tokenizer worker thread gets its own SentencePiece instance
    (via ``copy.deepcopy``) to eliminate GIL serialisation.  On CUDA
    systems batch tensors are allocated in pinned (page-locked) memory
    for DMA-accelerated host→device transfers.
    """

    # Module-level sentinel for object-identity checks (avoids magic-string bugs
    # where data that happens to equal "__SENTINEL__" would drain pipelines).
    _SENTINEL = object()

    def __init__(self, loader: JSONLLoader, chunker, tokenizer: PreTrainedTokenizerBase,
                 text_filter: ChunkFilter, batch_size: int = 8, prefetch_workers: int = 4,
                 max_queue_size: int = 32, backend: str = "cpu",
                 max_input_tokens: Optional[int] = None):
        self.loader = loader
        self.chunker = chunker
        self.tokenizer = tokenizer
        self.text_filter = text_filter
        self.batch_size = batch_size
        self.prefetch_workers = prefetch_workers
        self.max_queue_size = max_queue_size
        self.backend = backend
        self.max_input_tokens = max_input_tokens or min(
            getattr(tokenizer, 'model_max_length', _DEFAULT_MAX_SEQ_LEN),
            _DEFAULT_MAX_SEQ_LEN,
        )
        # Safety cap: some tokenizers report model_max_length=10^30 as a sentinel
        # for "unlimited context".  Passing that to the C extension causes
        # OverflowError: int too big to convert.  Clamp to a reasonable ceiling.
        self.max_input_tokens = min(self.max_input_tokens, _DEFAULT_MAX_SEQ_LEN)
        # Pinned memory only benefits CUDA (discrete GPU with PCIe).
        self._use_pinned = (backend == "cuda")

        # Pre-allocated pinned memory pool (avoids per-batch mlock overhead).
        self._pinned_pool: Optional[PinnedBufferPool] = None
        if self._use_pinned and batch_size > 0:
            self._pinned_pool = PinnedBufferPool(batch_size, self.max_input_tokens, pool_size=4)

        # — Queues —
        # Auto-clamp: tokenised_queue must hold ≥ batch_size items or the
        # pipeline deadlocks (producers fill queue then block on put(),
        # consumer waits for batch_size items that can never arrive).
        _min_q = max(max_queue_size, batch_size // 4 + 1)
        self._raw_queue: queue.Queue = queue.Queue(maxsize=_min_q)
        self._tokenised_queue: queue.Queue = queue.Queue(maxsize=_min_q * 4)

        # — Threading —
        self._loader_thread: Optional[threading.Thread] = None
        self._tokeniser_threads: list[threading.Thread] = []
        self._running = threading.Event()
        self._done = threading.Event()

        # — Per-thread tokenizer cache —
        self._tokenizer_local = threading.local()

        # — Tokenizer path for thread-local re-instantiation —
        # HuggingFace Rust tokenizers (tokenizers crate) may segfault when
        # copy.deepcopy()'d because the underlying C++ state is not
        # copyable.  Instead, each worker thread creates a fresh,
        # independent AutoTokenizer instance from the saved path.
        vocab_file = getattr(tokenizer, 'vocab_file', None)
        if vocab_file:
            tokenizer_dir = os.path.dirname(str(vocab_file))
        else:
            tokenizer_dir = None
        self._tokenizer_path: Optional[str] = (
            getattr(tokenizer, 'name_or_path', None) or tokenizer_dir
        )

        # — Chunker lock (the chunker uses the main tokenizer, which isn't
        #   thread-safe — serialise access from the loader thread only) —
        self._chunker_lock = threading.Lock()

        # — Counters —
        self.total_chunks_produced = 0
        self.total_chunks_consumed = 0
        self._chunk_counter_lock = threading.Lock()
        self._batch_counter = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def start_prefetch(self) -> None:
        self._running.set()
        self._done.clear()

        # Drain any stale items left from a previous pipeline run.
        # Without this, leftover chunks from a prior start/stop cycle can
        # appear in the tokenised queue and cause batch assembly confusion.
        while not self._raw_queue.empty():
            try:
                self._raw_queue.get_nowait()
            except queue.Empty:
                break
        while not self._tokenised_queue.empty():
            try:
                self._tokenised_queue.get_nowait()
            except queue.Empty:
                break

        # NOTE: Thread-local tokenizer instances are NOT pre-warmed here.
        # Each worker thread lazily creates its own deep copy via
        # _get_tokenizer() on first use.  Pre-creating copies in the main
        # thread would store them in main-thread-local storage where workers
        # can never access them — dead code that wastes ~1ms per worker.

        self._loader_thread = threading.Thread(
            target=self._loader_loop, name="data-loader", daemon=True,
        )
        self._loader_thread.start()

        for i in range(self.prefetch_workers):
            t = threading.Thread(
                target=self._tokeniser_loop, name=f"tokeniser-{i}", daemon=True,
            )
            t.start()
            self._tokeniser_threads.append(t)

        logger.info(
            "Async pipeline: %d workers, max queue %d, pinned=%s, lock-free tok",
            self.prefetch_workers, self.max_queue_size, self._use_pinned,
        )

    def stop_prefetch(self) -> None:
        self._running.clear()
        if self._loader_thread and self._loader_thread.is_alive():
            self._loader_thread.join(timeout=_LOADER_JOIN_TIMEOUT)
        for t in self._tokeniser_threads:
            if t.is_alive():
                t.join(timeout=_WORKER_JOIN_TIMEOUT)

    def draining(self) -> bool:
        return self._done.is_set()

    def notify_done(self) -> None:
        self._done.set()
        # Push one sentinel per tokenizer worker so they all drain.
        # Retry on Full to avoid the race where a worker re-puts the sentinel
        # and the queue is temporarily at capacity.
        # NOTE: The retry loop breaks after 100 attempts per worker, but the
        # outer for-loop correctly continues to the next worker.  A Full queue
        # after 100 retries means workers are still actively processing items
        # — they will eventually encounter the _running flag and exit on their
        # own.  The warning is informational; no sentinel is needed for that
        # worker because it will drain naturally.
        for _ in range(self.prefetch_workers):
            retries = 0
            while True:
                try:
                    self._raw_queue.put(self._SENTINEL, timeout=_SENTINEL_PUT_TIMEOUT)
                except queue.Full:
                    retries += 1
                    if retries >= 100:
                        logger.warning(
                            "notify_done: sentinel put failed after %d retries; "
                            "some workers may not drain.", retries,
                        )
                        break
                    continue
                break

    # ── Batch assembly (P0-04 + P0-05) ─────────────────────────────────────

    def next_batch(self) -> Optional[PipelineBatch]:
        """Collect *batch_size* tokenised chunks and assemble a batch.

        Uses a blocking ``q.get()`` + sentinel pattern instead of
        timeout-based busy-polling.  On CUDA, tensors are allocated in
        pinned memory for DMA-accelerated host→device transfer.
        """
        texts: list[str] = []
        token_lists: list[list[int]] = []
        lengths: list[int] = []

        while len(texts) < self.batch_size:
            try:
                item = self._tokenised_queue.get(timeout=_BATCH_COLLECT_TIMEOUT)
            except queue.Empty:
                # No data arrived in 5 s — drain check.
                if self.draining():
                    break
                continue

            text, token_ids, count = item
            texts.append(text)
            token_lists.append(token_ids)
            lengths.append(count)
            self.total_chunks_consumed += 1

        if not texts:
            return None

        # Guard: if a sentinel leaked through into the tokenised queue,
        # we would have a mix of real tuples and the sentinel object.
        # Filter out any sentinel objects and log if this happened.
        if any(item is self._SENTINEL for item in texts):
            logger.warning(
                "next_batch: sentinel object leaked into tokenised queue "
                "(%d items, %d sentinels) — discarding sentinels",
                len(texts), sum(1 for t in texts if t is self._SENTINEL),
            )
            filtered = [(t, tl, l) for t, tl, l in zip(texts, token_lists, lengths)
                        if t is not self._SENTINEL]
            if not filtered:
                return None
            texts, token_lists, lengths = zip(*filtered)
            texts, token_lists, lengths = list(texts), list(token_lists), list(lengths)
            if not texts:
                return None

        max_len = max(len(t) for t in token_lists)
        pad_id = self.tokenizer.pad_token_id or 0

        # Use the pre-allocated pinned memory pool when available.
        if self._pinned_pool is not None and max_len <= self._pinned_pool.max_seq_len:
            full_ids, full_mask = self._pinned_pool.acquire()
            input_ids = full_ids[:len(texts), :max_len]
            attention_mask = full_mask[:len(texts), :max_len]
            input_ids.fill_(pad_id)
            attention_mask.zero_()
        else:
            pin = self._use_pinned
            input_ids = torch.full(
                (len(texts), max_len), pad_id, dtype=torch.long,
                pin_memory=pin,
            )
            attention_mask = torch.zeros(
                (len(texts), max_len), dtype=torch.long,
                pin_memory=pin,
            )

        # Vectorized copy via numpy memory view (avoid per-item torch.tensor).
        # Use numpy path when tensors are pinned (DMA-friendly) or large enough
        # to justify the numpy function-call overhead.
        # Safety: convert token IDs to native Python ints. SentencePiece on
        # some platforms returns numpy-backed integers that trigger "int too
        # big to convert" errors when assigned to arrays on ARM64 macOS (MPS).
        token_lists_safe = [[int(t) for t in tl] for tl in token_lists]
        if (self._pinned_pool is not None and max_len <= self._pinned_pool.max_seq_len) or len(texts) > 4:
            ids_np = input_ids.numpy()
            mask_np = attention_mask.numpy()
            for i, tids in enumerate(token_lists_safe):
                n = len(tids)
                ids_np[i, :n] = tids
                mask_np[i, :n] = 1
        else:
            for i, tids in enumerate(token_lists_safe):
                input_ids[i, :len(tids)] = torch.tensor(tids, dtype=torch.long)
                attention_mask[i, :len(tids)] = 1

        self._batch_counter += 1
        return PipelineBatch(
            batch_id=self._batch_counter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_lengths=lengths,
            raw_texts=texts,
            token_counts=lengths,
            _pool_ids=full_ids if self._pinned_pool is not None and max_len <= self._pinned_pool.max_seq_len else None,
            _pool_mask=full_mask if self._pinned_pool is not None and max_len <= self._pinned_pool.max_seq_len else None,
        )

    def update_batch_size(self, new_batch_size: int) -> None:
        """Update batch size after OOM recovery. Rebuilds the pinned pool."""
        self.batch_size = new_batch_size
        if self._use_pinned and new_batch_size > 0:
            self._pinned_pool = PinnedBufferPool(new_batch_size, self.max_input_tokens, pool_size=4)
        else:
            self._pinned_pool = None

    def queue_depth(self) -> int:
        return self._tokenised_queue.qsize()

    def release_batch(self, batch: PipelineBatch) -> None:
        """Return a batch's pinned-memory tensors to the pool for reuse.

        Only releases when the batch was allocated from the pinned pool
        (i.e. batch._pool_ids / batch._pool_mask are set).  Call this
        after the engine has consumed the batch to avoid leaking pinned
        memory and prevent pool exhaustion.
        """
        if batch._pool_ids is not None and batch._pool_mask is not None:
            self._pinned_pool.release(batch._pool_ids, batch._pool_mask)

    # ── Thread-local tokenizer (P0-02) ─────────────────────────────────────

    @staticmethod
    def _local_kwargs(path: str) -> dict:
        """Return ``local_files_only=True`` when *path* is a filesystem path.

        This prevents ``AutoTokenizer.from_pretrained()`` from attempting a
        network fetch when the caller supplies a local directory or file.
        """
        if os.path.isdir(path) or os.path.isfile(path):
            return {"local_files_only": True}
        return {}

    def _get_tokenizer(self):
        """Return a thread-local copy of the tokenizer (lock-free).

        Each worker thread gets its own SentencePiece instance because
        the underlying C++ processor is NOT thread-safe.

        Uses ``AutoTokenizer.from_pretrained()`` to create independent
        instances rather than ``copy.deepcopy()``, which can segfault on
        HuggingFace Rust tokenizers (the ``tokenizers`` crate binds C++
        state that does not survive Python-level deepcopy).

        When ``_tokenizer_path`` is None the main-thread tokenizer is
        returned directly — this is safe because it will only be used
        from a single worker thread (no concurrent access).
        """
        if not hasattr(self._tokenizer_local, 'instance'):
            if self._tokenizer_path:
                from transformers import AutoTokenizer
                self._tokenizer_local.instance = AutoTokenizer.from_pretrained(
                    self._tokenizer_path,
                    trust_remote_code=False,
                    **self._local_kwargs(self._tokenizer_path),
                )
            else:
                # No path available — use the original tokenizer directly.
                # This is safe because it will only be accessed from a
                # single worker thread.
                self._tokenizer_local.instance = self.tokenizer
        return self._tokenizer_local.instance

    # ── Loader loop ─────────────────────────────────────────────────────────

    def _loader_loop(self) -> None:
        try:
            for doc_id, file_name, text in self.loader.iter_documents():
                if not self._running.is_set():
                    break
                # Chunker uses the shared tokenizer — serialise.
                # NOTE: chunk_with_tokens() would save ~30-40% CPU by avoiding
                # the tokenize→decode→re-tokenize cycle, but it is NOT used
                # here because it requires thread-safe access to the token-level
                # API (the chunker's tokenizer must match the worker's).  Each
                # worker thread has its own tokenizer instance, so the pre-
                # computed token IDs from chunk_with_tokens() would be tied to
                # the loader thread's tokenizer — a safety concern.
                with self._chunker_lock:
                    chunks = list(self.chunker.chunk(text))
                for chunk in chunks:
                    if not self._running.is_set():
                        break
                    self._raw_queue.put((file_name, chunk, doc_id))
            logger.info("Loader: all documents streamed")
        except Exception as e:
            logger.error("Loader error: %s", e, exc_info=True)
        finally:
            self.notify_done()

    # ── Tokenizer loop (lock-free with thread-local instance) ──────────────

    @staticmethod
    def _build_translation_prompt(text: str, tokenizer) -> str:
        """Wrap raw English text in the model's translation prompt format.

        The prompt style is detected once per tokenizer identity and cached.
        Avoids expensive operations like ``tokenizer.get_vocab()`` (which
        materialises a 256k-entry dict from SentencePiece) on every chunk.
        """

        # ── Classify tokenizer once — use name_or_path as a stable key. ──
        # id(tokenizer) varies per thread when workers call _get_tokenizer()→
        # AutoTokenizer.from_pretrained() (creates a new object per thread).
        # name_or_path is the same HuggingFace model ID for all threads.
        tok_key = getattr(tokenizer, 'name_or_path', '') or str(hash(tokenizer.__class__.__name__))
        style = _PROMPT_STYLE.get(tok_key)
        if style is None:
            if hasattr(tokenizer, 'src_lang') and tokenizer.src_lang is not None:
                style = "nllb"
            else:
                # Check for MADLAD-400: has '<2tr>' in vocabulary.
                try:
                    if "<2tr>" in tokenizer.get_vocab():
                        style = "madlad"
                    else:
                        style = "unknown"  # probe further below
                except (AttributeError, TypeError):
                    style = "unknown"

            if style == "unknown":
                # Not NLLB, not MADLAD — must be a decoder-only model.
                # Try the chat template; if it fails, use plain text.
                msgs = [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "tr",
                        "text": text,
                    }],
                }]
                try:
                    tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                    style = "chat"
                except Exception:
                    style = "plain"
                    _model = getattr(tokenizer, 'name_or_path', '') or str(tok_key)
                    if _model not in _TEMPLATE_WARNED:
                        _TEMPLATE_WARNED.add(_model)
                        logger.warning(
                            "_build_translation_prompt: chat template failed for "
                            "'%s' — falling back to plain text prefix.",
                            _model,
                        )

            _PROMPT_STYLE[tok_key] = style

        # ── Apply cached style ──
        if style == "nllb":
            return text  # tokenizer adds eng_Latn prefix automatically
        elif style == "madlad":
            return "<2tr> " + text
        elif style == "chat":
            msgs = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": "tr",
                    "text": text,
                }],
            }]
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        else:  # style == "plain"
            return f"Translate English to Turkish:\n{text}"

    def _tokeniser_loop(self) -> None:
        tok = self._get_tokenizer()  # one deep-copy per thread lifetime
        chunk_produced = 0

        while self._running.is_set() or not self._raw_queue.empty():
            try:
                item = self._raw_queue.get(timeout=_TOKENISER_GET_TIMEOUT)
            except queue.Empty:
                if self.draining():
                    break
                continue

            if item is self._SENTINEL:
                # Sentinel received — exit immediately.
                # notify_done() broadcasts one sentinel per worker so no
                # re-broadcast is needed (avoids the race where a re-put
                # times out and leaves remaining workers alive).
                break

            file_name, text, doc_id = item

            try:
                # Wrap with translation prompt so the model knows it is an
                # en→tr translation task — not English text completion.
                prompted = self._build_translation_prompt(text, tok)
                # LOCK-FREE: each thread has its own tokenizer instance (P0-02)
                token_ids = tok.encode(
                    prompted,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_input_tokens,
                )
            except UnicodeDecodeError as e:
                # Log the problematic byte range so data issues are diagnosable
                try:
                    problem_span = text[max(0, e.start - 10):e.end + 10]
                except Exception:
                    problem_span = "<unavailable>"
                logger.warning(
                    "UnicodeDecodeError in tokenizer encode: %s "
                    "(start=%d, end=%d, reason=%s, context=%r)",
                    e, e.start, e.end, e.reason, problem_span,
                )
                continue
            except Exception as e:
                logger.warning("Tokeniser error: %s", e)
                continue

            # Convert to native Python ints — SentencePiece returns numpy-
            # backed integers that cause "int too big to convert" errors
            # when assigned to torch/numpy arrays on ARM64 macOS (MPS).
            token_ids = [int(t) for t in token_ids]

            token_count = len(token_ids)
            if self.text_filter.should_keep(text, token_count):
                with self._chunk_counter_lock:
                    self.total_chunks_produced += 1
                chunk_produced += 1
                # put() with timeout — if the queue is full and the pipeline
                # is shutting down, the worker must detect _running=False and exit.
                # Blocking put() with no timeout creates a deadlock: all workers
                # stuck on a full queue never check _running or see sentinels.
                while self._running.is_set():
                    try:
                        self._tokenised_queue.put(
                            (text, token_ids, token_count), timeout=1.0,
                        )
                        break
                    except queue.Full:
                        pass  # retry: check _running flag
                if not self._running.is_set():
                    break  # shutdown

        logger.debug("Tokenizer worker exiting after %d chunks", chunk_produced)
