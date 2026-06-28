"""Auto batch-size tuner with backend-aware OOM handling.

v2.0: Performance-model initial guess to reduce binary-search steps.
      Pinned-memory test tensors for realistic CUDA transfer simulation.
v3.3: Robust MPS OOM detection — handles PyTorch 2.x MPS error variants.
      Batch tuning failure gracefully falls back to batch_size=1.
v3.6: MPS uses a single-probe fast path to avoid MPSGraph compilation cache
      accumulation (binary search creates 15-25 GB of unfreeable memory).
      CUDA path uses an optional TPS-aware sweep (gated by BATCH_TUNE_TPS_SWEEP
      env var) to find throughput-maximizing batch size for HBM-bound models.
"""

import logging
import os
import torch
import torch.nn as nn

from benchmark.config.constants import DEFAULT_HEAD_DIM, DEFAULT_NUM_LAYERS, DEFAULT_NUM_KV_HEADS

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
# Performance model: estimate KV-cache memory per sequence.
# Formula: num_layers * 2 * num_kv_heads * head_dim * max_seq_len * bytes_per_elem
# These are conservative defaults tuned for ~7B-12B parameter models.
KV_CACHE_NUM_LAYERS = DEFAULT_NUM_LAYERS
KV_CACHE_BYTES_PER_ELEM = 2  # bf16/fp16 = 2 bytes
KV_CACHE_KV_FACTOR = 2        # K + V tensors
KV_CACHE_MAX_SEQ_LEN = 512
KV_CACHE_HEAD_DIM = DEFAULT_HEAD_DIM  # per-head dimension, NOT hidden_size
KV_CACHE_NUM_KV_HEADS = DEFAULT_NUM_KV_HEADS
MEMORY_USABLE_FRACTION = 0.75   # for large GPUs (>80 GB): measured 2026-06-24 at 0.939 for 4B on H200. 0.75 is conservative (136 GiB usable of 139.8 GiB total). See M1.1.
MEMORY_USABLE_FRACTION_SMALL = 0.30  # for GPU memory ≤ 80 GB
DEFAULT_SAFETY_MARGIN = 0.05  # reduced from 0.15: measured 2026-06-24 on H200 — 15% margin leaves 8.3% throughput on table. At H200 memory sizes, CUDA OOM is a clean torch.cuda.OutOfMemoryError (not a system crash). 5% is safe. See M1.3.
DEFAULT_MAX_CUDA = 2048          # H200-class: 2× 143 GB — go big
DEFAULT_MAX_CUDA_SMALL = 128     # ≤ 80 GB GPU — conservative
DEFAULT_MAX_MPS = 32
DEFAULT_MAX_INPUT_TOKENS = 512
TEST_SENTENCE_MULTIPLIER = 10
TEST_GENERATION_TOKENS = 10
PIN_MEMORY_BACKEND = "cuda"
BYTES_PER_KB = 1024
BYTES_PER_MB = BYTES_PER_KB * BYTES_PER_KB  # 1,048,576
MIN_BATCH_SIZE = 1
BINARY_SEARCH_DIVISOR = 2
FALLBACK_PAD_TOKEN_ID = 0
TRANSFORMERS_VERBOSITY_ERROR = "error"
LARGE_GPU_THRESHOLD_GB = 80     # per-GPU threshold for "large" GPU

# MPS OOM error class (exists in PyTorch 2.1-2.4, removed in 2.5+).
# When absent, use a broader catch that handles RuntimeError variants
# including "Placeholder storage has not been allocated on MPS".
try:
    _MPS_OOM = torch.mps.OutOfMemoryError
except AttributeError:
    class _MPS_OOM_Sentinel(Exception):
        pass
    _MPS_OOM = _MPS_OOM_Sentinel

# Under PyTorch 2.5+, MPS errors surface as RuntimeError with specific
# messages rather than typed OOM exceptions.
_MPS_OOM_MESSAGES = (
    "Placeholder storage has not been allocated on MPS",
    "MPS out of memory",
    "not enough memory on the device",
)


class BatchSizeTuner:
    """Auto batch-size tuner that discovers the maximum viable batch size for a given
    model and backend via binary search (CUDA) or single-probe (MPS).

    On CUDA, it performs a binary search up to a per-GPU-memory cap, optionally
    followed by a TPS-aware sweep to pick the throughput-maximizing batch size
    rather than the largest one.  On MPS, it probes the cap once to avoid
    accumulating unfreeable MPSGraph compilation caches.

    The tuner uses a performance model based on model config (or hardcoded
    defaults) to produce an informational KV-cache estimate, but the estimate
    does NOT cap the binary search — the actual OOM boundary is found
    empirically.

    Public API:
        tune(model, tokenizer, device, backend, max_input_tokens=512) -> int

    Attributes (set during tune()):
        safety_margin (float): Fraction of max-viable batch to reserve.
        _default_max_cuda (int): Hard upper bound for CUDA batch size.
        _default_max_mps (int): Hard upper bound for MPS batch size.
    """
    def __init__(self, safety_margin: float = DEFAULT_SAFETY_MARGIN, max_cuda: int = DEFAULT_MAX_CUDA, max_mps: int = DEFAULT_MAX_MPS):
        """Initialize the batch-size tuner.

        Args:
            safety_margin: Fraction of the max-viable batch size to reserve as
                headroom (e.g. 0.05 reserves 5%). Applied after binary search
                to avoid borderline OOMs during real workloads.
            max_cuda: Hard upper bound for CUDA batch size. Overridden at tune()
                time for GPUs with >= 80 GB memory (raised to DEFAULT_MAX_CUDA=2048).
            max_mps: Hard upper bound for MPS batch size.
        """
        self.safety_margin = safety_margin
        # max_cuda/max_mps are initial defaults; tune() may override for large GPUs.
        self._default_max_cuda = max_cuda
        self._default_max_mps = max_mps

    def tune(self, model: nn.Module, tokenizer, device: torch.device, backend: str, max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS) -> int:
        """Find the optimal batch size for the given model and backend.

        On CUDA, this performs a binary search between 1 and the per-GPU-memory
        cap, then optionally runs a TPS-aware sweep (gated by the
        BATCH_TUNE_TPS_SWEEP env var, default enabled) to select the
        throughput-maximizing batch size rather than the largest viable one.
        A safety margin is subtracted from the final result.

        On MPS, this uses a single-probe fast path (see _tune_mps_single_probe).

        Args:
            model: An nn.Module that supports model.generate(...). Its config
                attribute (if present) is read to compute a KV-cache performance
                estimate for logging.
            tokenizer: A tokenizer with encode(), pad_token_id, and eos_token_id.
            device: The torch.device to run on ("cuda" or "mps").
            backend: Backend identifier string ("cuda" or "mps").
            max_input_tokens: Maximum number of input tokens for the test
                sequences. Controls the test prompt length and feeds into the
                KV-cache estimate.

        Returns:
            int: The tuned batch size, guaranteed to be >= 1. A safety margin
            has already been subtracted from the max-viable size.

        Raises:
            RuntimeError: Re-raised if a non-OOM RuntimeError occurs during
                binary search (i.e. the error is not memory-related).

        Side effects:
            - Reads and temporarily sets the TRANSFORMERS_VERBOSITY env var.
            - Calls torch.cuda.empty_cache() or torch.mps.empty_cache() after
              each OOM to release fragmented memory.
            - Logs performance-model estimates and tuning progress at INFO level.
        """
        # Detect per-GPU memory to classify GPU tier.
        if backend == "cuda":
            try:
                # Use the FIRST GPU's memory for tier detection (not total across GPUs).
                # accelerate may split the model across GPUs, but KV-cache is per-GPU.
                single_gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                total_mem_gb = sum(
                    torch.cuda.get_device_properties(i).total_memory
                    for i in range(torch.cuda.device_count())
                ) / (1024**3)
            except (RuntimeError, torch.cuda.CudaError, AttributeError):
                single_gpu_mem_gb = 0
                total_mem_gb = 0

            # Safety: if GPU detection failed, use conservative defaults
            # rather than computing usable_mem = 0 which collapses cap to 1.
            if total_mem_gb <= 0:
                cap = DEFAULT_MAX_CUDA_SMALL
                mem_frac = MEMORY_USABLE_FRACTION_SMALL
                logger.warning(
                    "GPU memory detection failed — using conservative defaults: "
                    "cap=%d, mem_frac=%.0f%%",
                    cap, mem_frac * 100,
                )
            elif single_gpu_mem_gb >= LARGE_GPU_THRESHOLD_GB:
                # H200-class: 141 GB per GPU — use high cap, high memory fraction.
                cap = DEFAULT_MAX_CUDA  # 2048
                mem_frac = MEMORY_USABLE_FRACTION  # 0.75
                logger.info(
                    "Large GPU detected: %.0f GB per GPU (%.0f GB total) — "
                    "batch cap=%d, memory fraction=%.0f%%",
                    single_gpu_mem_gb, total_mem_gb, cap, mem_frac * 100,
                )
            else:
                cap = DEFAULT_MAX_CUDA_SMALL  # 128
                mem_frac = MEMORY_USABLE_FRACTION_SMALL  # 0.30
                logger.info(
                    "Standard GPU: %.0f GB per GPU — batch cap=%d, memory fraction=%.0f%%",
                    single_gpu_mem_gb, cap, mem_frac * 100,
                )

            # ── Performance-model initial guess ──
            # Skip when GPU detection failed (total_mem_gb=0 would collapse cap to 1).
            # NOTE: The performance model provides an informational estimate but does
            # NOT cap the binary search.  The model systematically overestimates
            # per‑sequence KV‑cache because: (a) KV_CACHE_MAX_SEQ_LEN=512 but the
            # actual test uses max_input_tokens=128 with TEST_GENERATION_TOKENS=10
            # → 138 max total tokens, and (b) most sequences hit EOS early.
            # Capping at the estimate prevented the OOM binary search from finding
            # the TRUE limit — on H200 with NLLB‑600M, est=559 but actual capacity
            # was 1600+ (only 29 GB of 210 GB used at bs=475).
            # The estimate is logged for diagnostics; the binary search (below)
            # tests up to |cap| and finds the real OOM boundary.
            if total_mem_gb > 0:
                # Try to read actual model config for accurate KV-cache estimate.
                try:
                    cfg = getattr(model, "config", None)
                    n_layers = (
                        getattr(cfg, "num_hidden_layers", None)
                        or getattr(cfg, "decoder_layers", None)
                        or KV_CACHE_NUM_LAYERS
                    )
                    n_kv_heads = (
                        getattr(cfg, "num_key_value_heads", None)
                        or getattr(cfg, "num_attention_heads", None)
                        or KV_CACHE_NUM_KV_HEADS
                    )
                    h_dim = (
                        getattr(cfg, "head_dim", None)
                        or getattr(cfg, "d_model", None)
                    )
                    if h_dim is None and hasattr(cfg, "hidden_size"):
                        n_attn = getattr(cfg, "num_attention_heads", None) or n_kv_heads
                        h_dim = cfg.hidden_size // max(n_attn, 1)
                    h_dim = h_dim or KV_CACHE_HEAD_DIM

                    # Use the actual test sequence length for the estimate, not 512.
                    # The tuner's _test_batch uses max_input_tokens + TEST_GENERATION_TOKENS.
                    test_seq_len = min(max_input_tokens + TEST_GENERATION_TOKENS, KV_CACHE_MAX_SEQ_LEN)
                    kv_per_seq = (n_layers * KV_CACHE_KV_FACTOR *
                                  n_kv_heads * h_dim *
                                  test_seq_len * KV_CACHE_BYTES_PER_ELEM)
                    usable_mem = total_mem_gb * (1024**3) * mem_frac
                    est = int(usable_mem / kv_per_seq) if kv_per_seq > 0 else cap
                    logger.info(
                        "Performance model (from config): n_layers=%d n_kv_heads=%d "
                        "head_dim=%d test_seq_len=%d → kv_per_seq=%.1fMB → est=%d "
                        "(informational — binary search uncapped)",
                        n_layers, n_kv_heads, h_dim, test_seq_len,
                        kv_per_seq / BYTES_PER_MB, est,
                    )
                except Exception:
                    # Fallback to hardcoded defaults.
                    test_seq_len = min(max_input_tokens + TEST_GENERATION_TOKENS, KV_CACHE_MAX_SEQ_LEN)
                    kv_per_seq = (KV_CACHE_NUM_LAYERS * KV_CACHE_KV_FACTOR *
                                  KV_CACHE_NUM_KV_HEADS * KV_CACHE_HEAD_DIM *
                                  test_seq_len * KV_CACHE_BYTES_PER_ELEM)
                    usable_mem = total_mem_gb * (1024**3) * mem_frac
                    est = int(usable_mem / kv_per_seq) if kv_per_seq > 0 else cap
                    logger.info(
                        "Performance model (defaults): kv_per_seq=%.1fMB, est=%d "
                        "(informational — binary search uncapped)",
                        kv_per_seq / BYTES_PER_MB, est,
                    )
        else:
            cap = self._default_max_mps

        logger.info("Tuning batch size for %s (cap=%d, max_input=%d)", backend, cap, max_input_tokens)

        # ── MPS fast path: probe the cap once, fall back to 1 on OOM ───
        # Binary search over 5-6 shapes creates 15-25 GB of MPSGraph
        # compilation caches in IOAccelerator memory that cannot be freed.
        # A single probe tests the cap; if it OOMs, halve once or fall to 1.
        if backend == "mps":
            return self._tune_mps_single_probe(
                model, tokenizer, device, cap, max_input_tokens,
            )

        low, high = MIN_BATCH_SIZE, cap

        while low <= high:
            mid = (low + high) // BINARY_SEARCH_DIVISOR
            try:
                self._test_batch(model, tokenizer, device, mid, max_input_tokens, backend)
                low = mid + 1
            except _MPS_OOM:
                logger.debug("MPS OOM at batch_size=%d", mid)
                high = mid - 1
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except torch.cuda.OutOfMemoryError:
                logger.debug("CUDA OOM at batch_size=%d", mid)
                high = mid - 1
                torch.cuda.empty_cache()
            except RuntimeError as e:
                # PyTorch 2.5+ reports MPS errors as RuntimeError strings.
                msg = str(e)
                if any(oom in msg for oom in _MPS_OOM_MESSAGES):
                    logger.debug("MPS OOM at batch_size=%d: %s", mid, msg)
                    high = mid - 1
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                else:
                    # Not an OOM — re-raise so the crash is visible.
                    raise
            except torch.OutOfMemoryError:
                logger.debug("OOM at batch_size=%d", mid)
                high = mid - 1
                if backend == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                msg = str(e).lower()
                if any(kw in msg for kw in ("out of memory", "mps", "memory", "allocat")):
                    logger.debug("OOM at batch_size=%d (%s: %s)", mid, type(e).__name__, msg[:100])
                    high = mid - 1
                    if backend == "cuda":
                        torch.cuda.empty_cache()
                    elif hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                else:
                    logger.warning(
                        "Unexpected error at batch_size=%d: %s — lowering cap",
                        mid, str(e)[:120],
                    )
                    high = mid - 1  # treat unexpected errors as OOM, lower cap

        safety_factor = 1.0 - self.safety_margin

        # ── TPS-aware sweep: find throughput-maximizing batch size ──
        # For memory-bandwidth-bound models on H200, larger batches can have
        # LOWER throughput due to HBM contention.  Measure actual TPS at
        # several batch sizes between high//2 and high and pick the best.
        # Gate with BATCH_TUNE_TPS_SWEEP=0 for fast restarts.
        if backend == "cuda" and high > MIN_BATCH_SIZE:
            tps_sweep_enabled = os.environ.get("BATCH_TUNE_TPS_SWEEP", "1") != "0"
            if tps_sweep_enabled:
                best_tps = 0.0
                best_bs = high
                # Test 4-5 points between half and max
                step = max(1, high // 5)
                candidates = sorted(set(
                    max(MIN_BATCH_SIZE, bs)
                    for bs in range(max(MIN_BATCH_SIZE, high // 2), high + 1, step)
                ))
                if len(candidates) < 3:
                    # Too few points for meaningful sweep — just test high
                    candidates = [high]
                for candidate_bs in candidates:
                    try:
                        import time
                        # 3-5 iterations to amortize startup overhead
                        n_iters = max(2, 8 // candidate_bs)
                        t0 = time.monotonic()
                        for _ in range(n_iters):
                            self._test_batch(model, tokenizer, device,
                                             candidate_bs, max_input_tokens, backend)
                        elapsed = time.monotonic() - t0
                        tokens_per_iter = candidate_bs * TEST_GENERATION_TOKENS
                        tps = (tokens_per_iter * n_iters) / elapsed if elapsed > 0 else 0
                        logger.info(
                            "  TPS sweep: bs=%d → %.0f tok/s (%d iters, %.2fs)",
                            candidate_bs, tps, n_iters, elapsed,
                        )
                        if tps > best_tps:
                            best_tps = tps
                            best_bs = candidate_bs
                    except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError):
                        logger.debug("TPS sweep: bs=%d OOM, skipping", candidate_bs)
                        if backend == "cuda":
                            torch.cuda.empty_cache()
                        continue
                logger.info(
                    "TPS-optimal batch: %d (%.0f tok/s, max viable: %d)",
                    best_bs, best_tps, high,
                )
                high = best_bs  # Use TPS-optimal, not max-viable

        optimal = max(int(high * safety_factor), MIN_BATCH_SIZE)
        logger.info("Tuned batch size: %d (max viable: %d)", optimal, high)
        return optimal

    def _tune_mps_single_probe(
        self, model, tokenizer, device, cap, max_input_tokens,
    ) -> int:
        """MPS single-probe: test the cap, halve on OOM, else use cap.

        Avoids binary search which creates 5-6 unique MPSGraph compilations
        (15-25 GB of IOAccelerator memory that cannot be freed).
        """
        import torch as _torch

        # Try the cap.
        bs = cap
        for attempt in range(2):  # cap, then cap/2
            try:
                self._test_batch(model, tokenizer, device, bs, max_input_tokens, "mps")
                # Success — apply safety margin.
                optimal = max(int(bs * (1.0 - self.safety_margin)), MIN_BATCH_SIZE)
                logger.info(
                    "MPS tuned batch size: %d (probed %d, safety %.0f%%)",
                    optimal, bs, self.safety_margin * 100,
                )
                return optimal
            except (RuntimeError, torch.OutOfMemoryError):
                if hasattr(_torch.mps, "empty_cache"):
                    _torch.mps.empty_cache()
                if bs <= 1:
                    break
                bs = max(bs // 2, 1)
                logger.info("MPS: batch_size=%d OOM — retrying with %d", bs * 2, bs)

        logger.warning("MPS: batch_size=1 also failed — using 1")
        return MIN_BATCH_SIZE

    def _test_batch(self, model, tokenizer, device, batch_size, max_input_tokens, backend):
        """Run a single test generation to verify a candidate batch size fits in memory.

        Constructs a padded input tensor of shape (batch_size, seq_len) using a
        repeated test sentence, then calls model.generate() with greedy decoding.
        An OOM exception propagates to the caller; success means the batch size
        is viable.

        Args:
            model: An nn.Module that supports model.generate(input_ids, ...).
            tokenizer: A tokenizer with encode(), pad_token_id, and eos_token_id.
            device: The torch.device to run on.
            batch_size: Number of sequences to generate in parallel.
            max_input_tokens: Maximum input sequence length (tokens).
            backend: Backend identifier string ("cuda" or "mps").

        Side effects:
            - Temporarily sets TRANSFORMERS_VERBOSITY="error" to suppress
              HuggingFace generation warnings; restored in a finally block.
            - On CUDA, calls torch.cuda.synchronize() after generation.
            - On MPS, calls torch.mps.empty_cache() after generation (if available).

        Raises:
            torch.cuda.OutOfMemoryError: If the batch size exceeds CUDA memory.
            RuntimeError: If the batch size triggers an MPS or generic OOM.
            torch.OutOfMemoryError: If a generic PyTorch OOM occurs.
        """
        txt = "This is a test sentence for batch size tuning. " * TEST_SENTENCE_MULTIPLIER
        tids = tokenizer.encode(txt, add_special_tokens=True)[:max_input_tokens]
        pad_id = tokenizer.pad_token_id or FALLBACK_PAD_TOKEN_ID

        pin = (backend == PIN_MEMORY_BACKEND)
        input_ids = torch.full((batch_size, len(tids)), pad_id, dtype=torch.long, pin_memory=pin)
        for i in range(batch_size):
            input_ids[i, :len(tids)] = torch.tensor(tids, dtype=torch.long)

        input_ids = input_ids.to(device)
        mask = (input_ids != pad_id).long().to(device)

        # Suppress HF generation warnings during batch tuning.
        import os
        prev_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["TRANSFORMERS_VERBOSITY"] = TRANSFORMERS_VERBOSITY_ERROR

        try:
            with torch.no_grad():
                model.generate(
                    input_ids=input_ids, attention_mask=mask,
                    max_new_tokens=TEST_GENERATION_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        finally:
            if prev_verbosity is not None:
                os.environ["TRANSFORMERS_VERBOSITY"] = prev_verbosity
            else:
                os.environ.pop("TRANSFORMERS_VERBOSITY", None)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
