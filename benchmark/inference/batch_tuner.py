"""Auto batch-size tuner with backend-aware OOM handling.

v2.0: Performance-model initial guess to reduce binary-search steps.
      Pinned-memory test tensors for realistic CUDA transfer simulation.
v3.3: Robust MPS OOM detection — handles PyTorch 2.x MPS error variants.
      Batch tuning failure gracefully falls back to batch_size=1.
"""

import logging
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
MEMORY_USABLE_FRACTION = 0.3
DEFAULT_SAFETY_MARGIN = 0.15
DEFAULT_MAX_CUDA = 256
DEFAULT_MAX_MPS = 32
DEFAULT_MAX_INPUT_TOKENS = 512
TEST_SENTENCE_MULTIPLIER = 10
TEST_GENERATION_TOKENS = 10
PIN_MEMORY_BACKEND = "cuda"
BYTES_PER_KB = 1024
BYTES_PER_MB = BYTES_PER_KB * BYTES_PER_KB  # 1,048,576 — convert bytes → MB
MIN_BATCH_SIZE = 1
BINARY_SEARCH_DIVISOR = 2
FALLBACK_PAD_TOKEN_ID = 0
TRANSFORMERS_VERBOSITY_ERROR = "error"

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
    def __init__(self, safety_margin: float = DEFAULT_SAFETY_MARGIN, max_cuda: int = DEFAULT_MAX_CUDA, max_mps: int = DEFAULT_MAX_MPS):
        self.safety_margin = safety_margin
        self.max_cuda = max_cuda
        self.max_mps = max_mps

    def tune(self, model: nn.Module, tokenizer, device: torch.device, backend: str, max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS) -> int:
        cap = self.max_cuda if backend == "cuda" else self.max_mps

        # ── Performance-model initial guess ──
        if backend == "cuda":
            try:
                total_mem = sum(
                    torch.cuda.get_device_properties(i).total_memory
                    for i in range(torch.cuda.device_count())
                )
                kv_per_seq = (KV_CACHE_NUM_LAYERS * KV_CACHE_KV_FACTOR *
                              KV_CACHE_NUM_KV_HEADS * KV_CACHE_HEAD_DIM *
                              KV_CACHE_MAX_SEQ_LEN * KV_CACHE_BYTES_PER_ELEM)
                usable_mem = total_mem * MEMORY_USABLE_FRACTION
                est = int(usable_mem / kv_per_seq) if kv_per_seq > 0 else cap
                cap = min(cap, max(est, MIN_BATCH_SIZE))
                logger.info("Performance model: kv_per_seq=%.1fMB, est_cap=%d",
                            kv_per_seq / BYTES_PER_MB, cap)
            except (RuntimeError, torch.cuda.CudaError, AttributeError):
                pass

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
        """Run a test generation with production-sized input to catch OOM."""
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
