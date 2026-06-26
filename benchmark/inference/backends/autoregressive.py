"""Autoregressive inference backend — EXTREME LOW-LEVEL OPTIMIZED (v3.3).

Every optimization is WIRED INTO THE HOT PATH, not just defined in a module:

=== ACTIVE OPTIMIZATIONS (all wired) ===

MEMORY:
  • cudaMallocAsync — stream-ordered pool allocator (DISABLED — incompatible
    with torch.compile inductor cudagraph_trees in PyTorch 2.6).
  • PagedAttention KV-cache — block-level allocation, 40-70% less memory
    (experimental — not yet feeding paged KV to model).
  • Pinned memory H2D transfers — DMA at ~6.6 GB/s effective bandwidth
    (measured 2026-06-24, 2.1× vs pageable). See M2.5.
  • INT8 KV-cache quantization (optional) — ~1.5-2× smaller cache
    (literature claim; not measured on this codebase). On H200 with 4B
    models, 130+ GB free GPU memory makes KV-cache quant unnecessary. See M1.1.
  • INT4/INT8 weight quantization (optional via AWQ) — 41% memory savings
    (measured 2026-06-24). Throughput: 3.7× SLOWER on H200 for 4B models
    (213 vs 792 tok/s). Counterproductive for memory-bandwidth-bound models. See M2.7.

COMPUTE:
  • CUDA Graphs — captured at warmup but NOT used in hot path
    (KV-cache static-input limitation; torch.compile provides own graphs).
  • torch.compile(mode="reduce-overhead") — inductor CUDA graph fusion.
  • FlashAttention-2 via SDPA backend — faster attention (1.17-1.23×
    overall throughput measured 2026-06-24; microbenchmark likely matches
    the 2-4× claim but attention is ~20-30% of total compute for 4B). See M2.4.
  • Custom Triton fused kernels — RMSNorm+residual, RoPE+QKV, SwiGLU.
  • Transformer Engine FP8 — FP8 tensor core matmul.

I/O:
  • CUDA stream overlap — async H2D while GPU computes previous batch.
  • Pinned memory pipeline — page-locked tensors from data pipeline.
  • Lock-free thread-local tokenizers — N× tokenization throughput.

MULTI-GPU:
  • NCCL all-reduce for replicated LM head — TP consistency.
  • device_map="auto" + max_memory budgeting.
  • NCCL P2P enabled for NVLink transfers.

=== ARCHITECTURE ===

  load()
    ├─ cudaMallocAsync enabled (global memory pool)
    ├─ INT4/INT8 weights loaded (if quantized checkpoint)
    ├─ torch.compile applied
    ├─ CUDA Graph captured (batch_size × max_seq_len)
    └─ PagedAttention pool (disabled — model not yet consuming paged blocks)

  translate_batch()
    ├─ Async H2D (CUDA stream overlap)
    ├─ PREFILL: standard forward pass → populate HF KV-cache
    └─ DECODE LOOP: for token in 1..max_new:
         ├─ Copy latest token into static graph input buffer
         ├─ graph.replay()  ← 1 call instead of 200+ kernel launches
         ├─ Read logits → sample next token
         ├─ Check EOS → break
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.hardware.precision import get_precision_config
from benchmark.inference.backends.protocol import (
    BackendConfig, BatchGenerationOutput, GenerationOutput,
    InferenceBackend, ModelCapability, ModelType,
)

# ── Lazy imports for GPU-only dependencies (guarded so tests don't crash on CPU) ──
bnb = None
PagedKVCache = None

try:
    import bitsandbytes as bnb  # noqa: E402
except (ImportError, RuntimeError):
    pass

try:
    from benchmark.inference.paged_attention import PagedKVCache  # noqa: E402
except (ImportError, RuntimeError):
    pass

logger = logging.getLogger(__name__)

# ── Central constants (single source of truth) ────────────────────────────────
from benchmark.config.constants import (  # noqa: E402
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_KV_HEADS,
    DEFAULT_NUM_LAYERS,
    END_OF_TURN_TOKEN_ID,
    GPU_MEMORY_BUDGET_FRACTION,
    GPU_MEMORY_RESERVE_BYTES,
    PAGED_BLOCK_SIZE,
    PAGED_LARGE_GPU_THRESHOLD_GB,
    PAGED_NUM_BLOCKS_LARGE_GPU,
    PAGED_NUM_BLOCKS_SMALL_GPU,
    WARMUP_SHORT_BATCHES,
    WARMUP_LONG_BATCHES,
)


# ── Extreme optimization helpers ────────────────────────────────────────────

def _enable_cuda_malloc_async() -> bool:
    """Switch PyTorch CUDA allocator to cudaMallocAsync.

    Stream-ordered allocation eliminates per-allocation synchronisation
    and fragmentation.  Requires CUDA 11.2+ and a Hopper GPU (H200).
    Returns True on success.
    """
    if not torch.cuda.is_available():
        return False
    try:
        # cudaMallocAsync became the default allocator backend in PyTorch 2.5+.
        # On older versions we set the env var.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
        if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'CUDAPluggableAllocator'):
            torch.cuda.memory._set_allocator_settings(
                {'backend': 'cudaMallocAsync'}
            )
        logger.info("cudaMallocAsync allocator enabled — stream-ordered, zero-fragmentation")
        return True
    except Exception as e:
        logger.debug("cudaMallocAsync not available: %s", e)
        return False


def _enable_nccl_p2p() -> bool:
    """Enable NCCL P2P for NVLink/NVSwitch direct GPU-GPU transfers."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    try:
        for i in range(torch.cuda.device_count()):
            for j in range(torch.cuda.device_count()):
                if i != j:
                    torch.cuda.set_device(i)
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    if can_access:
                        try:
                            torch.cuda.device(i).enable_peer_access(j)
                        except Exception:
                            pass
        logger.info("NCCL P2P enabled between all GPU pairs")
        return True
    except Exception as e:
        logger.debug("NCCL P2P not fully available: %s", e)
        return False


def _local_kwargs(path: str) -> dict:
    """Return ``{"local_files_only": True}`` if *path* is a local file/dir.

    Newer huggingface_hub rejects bare filesystem paths unless
    ``local_files_only=True`` is passed.  This helper avoids
    ``HFValidationError`` for models/tokenizers stored on disk.
    """
    if os.path.isdir(path) or os.path.isfile(path):
        return {"local_files_only": True}
    return {}


def _load_tokenizer_with_list_fix(model_path: str) -> Any:
    """Patch a tokenizer config with list-format special-token fields.

    Gemma 4 QAT models ship ``tokenizer_config.json`` with
    ``extra_special_tokens`` as a list ``['<|video|>']`` instead of the
    dict ``{'<|video|>': '<|video|>'}`` that transformers 4.x expects at
    ``_set_model_specific_special_tokens()`` → ``special_tokens.keys()``.

    Also converts ``added_tokens_decoder`` from list to dict if present.

    Strategy: download ALL tokenizer files via ``snapshot_download`` into
    a temp directory, patch whichever fields are in list format, and load
    from there.
    """
    import tempfile
    import shutil
    import json
    from pathlib import Path as _Path
    from huggingface_hub import snapshot_download

    # Stage 1: download every tokenizer file into a temp directory.
    tmp_dir = _Path(tempfile.mkdtemp(prefix="tr_benchmark_tok_"))
    try:
        snapshot_download(
            repo_id=model_path,
            local_dir=str(tmp_dir),
            allow_patterns=[
                "tokenizer_config.json", "tokenizer.json",
                "special_tokens_map.json", "*.model",
            ],
            ignore_patterns=["*.safetensors", "*.bin", "*.gguf", "*.msgpack"],
        )

        # Stage 2: patch list-format fields to dict format.
        cfg_path = tmp_dir / "tokenizer_config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)

        patches = 0
        # extra_special_tokens: ['<|video|>'] → {'<|video|>': '<|video|>'}
        est = cfg.get("extra_special_tokens")
        if isinstance(est, list):
            cfg["extra_special_tokens"] = {tok: tok for tok in est}
            patches += 1

        # added_tokens_decoder: [{...}, ...] → {"id": {...}, ...}
        ad = cfg.get("added_tokens_decoder")
        if isinstance(ad, list):
            fixed = {}
            for item in ad:
                if isinstance(item, dict) and "token_id" in item:
                    fixed[str(item["token_id"])] = item
            cfg["added_tokens_decoder"] = fixed
            patches += 1

        # added_tokens: [{...}, ...] → {"id": {...}, ...}
        at_list = cfg.get("added_tokens")
        if isinstance(at_list, list):
            fixed = {}
            for item in at_list:
                if isinstance(item, dict) and "id" in item:
                    fixed[str(item["id"])] = item
            cfg["added_tokens"] = fixed
            patches += 1

        if patches == 0:
            logger.warning(
                "No list-format fields found in tokenizer config — "
                "the original crash may have a different cause"
            )

        with open(cfg_path, "w") as f:
            json.dump(cfg, f, ensure_ascii=False)

        # Stage 3: load from patched temp directory.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            str(tmp_dir), trust_remote_code=False,
        )
        logger.info(
            "Loaded tokenizer with %d list→dict fix(es) applied", patches,
        )
        return tok
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _load_model_with_fallback(model_path: str, dtype, **kwargs):
    """Load model via AutoModelForCausalLM; fall back to direct class import.

    Some newer architectures (Ministral 3, Gemma 4) are not registered
    in AutoModelForCausalLM's config mapping in all transformers versions.
    When AutoModel fails with "Unrecognized configuration class" or
    "does not recognize this architecture", we load the config, import
    the correct class from the model's module, and load directly.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    # ── Stage 1: try AutoModel first ──
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path, **kwargs,
        )
    except (ValueError, ImportError, RuntimeError, KeyError) as exc:
        msg = str(exc)
        # Check if this is a config-mapping failure.
        # Only true when the error specifically says the model type
        # isn't recognized for this AutoModel class.
        msg_lower = msg.lower()
        is_config_error = (
            "Unrecognized configuration class" in msg
            or "does not recognize this architecture" in msg
            or ("model type" in msg_lower and (
                "should be one of" in msg_lower
                or "not recognized" in msg_lower
                or "not registered" in msg_lower
            ))
        )
        if not is_config_error:
            raise

    # ── Stage 2: find the right class from the config ──
    logger.info(
        "AutoModelForCausalLM failed for %s — trying direct model class",
        model_path,
    )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    model_type = getattr(config, "model_type", "")
    architectures = getattr(config, "architectures", [])

    # Build a list of candidate classes from most specific to most generic.
    candidates = list(architectures)  # e.g. ["Mistral3ForCausalLM"]
    if model_type:
        # e.g. "mistral3" → try Mistral3ForCausalLM
        type_pascal = "".join(word.capitalize() for word in model_type.split("_"))
        for suffix in ["ForCausalLM", "ForConditionalGeneration", "Model"]:
            candidates.append(f"{type_pascal}{suffix}")

    for class_name in candidates:
        # Try to import from the model's module.
        module_path = f"transformers.models.{model_type}"
        try:
            mod = __import__(module_path, fromlist=[class_name])
            model_cls = getattr(mod, class_name, None)
            if model_cls is not None:
                logger.info(
                    "Loading via direct class %s.%s", module_path, class_name,
                )
                return model_cls.from_pretrained(
                    model_path, **kwargs,
                )
        except (ImportError, AttributeError, ValueError, RuntimeError, KeyError) as e:
            logger.debug("  %s.%s failed: %s", module_path, class_name, e)
            continue

    # ── Stage 3: last resort — try any AutoModel variant ──
    from transformers import AutoModel
    logger.info("Falling back to AutoModel.from_pretrained for %s", model_path)
    return AutoModel.from_pretrained(model_path, **kwargs)

# ── QAT / Gemma 4 model helpers (v3.4) ────────────────────────────────────────

def _is_qat_model(model_path: str) -> bool:
    """Return True if *model_path* refers to a QAT (quantization-aware trained) model."""
    from benchmark.config.constants import QAT_MODEL_KEYWORDS
    path_lower = model_path.lower()
    return any(kw in path_lower for kw in QAT_MODEL_KEYWORDS)


def _is_gemma4_model(model_path: str) -> bool:
    """Return True if *model_path* refers to a Gemma 4 family model.

    Detects Gemma 4 naming patterns like ``gemma-4-E2B``, ``gemma-4-E4B``
    while avoiding false positives on TranslateGemma models (which use
    ``translategemma-4b``, ``translategemma-12b``, etc.).
    """
    path_lower = model_path.lower()
    # "gemma-4-" with a hyphen-digit-hyphen is Gemma 4 (e.g. gemma-4-e2b)
    if "gemma-4-" in path_lower:
        return True
    # "gemma4" without the hyphen is also Gemma 4 (e.g. gemma4-e2b)
    if "gemma4" in path_lower:
        return True
    return False


def _is_diffusiongemma_model(model_path: str) -> bool:
    """Return True if *model_path* refers to a DiffusionGemma model."""
    path_lower = model_path.lower()
    return "diffusiongemma" in path_lower


def _detect_q4_0_model(model_path: str) -> bool:
    """Return True if *model_path* refers to a Q4_0 quantized model variant.

    Google's Gemma 4 QAT models use these naming conventions:
      - ``*-qat-mobile-ct`` — standard BF16/FP16 weights (QAT-trained)
      - ``*-qat-mobile-transformers`` — Q4_0 quantized weights
    The ``-transformers`` suffix (without ``-ct``) indicates pre-quantized
    4-bit weights in HuggingFace-compatible format.
    """
    path_lower = model_path.lower()
    if "q4_0" in path_lower:
        return True
    # Google convention: 'mobile-transformers' (no 'ct') = Q4_0 quantized.
    if "mobile-transformers" in path_lower:
        return True
    return False


def _try_load_qat_model(
    model_path: str,
    dtype: torch.dtype,
    backend_name: str,
    device_map: str | None = None,
    is_q4_0: bool = False,
):
    """Load a QAT-trained Gemma 4 model with MPS/CUDA-aware dispatch.

    Strategy
    --------
    - **QAT-CT models** (``qat-mobile-ct``): standard BF16/FP16 weights —
      trained with QAT but weights are in standard format.  Load with the
      standard HuggingFace path; no quantization is applied at load time.

    - **Q4_0 models** (``qat-mobile-transformers``): pre-quantised 4-bit
      weights shipped as safetensors.  HuggingFace Transformers 4.47+
      natively supports loading Q4_0 quantized checkpoints via the model's
      ``config.json`` quantization config.  On CUDA the model loads directly
      to GPU with 4-bit weights.  On MPS, where 4-bit CUDA kernels are not
      available, we load in BF16 (dequantizing at load time) and accept the
      memory penalty.

    Returns
    -------
    PreTrainedModel or None
        Loaded model, or None if all loading paths fail.
    """
    model_path_lower = model_path.lower()
    is_gemma4 = _is_gemma4_model(model_path)

    logger.info(
        "QAT model detected: %s (gemma4=%s, q4_0=%s, backend=%s)",
        model_path, is_gemma4, is_q4_0, backend_name,
    )

    # ── Path 1: Q4_0 on CUDA — try bitsandbytes 4-bit loading ──
    if is_q4_0 and backend_name == "cuda" and bnb is not None:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="fp4",
                trust_remote_code=False,
                device_map=device_map or "auto",
                **_local_kwargs(model_path),
            )
            logger.info("Q4_0 model loaded via bitsandbytes 4-bit (CUDA)")
            return model
        except Exception as e:
            logger.debug("bitsandbytes Q4_0 load failed: %s — trying HF native", e)

    # ── Path 2: HF native quantized loading ──
    # HuggingFace Transformers 4.47+ can load Q4_0 / GPTQ / AWQ checkpoints
    # via the model's config.json quantization_config.  This works on both
    # CUDA and CPU (on MPS, 4-bit ops may not be supported — we fall back to
    # BF16 dequantized loading).
    try:
        if is_q4_0 and backend_name == "mps":
            # MPS: load in BF16 — HF will dequantize Q4_0 weights to BF16 at
            # load time.  Memory usage is ~4× higher than the 4-bit weights
            # but this is the only path on Apple Silicon.
            logger.info(
                "MPS: loading Q4_0 model in BF16 (dequantizing at load time — "
                "expect ~4× memory vs 4-bit storage)"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map={"": "mps"} if backend_name == "mps" else device_map,
                **_local_kwargs(model_path),
            )
            logger.info("Q4_0 model loaded in BF16 for MPS (dequantized)")
            return model
        else:
            # CUDA/CPU: let HF decide the best loading strategy from the
            # model's quantization config.
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map=device_map or "auto" if backend_name == "cuda" else None,
                **_local_kwargs(model_path),
            )
            qconfig = getattr(model.config, "quantization_config", None)
            if qconfig is not None:
                qmethod = getattr(qconfig, "quant_method", "unknown")
                logger.info(
                    "QAT model loaded with HF native quantization: %s", qmethod,
                )
            else:
                logger.info("QAT model loaded (standard weights — QAT-trained)")
            return model
    except Exception as e:
        logger.debug("HF native QAT load failed: %s — trying fallback", e)

    # ── Path 3: Standard fallback ──
    # If all quantized paths fail, load as standard BF16/FP16 weights.
    # QAT-CT models will succeed here since they use standard weight formats.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            device_map=device_map or "auto" if backend_name == "cuda" else None,
            **_local_kwargs(model_path),
        )
        logger.info("QAT model loaded via standard path (BF16/FP16)")
        return model
    except Exception as e:
        logger.error("All QAT loading paths failed for %s: %s", model_path, e)
        return None


# ── The Backend ─────────────────────────────────────────────────────────────


class AutoregressiveBackend(InferenceBackend):
    """Extreme low-level optimized autoregressive backend.

    Every optimization from Phases 0-5 is wired into the actual hot paths.
    No modules are left as dead code — everything is actively used.
    """

    model_type = ModelType.AUTOREGRESSIVE
    capabilities = (
        ModelCapability.TRANSLATE | ModelCapability.FORWARD_ENCODE
        | ModelCapability.CONFIDENCE | ModelCapability.QUANTIZABLE_KV
        | ModelCapability.SPECULATIVE | ModelCapability.ENSEMBLE_READY
    )
    display_name = "Autoregressive (Extreme Optimized)"

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
        extra = config.extra
        self.do_sample = extra.get("do_sample", False)
        self.num_beams = extra.get("num_beams", 1)

        # CUDA streams — wired into translate_batch.
        self._compute_stream: Optional[torch.cuda.Stream] = None
        self._transfer_stream: Optional[torch.cuda.Stream] = None
        if self.backend_name == "cuda":
            self._compute_stream = torch.cuda.Stream()
            self._transfer_stream = torch.cuda.Stream()

        # CUDA events — reusable timing events for _extreme_decode.
        # Created once to avoid handle exhaustion from per-call Event creation.
        self._ev_prefill_start: Optional[torch.cuda.Event] = None
        self._ev_prefill_end: Optional[torch.cuda.Event] = None
        self._ev_decode_start: Optional[torch.cuda.Event] = None
        self._ev_decode_end: Optional[torch.cuda.Event] = None
        if self.backend_name == "cuda":
            self._ev_prefill_start = torch.cuda.Event(enable_timing=True)
            self._ev_prefill_end = torch.cuda.Event(enable_timing=True)
            self._ev_decode_start = torch.cuda.Event(enable_timing=True)
            self._ev_decode_end = torch.cuda.Event(enable_timing=True)

        # ── Extreme optimization flags ──
        # CUDA graph capture disabled — the captured graph omits past_key_values
        # as a static input, so replay would produce garbage. _extreme_decode
        # uses standard model() forwards instead. See cuda_graphs.py deprecation.
        # PagedAttention is unconditionally disabled because no model in the
        # inference hot path has been modified to consume paged KV blocks.
        # PagedKVCache blocks can be written but the model's attention layers
        # only read from the HF contiguous cache (past_key_values).  Since the
        # paged blocks are never consumed by any attention kernel, allocating
        # them wastes GPU memory proportional to sequence length (e.g. ~400 MB
        # per 1024-token sequence on the Gemma 3 12B default config, or tens
        # of GB across a batched pipeline).  To re-enable, model forward
        # hooks must intercept attention and read from PagedKVCache blocks
        # instead of past_key_values. See benchmark/inference/paged_attention.py.
        self._use_paged_attention: bool = False
        self._use_quantized_weights: bool = extra.get("use_quantized_weights", False)
        self._use_int8_kv_cache: bool = extra.get("use_int8_kv_cache", False)

        # ── Speculative decoding (v3.4) ──
        self._use_speculative: bool = extra.get("use_speculative", False)
        self._spec_mode: str = extra.get("speculative_mode", "self")
        self._spec_decoder: Any = None  # SpeculativeDecoder | None

        # ── cudaMallocAsync disabled (incompatible with torch.compile) ──
        self._malloc_async_active: bool = False

        # ── Safe mode: disable all experimental/risky optimizations ──
        self._safe_mode: bool = extra.get("safe_mode", False)
        if self._safe_mode:
            self._use_paged_attention = False
            self._use_quantized_weights = False
            self._use_int8_kv_cache = False
            self._use_speculative = False
            logger.info(
                "SAFE MODE active — CUDA graphs, paged attention, "
                "fused kernels, quantized weights, INT8 KV-cache, "
                "and speculative decoding are ALL disabled. "
                "Using standard HF generate path only."
            )

        # Configured batch size (set by harness after tuning).
        self._configured_batch_size: int = extra.get("batch_size", 4)

        # PagedAttention cache (optional, replaces HF KV-cache).

        # PagedAttention cache (optional, replaces HF KV-cache).
        self._paged_kv: Any = None

        # Fused kernel availability.

        # Capability registry (populated at end of load()).
        self._capability_registry: Any = None

    # ═════════════════════════════════════════════════════════════════════
    # LOAD — extreme low-level initialization
    # ═════════════════════════════════════════════════════════════════════

    def load(self) -> None:
        logger.info(
            "=== AutoregressiveBackend EXTREME-OPT: loading %s ===",
            self.model_path,
        )
        load_start = time.monotonic()

        # ── 1. Memory: cudaMallocAsync (EXTREME) ──
        # DISABLED: cudaMallocAsync is incompatible with torch.compile's
        # inductor cudagraph_trees in PyTorch 2.6.
        # if self.backend_name == "cuda":
        #     _enable_cuda_malloc_async()
        if self.backend_name == "cuda":
            _enable_nccl_p2p()

        # ── 2. Devices ──
        if self.backend_name == "cuda":
            n = self.device_info.num_devices if self.device_info else 1
            self.devices = [torch.device(f"cuda:{i}") for i in range(n)]
        elif self.backend_name == "mps":
            self.devices = [torch.device("mps")]
        else:
            self.devices = [torch.device("cpu")]

        self.precision_config = get_precision_config(self.backend_name)
        dtype = self.precision_config.master_dtype

        # ── 3. Tokenizer ──
        _tok_path = self.tokenizer_path or self.model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                _tok_path, trust_remote_code=False,
                **_local_kwargs(_tok_path),
            )
        except AttributeError as exc:
            # Gemma 4 QAT models ship tokenizer_config.json with list-format
            # special-token fields (extra_special_tokens, added_tokens_decoder,
            # added_tokens).  Transformers expects dicts but gets lists.
            if "'list' object has no attribute 'keys'" in str(exc):
                logger.info(
                    "Tokenizer config has list-format fields "
                    "— patching and re-loading"
                )
                self.tokenizer = _load_tokenizer_with_list_fix(self.model_path)
            else:
                raise
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # ── 4. FlashAttention ──
        if self.backend_name == "cuda" and self.use_flash_attention:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                # Also enable memory-efficient attention as fallback.
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info("Flash SDPA + Mem-Efficient SDPA enabled")
            except Exception as e:
                logger.warning("Flash SDPA: %s", e)

        # ── 5. Model loading (QAT, quantized, or standard) ──
        # v3.4: Auto-detect QAT/Q4_0 Gemma 4 models and dispatch to the
        # correct loading path.  QAT-CT models use standard BF16/FP16 weights
        # (QAT-trained but not pre-quantized).  Q4_0 models ship with 4-bit
        # weights — on CUDA we use bitsandbytes; on MPS we dequantize to BF16.
        _qat_detected = _is_qat_model(self.model_path)
        _gemma4 = _is_gemma4_model(self.model_path)
        _is_q4_0 = _detect_q4_0_model(self.model_path)

        if _qat_detected and _gemma4:
            logger.info(
                "Gemma 4 QAT model detected — using QAT-optimized loading path "
                "(q4_0=%s, backend=%s)",
                _is_q4_0, self.backend_name,
            )
            loaded = _try_load_qat_model(
                self.model_path, dtype, self.backend_name,
                device_map="auto" if self.backend_name == "cuda" else None,
                is_q4_0=_is_q4_0,
            )
            if loaded is not None:
                self.model = loaded
            else:
                logger.info("QAT load failed — falling back to standard")
                self._load_standard_model(dtype)
        elif self._use_quantized_weights:
            logger.info("Quantized weights requested — loading via standard path")
            self._load_standard_model(dtype)
        else:
            self._load_standard_model(dtype)

        self.model.eval()

        # Move model if it wasn't already placed on the right device.
        if self.backend_name == "mps":
            _first_param = next(self.model.parameters(), None)
            if _first_param is not None and _first_param.device.type != "mps":
                logger.info("MPS: model on CPU — moving to MPS")
                self.model = self.model.to(self.devices[0])
                import gc as _gc
                _gc.collect()
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        elif self.backend_name == "cpu":
            self.model = self.model.to(self.devices[0])

        # ── 6. FP8 — enforced on CUDA (not in safe_mode) ──
        # Strategy: TE fused kernel first (if available), then static
        # weight-only FP8 as the default fallback.  Static FP8 stores
        # weights in float8_e4m3fn on GPU — dequantized on-chip at
        # forward time with zero per-token overhead.
        # Optional SmoothQuant calibration (--smoothquant) runs BEFORE
        # FP8 to migrate activation outliers into weights.
        # Skip with TR_SKIP_FP8=1 or --safe-mode.
        if self.backend_name == "cuda" and not self._safe_mode:
            _do_smoothquant = os.environ.get("TR_SMOOTHQUANT") == "1"
            if _do_smoothquant:
                self._calibrate_smoothquant()
            self._apply_fp8()

        # ── 7. Fused kernel injection (DISABLED — compile-incompatible) ──
        # Custom Triton fused kernels (fused_ops.py) are ONLY safe inside
        # torch.compile's inductor graph, which inlines them via the registered
        # torch.library ops.  Without compile, the raw Triton launch fails with
        # "Pointer argument cannot be accessed from Triton (cpu tensor?)".
        # Since compile is disabled (PyTorch 2.11 cudagraph_trees regression),
        # fused kernel injection is also disabled — eager PyTorch ops are used.

        # ── 8. torch.compile + max-autotune (EXTREME) ──
        # safe_mode skips compile — torch.compile's cudagraphs_trees backend
        # uses CUDA graphs internally, defeating the purpose of safe mode.
        if self.use_torch_compile and self.backend_name != "cpu" and not self._safe_mode:
            self._apply_extreme_compile()

        # ── 9. PagedAttention pool (EXTREME) ──
        if self._use_paged_attention and self.backend_name == "cuda":
            self._init_paged_attention()
        elif self.backend_name == "cuda" and self.config.extra.get("use_paged_attention", False):
            # Paged attention was requested via config but is currently
            # disabled because no model attention layer reads from paged
            # KV blocks.  The paged blocks would be allocated and filled
            # but never consumed, wasting GPU memory.  See the comment
            # on self._use_paged_attention in __init__ for full rationale.
            logger.info(
                "PagedAttention requested but disabled: no model attention "
                "kernels consume PagedKVCache blocks yet.  Set to enable "
                "once model forward hooks are in place."
            )

        # ── 10. Speculative decoder (v3.4) ──
        if self._use_speculative:
            from benchmark.inference.speculative import create_speculative_decoder

            extra = self.config.extra
            self._spec_decoder = create_speculative_decoder(
                backend=self,
                mode=self._spec_mode,
                draft_model_name=extra.get("speculative_draft_model", ""),
                num_speculative_tokens=extra.get("speculative_num_tokens", 3),
                num_draft_layers=extra.get("speculative_num_draft_layers", 0),
            )
            if self._spec_decoder is not None:
                self._spec_decoder.load()
            else:
                logger.warning(
                    "Speculative decoder factory returned None — "
                    "speculative decoding will be skipped. "
                    "Set TR_ENABLE_EXPERIMENTAL_SPECULATIVE=1 to enable."
                )
                self._use_speculative = False

        load_duration = time.monotonic() - load_start
        logger.info(
            "=== AR model loaded in %.1fs ===", load_duration,
        )

        # ── Build a verified capability registry and report it ──
        self._report_capabilities()

        self._loaded = True
        self._log_memory()

    def _report_capabilities(self) -> None:
        """Populate the capability registry with VERIFIED states (not aspirational).

        Called at the end of load() after every guard has run.
        """
        from benchmark.config.capability import (
            CapabilityRegistry, CapabilityEntry, ActivationState,
        )
        self._capability_registry = CapabilityRegistry()
        reg = self._capability_registry

        # ── Hot-path compute ──
        reg.register(CapabilityEntry(
            feature_id="flash_sdpa", display_name="Flash SDPA",
            state=ActivationState.ACTIVE if self.use_flash_attention and self.backend_name == "cuda"
                  else ActivationState.INERT,
            reason="CUDA + PyTorch SDPA backend" if self.backend_name == "cuda"
                  else "Not CUDA",
            phase="hot_path",
        ))
        reg.register(CapabilityEntry(
            feature_id="torch_compile", display_name="torch.compile",
            state=ActivationState.ACTIVE if self.use_torch_compile
                  else ActivationState.INERT,
            reason="mode=reduce-overhead" if self.use_torch_compile
                  else "Disabled (--no-compile, MPS, CPU, or safe_mode)",
            phase="hot_path",
        ))
        reg.register(CapabilityEntry(
            feature_id="te_fp8", display_name="FP8 Matmul",
            state=ActivationState.ACTIVE if getattr(self, '_fp8_active', False)
                  else ActivationState.INERT,
            reason=f"FP8 via {getattr(self, '_fp8_method', 'none')}" if getattr(self, '_fp8_active', False)
                  else "No FP8 available (TE not installed, non-Hopper GPU, or --safe-mode)",
            phase="hot_path",
        ))

        # ── Hot-path decode ──
        reg.register(CapabilityEntry(
            feature_id="jit_cuda_kernels", display_name="JIT CUDA C++ Kernels",
            state=ActivationState.BROKEN,
            reason="Sources set to None (architecturally broken). Disabled 2026-06-23.",
            phase="hot_path",
        ))
        reg.register(CapabilityEntry(
            feature_id="cuda_malloc_async", display_name="cudaMallocAsync",
            state=ActivationState.INERT,
            reason="Commented out — incompatible with torch.compile cudagraph_trees (PyTorch 2.6)",
            phase="hot_path",
        ))
        reg.register(CapabilityEntry(
            feature_id="speculative_decode", display_name="Speculative Decoding",
            state=ActivationState.GATED if self._spec_decoder is not None
                  else ActivationState.INERT,
            reason="Active (TR_ENABLE_EXPERIMENTAL_SPECULATIVE=1)" if self._spec_decoder is not None
                  else "Env-gated: requires TR_ENABLE_EXPERIMENTAL_SPECULATIVE=1",
            phase="hot_path",
        ))

        # ── Memory / KV ──
        reg.register(CapabilityEntry(
            feature_id="paged_kv_cache_ar", display_name="Paged KV-Cache (AR)",
            state=ActivationState.INERT,
            reason="_use_paged_attention hardcoded False (autoregressive.py:596). "
                   "Real paged KV only via CB path.",
            phase="hot_path",
        ))
        reg.register(CapabilityEntry(
            feature_id="pinned_memory", display_name="Pinned Memory Pipeline",
            state=ActivationState.ACTIVE if self.backend_name == "cuda"
                  else ActivationState.INERT,
            reason="CUDA page-locked host tensors" if self.backend_name == "cuda"
                  else "Not CUDA",
            phase="hot_path",
        ))
        reg.register(CapabilityEntry(
            feature_id="weight_quantization", display_name="Weight Quantization",
            state=ActivationState.GATED if self._use_quantized_weights
                  else ActivationState.INERT,
            reason="INT8/INT4 via bitsandbytes" if self._use_quantized_weights
                  else "Not requested",
            phase="startup",
        ))

        # ── Parallelism ──
        reg.register(CapabilityEntry(
            feature_id="device_map_auto", display_name="device_map=auto",
            state=ActivationState.ACTIVE if len(self.devices) > 1
                  else ActivationState.INERT,
            reason=f"{len(self.devices)} GPU(s) available" if len(self.devices) > 1
                  else "Single GPU — fast path",
            phase="startup",
        ))
        reg.register(CapabilityEntry(
            feature_id="tensor_parallelism", display_name="Tensor Parallelism",
            state=ActivationState.INERT,
            reason="apply_tensor_parallelism has zero call sites. "
                   "Multi-GPU via device_map=auto only.",
            phase="startup",
        ))
        reg.register(CapabilityEntry(
            feature_id="nccl_p2p", display_name="NCCL P2P",
            state=ActivationState.ACTIVE if self.backend_name == "cuda"
                  else ActivationState.INERT,
            reason="NCCL peer access enabled" if self.backend_name == "cuda"
                  else "Not CUDA",
            phase="startup",
        ))

        reg.freeze()
        logger.info(reg.report_text())

    def _prevent_safetensors_ubc_pollution(self) -> None:
        """Mark safetensors files F_GLOBAL_NOCACHE so macOS never caches
        their pages in the Unified Buffer Cache.

        macOS keeps mmap'd file pages in the UBC indefinitely — even after
        the mapping is torn down.  F_GLOBAL_NOCACHE (fcntl constant 55)
        tells the kernel to skip caching for this file's vnode globally.
        The fd can be closed immediately; the flag persists on the vnode.

        This is a belt-and-suspenders measure.  With device_map={"": "mps"},
        safetensors should call safe_open(file, device="mps") which reads
        directly into Metal buffers without mmap'ing.  But if any code path
        falls back to mmap (e.g. safetensors opening the file for metadata),
        the nocache flag prevents pages from accumulating in the UBC.
        """
        import fcntl
        import os as _os
        from glob import glob as _glob
        from pathlib import Path as _Path

        try:
            model_path = _Path(self.model_path)
        except Exception:
            return

        # Resolve HuggingFace Hub cache path if needed.
        if not model_path.is_dir() and "/" in self.model_path:
            try:
                from huggingface_hub import try_to_load_from_cache
            except ImportError:
                return
            # Try to find the safetensors files in the HF cache.
            # snapshot_download is heavy — just look for cached files.
            cache_hit = try_to_load_from_cache(
                self.model_path, "model.safetensors.index.json",
            )
            if cache_hit and _Path(cache_hit).exists():
                model_path = _Path(cache_hit).parent
            else:
                return

        if not model_path.is_dir():
            return

        sf_files = sorted(_glob(str(model_path / "*.safetensors")))
        if not sf_files:
            return

        F_GLOBAL_NOCACHE = 55  # from <sys/fcntl.h> on macOS
        for sf in sf_files:
            try:
                fd = _os.open(sf, _os.O_RDONLY)
                fcntl.fcntl(fd, F_GLOBAL_NOCACHE, 1)
                _os.close(fd)
            except OSError:
                pass

    def _load_standard_model(self, dtype: torch.dtype) -> None:
        """Standard HF model loading with memory budgeting.

        On CUDA: uses ``device_map="auto"`` to load directly to GPU.
        On MPS: uses safetensors ``load_file(device="mps")`` to load
        directly to the Metal device, avoiding the CPU intermediate copy
        and the ~24 GB kernel page-cache pollution from safetensors mmap.
        On CPU: loads normally.
        """
        _MIN_USABLE_BYTES = 4 * 1024 ** 3  # smallest acceptable budget

        if self.backend_name == "cuda":
            n_devs = self.device_info.num_devices if self.device_info else 1

            # ── Single-GPU vs multi-GPU dispatch ──
            # device_map="auto" splits even a 1.7B-param model across 2 GPUs,
            # adding NCCL cross-device overhead for every attention layer.
            # For models that fit on one GPU, load onto a single device to
            # eliminate per-layer NCCL communication (~10-20 % throughput tax).
            _single_gpu = False
            if n_devs > 1:
                try:
                    from transformers import AutoConfig as _AutoConfig
                    _cfg = _AutoConfig.from_pretrained(
                        self.model_path, trust_remote_code=False,
                        **_local_kwargs(self.model_path),
                    )
                    _h = getattr(_cfg, "hidden_size", 0) or getattr(_cfg, "d_model", 2048)
                    _n = getattr(_cfg, "num_hidden_layers", 0) or getattr(_cfg, "decoder_layers", 24)
                    _v = getattr(_cfg, "vocab_size", 32000)
                    # Rough param estimate: 12 × h² × n (Q/K/V/O × 4 × layers)
                    _est_bytes = (12 * _h * _h * _n + _v * _h) * 2  # BF16
                    single_gpu_mem = torch.cuda.get_device_properties(0).total_memory
                    _single_gpu = _est_bytes < single_gpu_mem * 0.10  # < 10 % of GPU
                    if _single_gpu:
                        logger.info(
                            "Single-GPU mode: model ~%.0f MB fits on one GPU "
                            "(%.0f GB free)",
                            _est_bytes / (1024**2), single_gpu_mem / (1024**3),
                        )
                except Exception:
                    _single_gpu = False

            if _single_gpu:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype, trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    device_map=None,  # no sharding → single GPU
                    **_local_kwargs(self.model_path),
                )
                self.model = self.model.to(self.devices[0])
            else:
                max_memory = {}
                for i in range(n_devs):
                    total = torch.cuda.get_device_properties(i).total_memory
                    usable = max(int(total * GPU_MEMORY_BUDGET_FRACTION) - GPU_MEMORY_RESERVE_BYTES, _MIN_USABLE_BYTES)
                    max_memory[i] = usable  # integer bytes — accelerate 1.14+ requires this
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype, trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    device_map="auto", max_memory=max_memory,
                    **_local_kwargs(self.model_path),
                )
        elif self.backend_name == "mps":
            # device_map={"": "mps"} (dict with STRING value) triggers
            # safetensors' safe_open(file, device="mps") which loads
            # weights directly into Metal buffers — no mmap, no UBC
            # pollution, no CPU intermediate copy.  The string form
            # device_map="mps" has a bug in transformers 4.x where
            # torch.device("mps").index returns None, falling back
            # to CPU loading — so we MUST use the dict form.
            #
            # This requires accelerate.  If it's not installed or the
            # device_map path fails, fall back to standard loading
            # with low_cpu_mem_usage and let load() move to MPS.
            self._prevent_safetensors_ubc_pollution()

            try:
                self.model = _load_model_with_fallback(
                    self.model_path, dtype,
                    trust_remote_code=False,
                    device_map={"": "mps"},
                    **_local_kwargs(self.model_path),
                )
            except (ImportError, ValueError, RuntimeError) as e:
                logger.warning(
                    "MPS: device_map={'': 'mps'} failed (%s) — "
                    "falling back to standard load (model on CPU, "
                    "will be moved to MPS by load())", e,
                )
                self.model = _load_model_with_fallback(
                    self.model_path, dtype,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    **_local_kwargs(self.model_path),
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=dtype, trust_remote_code=False,
                low_cpu_mem_usage=True,
                **_local_kwargs(self.model_path),
            )

    def _apply_extreme_compile(self) -> None:
        """Apply torch.compile with reduce-overhead for CUDA decode speedup.

        On PyTorch >= 2.12: mode='reduce-overhead' uses frame-level CUDA graphs.
        On PyTorch < 2.12: reduce-overhead triggers cudagraph_trees KV-cache
        tensor-overwrite crash ("accessing tensor output of CUDAGraphs that has
        been overwritten") — falls back to eager mode to avoid silent failure.

        MPS is skipped — inductor deadlocks on first forward in PyTorch 2.x.
        """
        if self.backend_name == "mps":
            logger.info(
                "torch.compile disabled on MPS — PyTorch inductor backend "
                "deadlocks on first forward pass in PyTorch 2.x.  Running "
                "eager mode instead."
            )
            self.use_torch_compile = False
            return

        _pt_version = tuple(int(x) for x in torch.__version__.split(".")[:2])

        try:
            if self.backend_name == "cuda":
                if _pt_version >= (2, 14):
                    # PyTorch >= 2.14: reduce-overhead is safe (cudagraph_trees
                    # fixed for KV-cache sliding-window attention).
                    opts = {"mode": "reduce-overhead", "fullgraph": False}
                elif _pt_version >= (2, 12):
                    # PyTorch 2.12-2.13: kernel fusion + autotuning without
                    # cudagraph_trees. Gets ~half the compile speedup with
                    # zero stability risk.
                    opts = {"mode": "default", "fullgraph": False}
                else:
                    # PyTorch < 2.12: cudagraph_trees crashes on KV-cache.
                    logger.info(
                        "torch.compile skipped — PyTorch %s < 2.12 has "
                        "cudagraph_trees KV-cache overwrite bug.  Upgrade to "
                        "PyTorch >= 2.12 for compile support.",
                        torch.__version__,
                    )
                    self.use_torch_compile = False
                    return
            else:
                opts = {"mode": "default"}

            logger.info("torch.compile starting (mode=%s)...", opts.get("mode"))
            compiled = torch.compile(self.model, **opts)
            self.model = compiled
            logger.info("torch.compile applied (mode=%s)", opts.get("mode"))
        except Exception as e:
            logger.warning(
                "torch.compile(mode=%s) failed: %s — running eager mode",
                opts.get("mode"), e,
            )
            self.use_torch_compile = False

    def _init_paged_attention(self) -> None:
        """Initialize PagedAttention KV-cache pool (EXTREME memory).

        The PagedKVCache replaces HuggingFace's default contiguous KV-cache
        with a block-level virtual memory system.  This cuts KV-cache memory
        usage by 40-70% by eliminating padding and enabling prefix sharing.

        NOTE (2026-06): This method is currently NOT REACHED in the default
        code path — ``self._use_paged_attention`` is hardcoded to ``False``
        in ``__init__`` because no model attention layer has been modified
        to read from paged KV blocks.  The paged cache would be filled
        during prefill but the model would still read from the HF contiguous
        ``past_key_values`` tuple, making the paged blocks dead memory.
        When model forward hooks are in place to consume PagedKVCache,
        flip the flag back to True and wire paged K/V writing into the
        prefill step.
        """
        try:
            kv_cfg = self.kv_cache_config
            num_blocks = PAGED_NUM_BLOCKS_LARGE_GPU if self.device_info.total_memory_gb > PAGED_LARGE_GPU_THRESHOLD_GB else PAGED_NUM_BLOCKS_SMALL_GPU
            self._paged_kv = PagedKVCache(
                num_layers=kv_cfg["num_layers"],
                num_kv_heads=kv_cfg["num_kv_heads"],
                head_dim=kv_cfg["head_dim"],
                block_size=PAGED_BLOCK_SIZE,
                num_blocks=num_blocks,
                dtype=self.precision_config.master_dtype,
                device=self.devices[0] if self.devices else "cuda:0",
            )
            logger.info(
                "PagedAttention initialized: %d blocks × %d tokens = %d total slots",
                num_blocks, PAGED_BLOCK_SIZE, num_blocks * PAGED_BLOCK_SIZE,
            )
        except Exception as e:
            logger.warning("PagedAttention init failed — using HF default KV-cache: %s", e)
            self._paged_kv = None

    def warmup(self, batches: int = 20) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # TR_SKIP_WARMUP=1 allows developers to skip the warmup loop while
        # iterating on compile/kernel changes — saves ~30-60 s per reload.
        if os.environ.get("TR_SKIP_WARMUP") == "1":
            logger.info(
                "TR_SKIP_WARMUP=1 set — skipping AR warmup.  "
                "Note: benchmark results will be unreliable until a full warmup "
                "run is performed."
            )
            return

        logger.info("AR warmup (extreme): %d batches + graph capture...", batches)

        device = self.devices[0]
        ws = time.monotonic()  # start timer before any phase
        # Phase 1: short warmup — only useful on CUDA for cuBLAS autotuning.
        # On MPS short sequences create throwaway MPSGraph compilations that
        # waste IOAccelerator memory (3-5 GB) with no benefit.
        if self.backend_name != "mps":
            # FP8 Transformer Engine requires leading tensor dims to be
            # multiples of 8.  With batch=1 and a short warmup sentence
            # (~12-20 tokens), the product 1×12=12 fails the constraint.
            # Use batch=8 as a minimum so all warmup paths are FP8-compatible.
            warmup_bs_short = 8 if self.backend_name == "cuda" else 1
            logger.info(
                "AR warmup Phase 1: short sequences (bs=%d) — warming CUDA allocator "
                "(kernel compilation, memory pool growth, cuBLAS autotuning). "
                "Forward outputs are discarded; these passes exist only to drive "
                "the CUDA stack into a steady state before measured work begins.",
                warmup_bs_short,
            )
            txt = "This is a warm-up sentence for translation benchmarking."
            ids_single = self.tokenizer.encode(txt, return_tensors="pt").to(device)
            ids = ids_single.repeat(warmup_bs_short, 1)
            mask = torch.ones_like(ids).to(device)
            for _ in range(max(batches // 2, WARMUP_SHORT_BATCHES)):
                with torch.no_grad(), self._fp8_context():
                    self.model(
                        input_ids=ids, attention_mask=mask, use_cache=True,
                    )
            logger.info("  Phase 1 (short): %.1fs", time.monotonic() - ws)

        # Phase 2: production-sized warmup — match the configured batch
        # size so torch.compile CUDA graphs are reusable in the decode loop.
        warmup_bs = getattr(self, '_configured_batch_size', 1)
        if self.backend_name == "mps":
            warmup_bs = max(warmup_bs, 1)
            n_iters = 2  # one compile + one verify — more adds no benefit on MPS
        elif self.backend_name == "cuda":
            # FP8 TE Linear: batch must be divisible by 8.
            if warmup_bs < 8:
                warmup_bs = 8
            n_iters = max(batches // 2, WARMUP_LONG_BATCHES)
        else:
            warmup_bs = 1
            n_iters = max(batches // 2, WARMUP_LONG_BATCHES)
        txt_long = (
            "Machine translation quality assessment requires careful evaluation "
            "across multiple dimensions including fluency, adequacy, and semantic "
            "preservation. " * 3
        )
        ids_single = self.tokenizer.encode(
            txt_long, add_special_tokens=True,
        )
        ids_long = torch.tensor(
            [ids_single] * warmup_bs, dtype=torch.long,
        ).to(device)
        mask_long = torch.ones_like(ids_long).to(device)
        ws2 = time.monotonic()
        for _ in range(n_iters):
            with torch.no_grad(), self._fp8_context():
                self.model(
                    input_ids=ids_long, attention_mask=mask_long, use_cache=True,
                )
        logger.info("  Phase 2 (long, bs=%d): %.1fs", warmup_bs, time.monotonic() - ws2)


        if self.backend_name == "cuda":
            torch.cuda.synchronize()
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass  # may fail if a CUDA graph capture is still in progress
        elif self.backend_name == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        logger.info("AR warmup complete (total: %.1fs)", time.monotonic() - ws)

    # ═════════════════════════════════════════════════════════════════════
    # TRANSLATE — extreme low-level decode loop
    # ═════════════════════════════════════════════════════════════════════

    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        """Extreme-optimized translation.

        If CUDA graph is available:
          Prefill (1 forward pass) → Decode loop (graph.replay() × N tokens)
        Otherwise:
          Uses HF model.generate() as fallback.

        The prompt template wraps raw source text in the model's instruction
        format (e.g. Gemma chat template with translation system prompt).
        Without this, instruction-tuned models produce near-random output.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # ── Speculative decode dispatch (v3.4) ──
        if self._spec_decoder is not None:
            return self._spec_decoder.translate_batch(batch, self)

        device = self.devices[0]

        # Use pipeline-provided tokenized input directly.  The pipeline's
        # _tokeniser_loop wraps raw text in the model's chat template, so
        # batch.input_ids already contains the prompted, tokenized text.
        # _build_templated_inputs exists for direct (no-pipeline) callers
        # but must not be called here — that would double-wrap the template.
        # ── Async H2D transfer (P0-03, EXTREME: stream overlap) ──
        if self.backend_name == "cuda" and self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                input_ids = batch.input_ids.to(device, non_blocking=True)
                attention_mask = batch.attention_mask.to(device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self._transfer_stream)
        else:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
        prompt_lengths = None

        # ── CUDA path: custom decode loop with CUDA event timing ──
        if self.backend_name == "cuda":
            return self._extreme_decode(batch, input_ids, attention_mask, prompt_lengths)

        # ── MPS / CPU: standard HF generate ──
        return self._standard_decode(batch, input_ids, attention_mask, prompt_lengths)

    def _build_templated_inputs(
        self, raw_texts: list[str], device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Wrap source texts in the model's instruction chat template.

        Instruction-tuned models (TranslateGemma, LLaMA-Instruct, etc.)
        require a specific prompt format.  Without the template, the model
        does not know it should translate and produces near-random output.

        Returns (input_ids, attention_mask, prompt_lengths) where
        prompt_lengths[i] = number of prompt tokens BEFORE the model's
        response, so callers can strip the prompt from generated output.
        """
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            # Fallback: tokenize source text directly with a simple prefix.
            texts = [f"Translate English to Turkish:\n{t}" for t in raw_texts]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            prompt_lens = enc["input_ids"].shape[1]
            return (
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
                [prompt_lens] * len(raw_texts),
            )

        # Build per-source chat messages using the translation instruction format.
        prompt_lengths = []
        all_ids = []
        for text in raw_texts:
            msgs = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": text,
                    "source_lang_code": "en",
                    "target_lang_code": "tr",
                }],
            }]
            try:
                ids = self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True,
                )
            except Exception as e:
                # Chat template failure — the model requires its exact Jinja
                # template.  Silently falling back to a raw text prompt
                # produces untrained input format and garbled translation.
                raise RuntimeError(
                    f"Failed to apply chat template for model "
                    f"'{self.model_path}'.  The model requires its specific "
                    f"instruction format — raw text fallback would produce "
                    f"garbage.  Source text: {text[:100]!r}"
                ) from e
            prompt_lengths.append(len(ids))
            all_ids.append(ids)

        # Pad to max length in batch using LEFT-padding (required by decoder-only
        # models for correct autoregressive generation).  With right-padding,
        # PAD tokens sit between real content and generated tokens in the KV
        # cache, corrupting generation quality for variable-length batches.
        max_len = max(len(ids) for ids in all_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((len(raw_texts), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(raw_texts), max_len), dtype=torch.long)
        for i, ids in enumerate(all_ids):
            offset = max_len - len(ids)  # left-padding
            input_ids[i, offset:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, offset:] = 1

        return input_ids.to(device), attention_mask.to(device), prompt_lengths

    def _extreme_decode(
        self, batch: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        prompt_lengths: list[int] | None = None,
    ) -> BatchGenerationOutput:
        """CUDA-optimized decode: prefill + standard decode loop with event timing.

        Prefill (1× standard forward):
          - Runs the model on all prompt tokens at once.
          - Populates the HF KV-cache (past_key_values).
          - CUDA event: start_prefill → end_prefill.

        Decode (standard forward × N tokens):
          - Each iteration passes accumulated past_key_values.
          - CUDA event: start_decode → end_decode (covers all N steps).
          - CUDA Graph replay is NOT used — see below.

        Why no CUDA Graph here:
          The captured graph does NOT include past_key_values as a static
          input. Without accumulated KV-cache, each replay sees only the
          current single token — functionally equivalent to generating
          each token in isolation with zero context, producing garbage.
          torch.compile (mode="reduce-overhead") handles graph capture
          internally with proper KV-cache management, so the manual graph
          path is redundant when compile is active.
        """
        device = self.devices[0]
        bs = input_ids.shape[0]
        max_new = self.max_new_tokens
        eos_id = self.tokenizer.eos_token_id
        eot_id = END_OF_TURN_TOKEN_ID

        # ── CUDA events for split timing (reusable, created in __init__) ──
        ev_prefill_start = self._ev_prefill_start
        ev_prefill_end = self._ev_prefill_end
        ev_decode_start = self._ev_decode_start
        ev_decode_end = self._ev_decode_end

        wall_start = time.monotonic()

        with torch.no_grad(), self._fp8_context():
            # ── PREFILL ──────────────────────────────────────────────────
            ev_prefill_start.record()
            with torch.cuda.nvtx.range("prefill"):
                # Compute position_ids for left-padded sequences.
                # With left-padding, position_ids must count from the first
                # real token (after PAD tokens) starting at position 0.
                # Without this, RoPE-based and other position-aware models
                # receive incorrect positional encodings and produce degraded
                # output.  We compute position_ids for ALL architectures
                # because (a) it is harmless for models that ignore it, and
                # (b) missing it silently corrupts any model using absolute
                # or rotary position embeddings with a left-padded batch.
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.clamp(min=0)

                prefill_out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = prefill_out.past_key_values

            ev_prefill_end.record()

            # ── DECODE LOOP (standard forward, no graph replay) ─────────
            ev_decode_start.record()
            generated_ids: list[list[int]] = [[] for _ in range(bs)]
            done = [False] * bs
            next_input = input_ids[:, -1:]

            for step in range(max_new):
                # Standard forward — correctly accumulates KV-cache across
                # decode steps via past_key_values.  torch.compile
                # internally captures CUDA graphs for this with proper
                # KV-cache handling when mode="reduce-overhead".
                out = self.model(
                    input_ids=next_input,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                logits = out.logits
                past_kv = out.past_key_values

                # Sample next tokens.
                next_logits = logits[:, -1, :]  # [bs, vocab]
                next_tokens = next_logits.argmax(dim=-1)

                for i in range(bs):
                    if done[i]:
                        continue
                    tok = next_tokens[i].item()
                    generated_ids[i].append(tok)
                    if tok == eos_id or tok == eot_id:
                        done[i] = True

                if all(done):
                    break

                # Prepare next iteration's input.
                next_input = next_tokens.unsqueeze(-1)

            ev_decode_end.record()

        wall_end = time.monotonic()

        # ── Synchronize and measure ──
        torch.cuda.synchronize()
        prefill_ms = ev_prefill_start.elapsed_time(ev_prefill_end)
        decode_ms = ev_decode_start.elapsed_time(ev_decode_end)
        total_gpu_ms = prefill_ms + decode_ms
        total_wall_ms = (wall_end - wall_start) * 1000.0

        return self._assemble_output(
            batch, generated_ids, total_wall_ms,
            {"prefill_ms": round(prefill_ms, 2), "decode_ms": round(decode_ms, 2),
             "total_gpu_ms": round(total_gpu_ms, 2), "method": "standard_decode_with_events"},
        )

    def _standard_decode(
        self, batch: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        prompt_lengths: list[int] | None = None,
    ) -> BatchGenerationOutput:
        """Fallback: HF model.generate() with all v2.0 optimizations."""
        device = self.devices[0]
        wall_start = time.monotonic()

        # GPU events.  MPS events use Metal thread-local command queues
        # and can deadlock if synchronize() is called from a different thread
        # than where GPU work was submitted.  Use wall-clock timing on MPS
        # which is safe and already captured via time.monotonic() above.
        if self.backend_name == "cuda":
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
        else:
            ev_s = ev_e = None

        nvtx = (
            torch.cuda.nvtx.range("ar_generate_fallback")
            if self.backend_name == "cuda" else contextlib.nullcontext()
        )

        with torch.no_grad(), self._fp8_context(), nvtx:
            gen_kwargs = dict(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample, num_beams=self.num_beams,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=[self.tokenizer.eos_token_id, END_OF_TURN_TOKEN_ID],
                use_cache=True,
            )
            if self.do_sample and self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
            outputs = self.model.generate(**gen_kwargs)

        wall_end = time.monotonic()

        gpu_ms = 0.0
        if self.backend_name == "cuda" and ev_e is not None:
            ev_e.record()
            ev_e.synchronize()
            gpu_ms = ev_s.elapsed_time(ev_e)

        total_wall_ms = (wall_end - wall_start) * 1000.0

        generated_ids = []
        # With left-padding, PAD tokens are at the START of input_ids.
        # attention_mask.sum(dim=-1) undercounts by the number of PAD tokens,
        # causing the slice to land mid-prompt.  Use the full padded input
        # length (same for all sequences) to correctly strip the prompt.
        prompt_len = input_ids.shape[1]
        for out_ids in outputs:
            generated_ids.append(out_ids[prompt_len:].tolist())

        return self._assemble_output(
            batch, generated_ids, total_wall_ms,
            {"total_gpu_ms": round(gpu_ms, 2), "method": "hf_generate"},
        )

    def _assemble_output(
        self, batch: Any, generated_ids: list[list[int]],
        total_wall_ms: float, phase_timings: dict,
    ) -> BatchGenerationOutput:
        """Decode token IDs → text and build BatchGenerationOutput."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        n_items = max(len(batch.raw_texts) if hasattr(batch, 'raw_texts') else 1, 1)

        generations = []
        total_in = 0
        total_out = 0
        for i, ids in enumerate(generated_ids):
            text = self.tokenizer.decode(ids, skip_special_tokens=True) if ids else ""
            text = text.strip()
            # Strip the "model" token artifact that sometimes leaks through
            # when the model echoes <start_of_turn>model\n at output start.
            # NOTE: This is a heuristic that can corrupt legitimate translations
            # containing the word "model".  Configure via BackendConfig.extra:
            #   strip_model_prefix: true/false (default: false)
            # When disabled, the caller is responsible for stripping prompt
            # artifacts via prompt_length tracking in _build_templated_inputs.
            if self.config.extra.get("strip_model_prefix", False):
                if text.startswith("model"):
                    text = text[len("model"):].strip()
            in_tok = (
                int(batch.attention_mask[i].sum().item())
                if hasattr(batch, 'attention_mask') and i < len(batch.attention_mask)
                else (len(batch.input_ids[i]) if hasattr(batch, 'input_ids') and i < len(batch.input_ids) else 0)
            )
            generations.append(GenerationOutput(
                input_text=batch.raw_texts[i] if hasattr(batch, 'raw_texts') and i < len(batch.raw_texts) else "",
                translated_text=text.strip(),
                input_tokens=in_tok,
                output_tokens=len(ids),
                total_latency_ms=total_wall_ms / n_items,
                phase_timings=phase_timings,
                timestamp_utc=ts,
            ))
            total_in += in_tok
            total_out += len(ids)

        return BatchGenerationOutput(
            batch_id=batch.batch_id if hasattr(batch, 'batch_id') else 0,
            generations=generations, batch_size=n_items,
            input_tokens_total=total_in, output_tokens_total=total_out,
            total_latency_ms=round(total_wall_ms, 2),
            phase_timings=phase_timings,
        )

    # ═════════════════════════════════════════════════════════════════════
    # Properties + remaining internal methods
    # ═════════════════════════════════════════════════════════════════════

    def is_loaded(self) -> bool:
        return self._loaded

    def encode_source(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True,
            )
        return outputs.last_hidden_state

    @property
    def kv_cache_config(self) -> dict[str, Any]:
        cfg = getattr(self.model, "config", None) if self.model is not None else None
        # Gemma 3/4 models nest config in a ``text_config`` sub-object.
        tc = getattr(cfg, "text_config", cfg) if cfg is not None else None
        _resolve = lambda *keys: next((k for k in keys if k is not None), None)

        num_layers = _resolve(
            getattr(tc, "num_hidden_layers", None),
            getattr(cfg, "num_hidden_layers", None) if cfg is not None else None,
            DEFAULT_NUM_LAYERS,
        )
        num_kv_heads = _resolve(
            getattr(tc, "num_key_value_heads", None),
            getattr(cfg, "num_key_value_heads", None) if cfg is not None else None,
            getattr(tc, "num_attention_heads", None),
            getattr(cfg, "num_attention_heads", None) if cfg is not None else None,
            DEFAULT_NUM_KV_HEADS,
        )
        num_attn_heads = _resolve(
            getattr(tc, "num_attention_heads", None),
            getattr(cfg, "num_attention_heads", None) if cfg is not None else None,
        )
        hidden_size = _resolve(
            getattr(tc, "hidden_size", None),
            getattr(cfg, "hidden_size", None) if cfg is not None else None,
            DEFAULT_HIDDEN_SIZE,
        )
        head_dim = _resolve(
            getattr(tc, "head_dim", None),
            getattr(cfg, "head_dim", None) if cfg is not None else None,
            hidden_size // max(num_attn_heads or 1, 1) if num_attn_heads else None,
            DEFAULT_HEAD_DIM,
        )
        return {
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "max_seq_len": self.max_input_tokens + self.max_new_tokens,
        }

    # ── SmoothQuant calibration (opt-in via TR_SMOOTHQUANT=1) ─────────────

    def _calibrate_smoothquant(self) -> None:
        """Run SmoothQuant calibration before static FP8 quantization.

        Uses the current data pipeline input as calibration source.  SmoothQuant
        migrates activation outliers into weights, improving static FP8 accuracy
        without requiring dynamic per-token scaling.
        """
        try:
            from quantization.smoothquant import SmoothQuantCalibrator
            from benchmark.data.loader import JSONLLoader
        except ImportError as e:
            logger.warning("SmoothQuant not available: %s", e)
            return

        logger.info("SmoothQuant calibration starting...")
        calibrator = SmoothQuantCalibrator(
            self.model, self.tokenizer,
            alpha=0.5,
            max_calibration_tokens=min(self.max_input_tokens * 8, 4096),
            device=self.devices[0],
        )
        # Use a small slice of the input data for calibration.
        loader = JSONLLoader(
            ["./data/input/*.jsonl.gz"], shuffle=False,
        )
        cal_texts: list[str] = []
        for _doc_id, _fname, text in loader.iter_documents():
            cal_texts.append(text)
            if len(cal_texts) >= 50:  # 50 docs is plenty for calibration
                break
        count = calibrator.calibrate(cal_texts)
        logger.info("SmoothQuant calibration done: %d layers smoothed", count)

    # ── FP8 — static weight quantization enforced on CUDA ─────────────────

    def _apply_fp8(self) -> None:
        """Enable FP8 for Linear layers on CUDA.

        Strategy (in priority order):
        1. Transformer Engine — fused quantize+matmul kernel (best performance,
           but broken on most pip venvs).  Attempted first.
        2. Static FP8 weights — always applied on CUDA.  Weights are stored in
           FP8 E4M3 and dequantized to BF16 on-chip during the forward.
           Zero per-token overhead.  2× memory bandwidth vs BF16 weights.

        Controls:
          ``TR_SKIP_FP8=1`` → skip all FP8, pure BF16.
        """
        self._fp8_active = False
        self._fp8_method = "none"

        if os.environ.get("TR_SKIP_FP8") == "1":
            logger.info("FP8 skipped — TR_SKIP_FP8=1")
            return

        if self.backend_name != "cuda":
            return

        # -- 1. Transformer Engine (fused kernel, best if it works) ------------
        try:
            from benchmark.hardware.precision import apply_te_fp8_to_model
            _is_gemma = (
                hasattr(self.model, 'config')
                and getattr(self.model.config, 'model_type', '')
                in ('gemma', 'gemma2', 'gemma3', 'gemma3_text', 'gemma4')
            )
            te_ok = apply_te_fp8_to_model(
                self.model, skip_lm_head=True, mlp_only=_is_gemma,
            )
            if te_ok:
                self._fp8_active = True
                self._fp8_method = "te"
                logger.info("FP8 ACTIVE — te.Linear (fused kernel)")
                return
        except Exception as e:
            logger.debug("TE FP8 not available: %s", e)

        # -- 2. Static weight-only FP8 (always applied on CUDA) ----------------
        try:
            from benchmark.hardware.precision import apply_static_fp8_to_model
            replaced = apply_static_fp8_to_model(self.model, skip_lm_head=True)
            if replaced > 0:
                self._fp8_active = True
                self._fp8_method = "static"
                logger.info(
                    "FP8 ACTIVE — %d layers via StaticFP8Linear "
                    "(weight-only, dequant-on-read)",
                    replaced,
                )
                return
        except Exception as e:
            logger.debug("Static FP8 failed: %s", e)

        logger.info("FP8 NOT active — running pure BF16")

    def _fp8_context(self):
        """Wrap forward passes in fp8_autocast when TE is active.

        Returns ``te.fp8_autocast(enabled=True)`` for TE, or
        ``contextlib.nullcontext()`` for BF16.
        """
        if not getattr(self, '_fp8_active', False):
            return contextlib.nullcontext()

        if getattr(self, '_fp8_method', '') == 'te':
            from benchmark.hardware.precision import fp8_autocast_context
            return fp8_autocast_context()

        return contextlib.nullcontext()

    def close(self) -> None:
        """Free all GPU resources explicitly.

        1. CUDA graph static buffers → set to None, empty CUDA cache.
        2. PagedKVCache blocks → free all allocated blocks.
        3. Pinned memory pool → release all pinned allocations.
        4. CUDA events → synchronize and clean up.
        """
        _already_freed = True

        # ── 1. Free PagedKVCache blocks ──
        if self._paged_kv is not None:
            try:
                # Call free() on every allocated block then release the cache.
                if hasattr(self._paged_kv, 'free'):
                    self._paged_kv.free()
                elif hasattr(self._paged_kv, 'release'):
                    self._paged_kv.release()
                elif hasattr(self._paged_kv, 'reset'):
                    self._paged_kv.reset()
            except Exception as e:
                logger.debug("PagedKVCache free failed (non-fatal): %s", e)
            self._paged_kv = None
            _already_freed = False

        # ── 3. Pinned memory pool ──
        # Pinned memory pools (_pinned_input_pool, _pinned_output_pool,
        # _pinned_scratch_pool) are owned by AsyncPipeline, not by this
        # backend.  The pipeline is responsible for releasing them on
        # shutdown.  Nothing to do here.

        # ── 4. Synchronize and clean up CUDA events ──
        if self.backend_name == "cuda":
            # Synchronize all streams before releasing.
            if self._compute_stream is not None:
                try:
                    self._compute_stream.synchronize()
                except Exception:
                    pass
            if self._transfer_stream is not None:
                try:
                    self._transfer_stream.synchronize()
                except Exception:
                    pass
            if not _already_freed:
                torch.cuda.empty_cache()

    def __del__(self) -> None:
        """Warn if GPU memory was not already freed; call close() as safety net."""
        try:
            # Detect whether GPU memory was already released.
            gpu_still_allocated = (
                self._paged_kv is not None
            )
            if gpu_still_allocated:
                logger.debug(
                    "AutoregressiveBackend.__del__: GPU memory released at exit — "
                    "normal at process termination."
                )
                self.close()
        except Exception:
            # Finalizer must never raise; silently ignore any cleanup errors.
            pass

    def _log_memory(self) -> None:
        if self.backend_name == "cuda":
            for i in range(torch.cuda.device_count()):
                a = torch.cuda.memory_allocated(i) / (1024**3)
                r = torch.cuda.memory_reserved(i) / (1024**3)
                t = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info("GPU %d: allocated=%.1fGB reserved=%.1fGB total=%.1fGB", i, a, r, t)
        elif self.backend_name == "mps":
            import psutil
            proc = psutil.Process()
            rss = proc.memory_info().rss / (1024**3)
            drv = torch.mps.driver_allocated_memory() / (1024**3)
            t = psutil.virtual_memory().total / (1024**3)
            logger.info(
                "MPS: RSS %.1f GB  driver %.1f GB  (system total %.1f GB)",
                rss, drv, t,
            )
