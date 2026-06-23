"""TensorRT engine builder — ONNX export + TRT optimization + disk cache.

Converts a HuggingFace autoregressive model into a TensorRT serialized engine
(.engine file) with layer fusion, FP8/FP16/INT8 precision, kernel auto-tuning,
and shape optimization profiles.

Architecture
------------
  1. Export PyTorch model → ONNX (with dynamic batch × seq axes).
  2. Build TensorRT engine from ONNX (with calibration for INT8).
  3. Serialize engine to disk → ~/.cache/tr_benchmark/engines/<hash>.engine.
  4. Load engine on subsequent runs (instant, ~50ms vs 5-15 minutes).

Engine cache key
----------------
  SHA256(model_path + GPU_name + max_batch + max_seq + precision + calibration_hash).
  Different GPUs get different engines (SM80 vs SM90 instruction sets differ).

Precision modes
---------------
  fp16  — Fast, universally supported.  2× throughput vs FP32, no calibration.
  fp8   — Hopper only (SM90).  Requires calibration.  2× vs FP16 for matmul.
  int8  — Requires calibration dataset.  2-3× throughput, slight quality cost.

Requirements
------------
  tensorrt>=10.0, onnx>=1.16, onnxruntime, CUDA 12.4+, NVIDIA GPU (SM75+).

Graceful fallback
------------------
  If tensorrt or onnx are not installed, all functions return None and the
  caller falls back to the standard extreme-optimized AR backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Optional imports ────────────────────────────────────────────────────────

try:
    import tensorrt as trt
    HAS_TRT = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    HAS_TRT = False
    TRT_LOGGER = None

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ── Constants ───────────────────────────────────────────────────────────────

ENGINE_CACHE_ROOT = Path.home() / ".cache" / "tr_benchmark" / "engines"
TRT_PRECISIONS = ("fp16", "fp8", "int8")


def _detect_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    # Sanitize for filename.
    return name.replace(" ", "_").replace("-", "_").replace("/", "_")


def _detect_sm() -> str:
    if not torch.cuda.is_available():
        return "none"
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm{major}{minor}"


# ═══════════════════════════════════════════════════════════════════════════
# TensorRT Engine Builder
# ═══════════════════════════════════════════════════════════════════════════


class TRTEngineBuilder:
    """Builds and caches TensorRT engines for HF autoregressive models.

    Usage
    -----
    >>> builder = TRTEngineBuilder()
    >>> engine_path = builder.build_or_load(
    ...     model, tokenizer,
    ...     max_batch=32, max_input=512, max_output=512,
    ...     precision="fp16",
    ... )
    >>> # engine_path is a .engine file on disk — ready for TRTEngineBackend.
    """

    def __init__(self, cache_dir: Path | str = ENGINE_CACHE_ROOT):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_name = _detect_gpu_name()
        self._gpu_sm = _detect_sm()

    def build_or_load(
        self,
        model_path: str,
        model: nn.Module,
        tokenizer: Any,
        *,
        max_batch: int = 32,
        max_input_tokens: int = 512,
        max_output_tokens: int = 512,
        precision: str = "fp16",
        calibration_data: list[str] | None = None,
        force_rebuild: bool = False,
    ) -> Optional[str]:
        """Build a TensorRT engine or load from cache.

        Parameters
        ----------
        model_path : str
            HuggingFace model ID or local path (for cache key).
        model : nn.Module
            Loaded PyTorch model.
        tokenizer :
            HF tokenizer compatible with the model.
        max_batch : int
            Maximum batch size for the optimization profile.
        max_input_tokens : int
            Maximum input sequence length.
        max_output_tokens : int
            Maximum output (generated) sequence length.
        precision : str
            "fp16", "fp8", or "int8".
        calibration_data : list[str], optional
            Representative English text for INT8/FP8 calibration (100-500 samples).
        force_rebuild : bool
            If True, rebuild even if a cached engine exists.

        Returns
        -------
        str or None
            Path to .engine file, or None if TensorRT/ONNX unavailable.
        """
        if not HAS_TRT or not HAS_ONNX or not HAS_NUMPY:
            missing = []
            if not HAS_TRT: missing.append("tensorrt")
            if not HAS_ONNX: missing.append("onnx")
            logger.info("TensorRT engine build skipped — missing: %s", ", ".join(missing))
            return None

        if precision not in TRT_PRECISIONS:
            raise ValueError(f"precision must be one of {TRT_PRECISIONS}, got {precision!r}")
        if precision == "fp8" and self._gpu_sm != "sm90":
            logger.warning("FP8 requires Hopper (SM90) — falling back to FP16")
            precision = "fp16"

        # ── Cache key ──
        cache_key = self._engine_cache_key(
            model_path, max_batch, max_input_tokens, max_output_tokens,
            precision, calibration_data,
        )
        engine_path = self.cache_dir / f"{cache_key}.engine"

        # ── Cache hit ──
        if engine_path.exists() and not force_rebuild:
            logger.info("TensorRT engine cache HIT: %s", engine_path.name[:32])
            return str(engine_path)

        # ── Cache miss — build ──
        logger.info(
            "TensorRT engine cache MISS — building (precision=%s, bs=%d, seq=%d)...",
            precision, max_batch, max_input_tokens,
        )
        build_start = time.monotonic()

        try:
            # Use a TemporaryDirectory for the ONNX export that lives through
            # the TRT build step — avoids the mkdtemp leak.
            with tempfile.TemporaryDirectory(prefix="tr_benchmark_onnx_") as tmpdir:
                # Step 1: Export ONNX.
                onnx_path = self._export_onnx(
                    model, tokenizer, max_batch, max_input_tokens,
                    output_dir=tmpdir,
                )
                if onnx_path is None:
                    return None

                # Step 2: Build TensorRT engine.
                self._build_trt_engine(
                    onnx_path, str(engine_path),
                    max_batch, max_input_tokens, max_output_tokens,
                    precision, calibration_data, model_path,
                )

            elapsed = time.monotonic() - build_start
            size_mb = engine_path.stat().st_size / (1024 * 1024)
            logger.info(
                "TensorRT engine built in %.1fs — %s (%.1f MB)",
                elapsed, engine_path.name[:32], size_mb,
            )

            return str(engine_path)

        except Exception as e:
            # Provide an actionable diagnostic tailored to common failure modes.
            err_msg = str(e).lower()
            hints = []
            if "out of memory" in err_msg or "oom" in err_msg:
                hints.append(
                    "GPU out of memory — reduce max_batch or max_input_tokens, "
                    "or lower workspace limit via "
                    "config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, N)."
                )
            elif "onnx" in err_msg:
                hints.append(
                    "ONNX export/parse error — check that the model supports "
                    "torch.onnx.export (no data-dependent control flow, no "
                    "custom autograd functions).  Use opset_version=17 or later."
                )
            elif "cuda" in err_msg or "cudnn" in err_msg or "nvcc" in err_msg:
                hints.append(
                    "CUDA/TensorRT compatibility error — verify that the "
                    "installed TensorRT version matches the CUDA toolkit "
                    "(tensorrt>=10.0 requires CUDA 12.4+)."
                )
            elif "calibration" in err_msg or "calibrat" in err_msg:
                hints.append(
                    "Calibration failed — ensure calibration_data provides "
                    "100+ representative text samples covering the expected "
                    "input distribution (diverse sentence lengths, punctuation, "
                    "special tokens)."
                )
            elif "serialized" in err_msg or "returned none" in err_msg:
                hints.append(
                    "Builder returned None — usually caused by an unsupported "
                    "op in the ONNX graph.  Inspect ONNX with "
                    "'onnx.checker.check_model()' and verify all ops are "
                    "in the TensorRT operator support matrix."
                )
            hint_text = " " + " ".join(hints) if hints else ""
            logger.error(
                "TensorRT engine build FAILED for model_path=%r precision=%s "
                "max_batch=%d max_input=%d max_output=%d: %s.%s",
                model_path,
                precision,
                max_batch,
                max_input_tokens,
                max_output_tokens,
                e,
                hint_text,
                exc_info=True,
            )
            # Clean up partial engine.
            engine_path.unlink(missing_ok=True)
            return None

    def _export_onnx(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_batch: int,
        max_input: int,
        output_dir: str | None = None,
    ) -> Optional[str]:
        """Export the model to ONNX with dynamic axes for batch x sequence.

        We export TWO ONNX models:
          1. encoder.onnx — encodes the full prompt in one pass.  Uses the
             model's forward without KV-cache.
          2. decoder.onnx — one decode step.  Takes (input_ids, KV-cache_in)
             -> (logits, KV-cache_out).  This is what the TensorRT engine
             replays for each token.

        For simplicity, we export a single combined model.  The decoder-only
        architecture (TranslateGemma, LLaMA) uses causal attention — the
        encoder and decoder are the same model, just with different inputs.
        """
        model.eval()
        device = next(model.parameters()).device

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="tr_benchmark_onnx_")
        onnx_path = os.path.join(output_dir, "model.onnx")

        # Sample input for tracing.
        sample_text = "The quick brown fox jumps over the lazy dog. " * 5
        enc = tokenizer(
            sample_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        )
        sample_ids = enc["input_ids"].to(device)
        sample_mask = enc["attention_mask"].to(device)

        # Clamp to max_batch by padding.
        if sample_ids.shape[0] < max_batch:
            pad_id = tokenizer.pad_token_id or 0
            sample_ids = torch.cat([
                sample_ids,
                torch.full((max_batch - sample_ids.shape[0], sample_ids.shape[1]),
                           pad_id, dtype=torch.long, device=device),
            ], dim=0)
            sample_mask = torch.cat([
                sample_mask,
                torch.zeros(max_batch - sample_mask.shape[0], sample_mask.shape[1],
                            dtype=torch.long, device=device),
            ], dim=0)

        logger.info("Exporting ONNX (bs=%d, seq=%d)...", sample_ids.shape[0], sample_ids.shape[1])

        try:
            # Use torch.onnx.export with dynamic axes.
            torch.onnx.export(
                model,
                (sample_ids, sample_mask),
                onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset_version=17,
                do_constant_folding=True,
            )

            # Verify ONNX model.
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            logger.info(
                "ONNX exported: %s (%d nodes)",
                onnx_path, len(onnx_model.graph.node),
            )
            return onnx_path

        except (RuntimeError, ValueError, TypeError) as e:
            err_msg = str(e).lower()
            hints = []
            if "trace" in err_msg or "dynamic" in err_msg:
                hints.append(
                    "The model may contain data-dependent control flow or "
                    "dynamic shapes unsupported by torch.onnx.export.  "
                    "Try wrapping the forward pass to use static shapes."
                )
            elif "dtype" in err_msg or "type" in err_msg:
                hints.append(
                    "Unsupported dtype in model parameters or inputs.  "
                    "Ensure all tensors are float32 or float16 — bfloat16 "
                    "is not supported by ONNX opset 17."
                )
            if hints:
                logger.error("ONNX export failed: %s.  %s", e, "  ".join(hints))
            else:
                logger.error("ONNX export failed (max_batch=%d, max_input=%d): %s",
                             max_batch, max_input, e)

    def _build_trt_engine(
        self,
        onnx_path: str,
        engine_path: str,
        max_batch: int,
        max_input: int,
        max_output: int,
        precision: str,
        calibration_data: list[str] | None,
        tokenizer_path: str,
    ) -> None:
        """Build a TensorRT engine from an ONNX model.

        Uses the TensorRT Python API directly.
        """
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        config = builder.create_builder_config()

        # ── Memory pool ──
        # Limit workspace to 8 GB — enough for 12B models.
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024**3)

        # ── Precision ──
        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("TRT builder: FP16 mode")
        elif precision == "fp8":
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.FP8)
            logger.info("TRT builder: FP8 mode (Hopper)")
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            if calibration_data is not None:
                config.int8_calibrator = self._create_calibrator(
                    onnx_path, calibration_data, max_batch, max_input,
                    tokenizer_path,
                )
            logger.info("TRT builder: INT8 mode (with calibration)")

        # ── Parse ONNX ──
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parse errors: {errors}")

        # ── Optimization profile (dynamic shapes) ──
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)  # input_ids
        mask_tensor = network.get_input(1)    # attention_mask

        # Min / Opt / Max shapes.
        min_bs, opt_bs, max_bs = 1, max(1, max_batch // 2), max_batch
        min_seq, opt_seq, max_seq = 1, max_input, max_input + max_output

        profile.set_shape(
            input_tensor.name,
            (min_bs, min_seq), (opt_bs, opt_seq), (max_bs, max_seq),
        )
        profile.set_shape(
            mask_tensor.name,
            (min_bs, min_seq), (opt_bs, opt_seq), (max_bs, max_seq),
        )
        config.add_optimization_profile(profile)

        # ── Build ──
        logger.info("Building TensorRT engine (this may take 5-15 minutes)...")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build returned None")

        # ── Write engine to disk ──
        with open(engine_path, "wb") as f:
            f.write(serialized)

    def _create_calibrator(
        self,
        onnx_path: str,
        calibration_texts: list[str],
        max_batch: int,
        max_input: int,
        tokenizer_path: str,
    ):
        """Create an INT8 calibrator from representative text samples.

        Uses a simple batch-stream calibrator that feeds calibration data
        through the ONNX model to measure activation ranges per layer.
        """
        try:
            import onnxruntime as ort

            class _Calibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, texts, max_bs, max_seq, onnx_path, tokenizer_path):
                    super().__init__()
                    self.texts = texts
                    self.max_bs = max_bs
                    self.max_seq = max_seq
                    self.onnx_path = onnx_path
                    self._session = None
                    self._input_name = None
                    self._cache_path = onnx_path + ".calibration_cache"
                    # Use the caller-supplied tokenizer path instead of a
                    # hardcoded model name.
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path,
                    )
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                def get_batch_size(self):
                    return self.max_bs

                def get_batch(self, names):
                    if not self.texts:
                        return None
                    batch_texts = self.texts[:self.max_bs]
                    self.texts = self.texts[self.max_bs:]
                    enc = self.tokenizer(
                        batch_texts, padding=True, truncation=True,
                        max_length=self.max_seq, return_tensors="np",
                    )
                    # Pad to max_batch.
                    bs = enc["input_ids"].shape[0]
                    if bs < self.max_bs:
                        pad = np.zeros(
                            (self.max_bs - bs, self.max_seq), dtype=np.int64,
                        )
                        enc["input_ids"] = np.concatenate(
                            [enc["input_ids"], pad], axis=0,
                        )
                        enc["attention_mask"] = np.concatenate(
                            [enc["attention_mask"], pad], axis=0,
                        )
                    return [np.ascontiguousarray(enc["input_ids"])]

                def read_calibration_cache(self):
                    if os.path.exists(self._cache_path):
                        with open(self._cache_path, "rb") as f:
                            return f.read()
                    return None

                def write_calibration_cache(self, cache):
                    with open(self._cache_path, "wb") as f:
                        f.write(cache)

            return _Calibrator(
                calibration_texts, max_batch, max_input, onnx_path,
                tokenizer_path,
            )

        except ImportError:
            logger.warning(
                "onnxruntime not available — INT8 calibration skipped. "
                "Falling back to FP16."
            )
            return None

    def _engine_cache_key(
        self,
        model_path: str,
        max_batch: int,
        max_input: int,
        max_output: int,
        precision: str,
        calibration_data: list[str] | None,
    ) -> str:
        """Generate a deterministic cache key for the engine."""
        parts = [
            model_path,
            self._gpu_name,
            self._gpu_sm,
            f"bs{max_batch}",
            f"in{max_input}",
            f"out{max_output}",
            precision,
        ]
        if calibration_data:
            # Hash the calibration data to detect changes.
            cal_hash = hashlib.sha256(
                "\n".join(calibration_data[:20]).encode()
            ).hexdigest()[:12]
            parts.append(f"cal{cal_hash}")

        full = "_".join(parts)
        return hashlib.sha256(full.encode()).hexdigest()[:32]


# ═══════════════════════════════════════════════════════════════════════════
# TensorRT Runtime (engine loader)
# ═══════════════════════════════════════════════════════════════════════════


class TRTRuntime:
    """Load and run a serialized TensorRT engine.

    Wraps the TensorRT execution context for efficient per-batch inference.
    Supports multiple optimization profiles for different batch sizes.

    Usage
    -----
    >>> runtime = TRTRuntime("path/to/engine.engine")
    >>> logits = runtime.infer(input_ids, attention_mask)
    """

    def __init__(self, engine_path: str):
        if not HAS_TRT:
            raise RuntimeError("TensorRT not installed")

        self.engine_path = engine_path
        self._runtime: Optional[trt.Runtime] = None
        self._engine: Optional[trt.ICudaEngine] = None
        self._context: Optional[trt.IExecutionContext] = None
        self._stream: Optional[torch.cuda.Stream] = None

        # I/O bindings.
        self._input_ids_binding: int = -1
        self._mask_binding: int = -1
        self._output_binding: int = -1
        self._input_shape: tuple = ()
        self._output_shape: tuple = ()

        self._loaded = False

    def load(self) -> None:
        """Load the serialized engine and create execution context."""
        logger.info("Loading TensorRT engine: %s", self.engine_path)
        load_start = time.monotonic()

        self._runtime = trt.Runtime(TRT_LOGGER)
        with open(self.engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {self.engine_path}")

        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream()

        # Resolve bindings.
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if name == "input_ids":
                self._input_ids_binding = i
            elif name == "attention_mask":
                self._mask_binding = i
            elif name == "logits":
                self._output_binding = i

        self._loaded = True
        logger.info(
            "TRT engine loaded in %.1fs (%d layers, %d I/O tensors)",
            time.monotonic() - load_start,
            self._engine.num_layers,
            self._engine.num_io_tensors,
        )

    def infer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference with the TensorRT engine.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape [batch, seq_len], dtype long, device cuda.
        attention_mask : torch.Tensor
            Shape [batch, seq_len], dtype long, device cuda.

        Returns
        -------
        torch.Tensor
            Logits of shape [batch, seq_len, vocab_size].
        """
        if not self._loaded:
            raise RuntimeError("TRT engine not loaded")

        bs, seq = input_ids.shape

        # Set input shapes in the execution context.
        self._context.set_input_shape("input_ids", (bs, seq))
        self._context.set_input_shape("attention_mask", (bs, seq))

        # Allocate output — we need output shape from the engine.
        # Since the output has dynamic vocab, query the engine.
        out_shape = self._context.get_tensor_shape("logits")
        if out_shape[0] == -1:  # dynamic batch
            out_shape = (bs, seq, self._engine.get_binding_max_shape(self._output_binding)[2])

        output = torch.empty(out_shape, dtype=torch.float16, device="cuda")

        # Bind I/O.
        self._context.set_tensor_address("input_ids", input_ids.data_ptr())
        self._context.set_tensor_address("attention_mask", attention_mask.data_ptr())
        self._context.set_tensor_address("logits", output.data_ptr())

        # Execute asynchronously.
        self._context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        return output

    def is_loaded(self) -> bool:
        return self._loaded

    def close(self) -> None:
        """Release TensorRT resources."""
        self._loaded = False
        del self._context
        del self._engine
        del self._runtime
        del self._stream


# ═══════════════════════════════════════════════════════════════════════════
# Convenience — build engine from model + config
# ═══════════════════════════════════════════════════════════════════════════


def build_engine_if_needed(
    model_path: str,
    model: nn.Module,
    tokenizer: Any,
    *,
    max_batch: int = 32,
    max_input: int = 512,
    max_output: int = 512,
    precision: str = "fp16",
    calibration_texts: list[str] | None = None,
    force_rebuild: bool = False,
) -> Optional[str]:
    """Convenience: build TensorRT engine if TensorRT is available.

    Returns engine path on success, None if TensorRT unavailable.
    """
    builder = TRTEngineBuilder()
    return builder.build_or_load(
        model_path, model, tokenizer,
        max_batch=max_batch,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        precision=precision,
        calibration_data=calibration_texts,
        force_rebuild=force_rebuild,
    )
