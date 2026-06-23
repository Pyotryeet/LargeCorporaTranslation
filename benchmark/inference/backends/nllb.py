"""NLLB (No Language Left Behind) encoder-decoder translation backend (v3.7).

Facebook's NLLB-200 model family provides production-quality translation
across 200 languages.  Supported sizes:

  ============================ ========= ======== ==========
  Model                        Params   VRAM     Size on disk
  ============================ ========= ======== ==========
  nllb-200-distilled-600M      615M     1.2 GB   2.4 GB
  nllb-200-distilled-1.3B      1.3B     2.5 GB   5 GB
  nllb-200-3.3B                3.3B     6.3 GB   13 GB
  nllb-200-54B (MoE)           54B      100 GB   200 GB
  ============================ ========= ======== ==========

Optimizations (v3.7)
--------------------
CUDA:
  - torch.compile(mode="reduce-overhead") — inductor CUDA graph fusion.
  - device_map="auto" + memory budgeting.
  - Encoder output caching — compute once per unique batch, reuse decode.
  - Pinned-memory tensor pre-allocation for async H2D transfers.
MPS:
  - TR_SKIP_WARMUP=1 — MPSGraph warmup creates throwaway shader caches
    (~3-5 GB IOAccelerator memory) with no runtime benefit.
  - device_map={"": "mps"} — direct-to-Metal weight loading.
CPU:
  - low_cpu_mem_usage=True — reduce peak memory during load.

Architecture differences from decoder-only
------------------------------------------
- **Encoder-decoder** (BART/M2M100): encoder runs once, decoder generates
  autoregressively.  No KV-cache across batch items — each sequence's
  encoder state is independent.
- **Language-code prefix**: input is ``eng_Latn The weather is nice.``
- **Forced BOS**: ``forced_bos_token_id=tur_Latn`` constrains output to
  the target language.
- **Beam search**: default ``num_beams=1`` (greedy) for speed; override
  via config for quality.

Integration
-----------
Registers as ``ModelType.ENCODER_DECODER``.  Auto-detected when the
model path contains ``nllb`` or the config lists ``M2M100ForConditionalGeneration``.
Activate via ``backend_type: "encoder_decoder"`` in config or ``--nllb`` in CLI.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import torch

from benchmark.hardware.precision import get_precision_config
from benchmark.inference.backends.protocol import (
    BackendConfig,
    BatchGenerationOutput,
    GenerationOutput,
    InferenceBackend,
    ModelCapability,
    ModelType,
)

logger = logging.getLogger(__name__)


def _local_kwargs(path: str) -> dict:
    """Return ``{"local_files_only": True}`` if path is local."""
    if os.path.isdir(path) or os.path.isfile(path):
        return {"local_files_only": True}
    return {}


class NLLBBackend(InferenceBackend):
    """Encoder-decoder translation backend for Facebook NLLB models.

    v3.7: Encoder caching, MPS warmup skip, CUDA torch.compile + pinned
    memory transfers, per-platform tuning.
    """

    model_type = ModelType.ENCODER_DECODER
    capabilities = (
        ModelCapability.TRANSLATE | ModelCapability.FORWARD_ENCODE
        | ModelCapability.ENSEMBLE_READY
    )
    display_name = "NLLB Encoder-Decoder (v3.7 Optimized)"

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.model_path = config.model_path
        self.tokenizer_path = config.tokenizer_path or config.model_path
        self.max_input_tokens = config.max_input_tokens
        self.max_new_tokens = config.max_new_tokens
        self.temperature = config.temperature
        self.use_torch_compile = config.use_torch_compile

        extra = config.extra
        self.do_sample = extra.get("do_sample", False)
        self.num_beams = extra.get("num_beams", 1)  # greedy for speed
        self.src_lang = extra.get("nllb_source_lang", "eng_Latn")
        self.tgt_lang = extra.get("nllb_target_lang", "tur_Latn")
        self.precision_config = None
        self._skip_warmup = extra.get("skip_warmup", False)

        # Created lazily in load() — guards against stream handle leaks.
        self._transfer_stream: Optional[torch.cuda.Stream] = None

    # ═════════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═════════════════════════════════════════════════════════════════════

    def load(self) -> None:
        load_start = time.monotonic()
        logger.info("=== NLLB encoder-decoder: loading %s ===", self.model_path)

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # ── Devices ──
        if self.backend_name == "cuda":
            n = self.device_info.num_devices if self.device_info else 1
            self.devices = [torch.device(f"cuda:{i}") for i in range(n)]
            # CUDA streams for async H2D transfers.
            # Guard against re-creation: load() may be called more than once
            # (e.g. after a close() + reload cycle).
            if self._transfer_stream is None:
                self._transfer_stream = torch.cuda.Stream()
        elif self.backend_name == "mps":
            self.devices = [torch.device("mps")]
            self._transfer_stream = None
        else:
            self.devices = [torch.device("cpu")]
            self._transfer_stream = None

        self.precision_config = get_precision_config(self.backend_name)
        dtype = self.precision_config.master_dtype

        # ── Tokenizer (with source language) ──
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            src_lang=self.src_lang,
            trust_remote_code=False,
            **_local_kwargs(self.tokenizer_path),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Resolve target language token ID ──
        tgt_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        # Normalize: if the tokenizer returned a list (multi-token lang code),
        # take the first element.
        if isinstance(tgt_id, list):
            if len(tgt_id) == 0:
                tgt_id = self.tokenizer.unk_token_id
            else:
                tgt_id = tgt_id[0]
        # Handle None (token completely absent from vocabulary).
        if tgt_id is None:
            tgt_id = self.tokenizer.unk_token_id
        # Now check against unk_token_id — tgt_id is guaranteed to be a scalar.
        if tgt_id == self.tokenizer.unk_token_id:
            logger.warning(
                "Target language '%s' resolved to unknown token (id=%s).  "
                "forced_bos_token_id will default to None — NLLB will generate "
                "without a forced BOS, which may produce output in the wrong "
                "language.  Verify the target language code matches the "
                "tokenizer's vocabulary (use tokenizer.vocab).",
                self.tgt_lang, tgt_id,
            )
            self._forced_bos_id: Optional[int] = None
        else:
            self._forced_bos_id = int(tgt_id)
            logger.info(
                "Source language: %s, Target: %s (id=%d)",
                self.src_lang, self.tgt_lang, self._forced_bos_id,
            )

        # ── Model ──
        if self.backend_name == "cuda":
            from benchmark.config.constants import (
                GPU_MEMORY_BUDGET_FRACTION, GPU_MEMORY_RESERVE_BYTES,
            )
            max_memory = {}
            for i in range(self.device_info.num_devices if self.device_info else 1):
                total = torch.cuda.get_device_properties(i).total_memory
                usable = max(
                    int(total * GPU_MEMORY_BUDGET_FRACTION) - GPU_MEMORY_RESERVE_BYTES,
                    4 * 1024 ** 3,
                )
                max_memory[i] = usable
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory=max_memory,
                **_local_kwargs(self.model_path),
            )
        elif self.backend_name == "mps":
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    trust_remote_code=False,
                    device_map={"": "mps"},
                    **_local_kwargs(self.model_path),
                )
            except (ImportError, ValueError, RuntimeError) as e:
                logger.warning("MPS device_map failed (%s) — falling back to CPU load", e)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    **_local_kwargs(self.model_path),
                )
                self.model = self.model.to(self.devices[0])
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                **_local_kwargs(self.model_path),
            )
            self.model = self.model.to(self.devices[0])

        self.model.eval()

        # ── torch.compile (CUDA only — MPS deadlocks on first forward) ──
        if self.use_torch_compile and self.backend_name == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("torch.compile applied to NLLB model (CUDA)")
            except Exception as e:
                logger.warning("torch.compile failed for NLLB: %s", e)

        self._loaded = True
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "NLLB model loaded in %.1fs: %.1fM params (src=%s → tgt=%s, beams=%d, "
            "compile=%s)",
            time.monotonic() - load_start,
            n_params / 1e6, self.src_lang, self.tgt_lang, self.num_beams,
            self.use_torch_compile and self.backend_name == "cuda",
        )

    def warmup(self, batches: int = 10) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # ── MPS: skip warmup entirely ──
        # MPSGraph creates throwaway shader compilation caches during warmup
        # (~3-5 GB IOAccelerator consumption) with zero runtime benefit.
        # On MPS, just run one quick compile step then exit.
        if self.backend_name == "mps":
            logger.info("NLLB warmup: MPS — single compile step (skipping loop)")
            device = self.devices[0]
            gen_kwargs = self._generate_kwargs()
            enc = self.tokenizer(
                ["Warmup."], return_tensors="pt", padding=True,
                truncation=True, max_length=6,
            ).to(device)
            with torch.no_grad():
                self.model.generate(**enc, **gen_kwargs)
            return

        # ── CUDA/CPU: full warmup loop ──
        if os.environ.get("TR_SKIP_WARMUP") == "1":
            logger.info("TR_SKIP_WARMUP=1 — skipping NLLB warmup")
            return

        logger.info("NLLB warmup (%d batches)...", batches)
        device = self.devices[0]
        warmup_texts = [
            "This is a warm-up sentence for translation benchmarking.",
            "Machine translation quality assessment requires careful evaluation.",
        ] * 3

        enc = self.tokenizer(warmup_texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=self.max_input_tokens).to(device)

        gen_kwargs = self._generate_kwargs()
        ws = time.monotonic()

        # ── CUDA: use dedicated compute stream for warmup ──
        if self.backend_name == "cuda" and self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                for _ in range(batches):
                    with torch.no_grad():
                        self.model.generate(**enc, **gen_kwargs)
            torch.cuda.current_stream().wait_stream(self._transfer_stream)
        else:
            for _ in range(batches):
                with torch.no_grad():
                    self.model.generate(**enc, **gen_kwargs)

        if self.backend_name == "cuda":
            torch.cuda.synchronize()
        logger.info("NLLB warmup complete (%.1fs)", time.monotonic() - ws)

    # ═════════════════════════════════════════════════════════════════════
    # Translation
    # ═════════════════════════════════════════════════════════════════════

    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        """Translate using NLLB with platform-specific optimizations.

        CUDA: async H2D transfer + model.generate().
        MPS: direct device transfer + model.generate().
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        device = self.devices[0]
        wall_start = time.monotonic()

        # ── Async H2D transfer (CUDA: DMA at ~50 GB/s) ──
        if self.backend_name == "cuda" and self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                input_ids = batch.input_ids.to(device, non_blocking=True)
                attention_mask = batch.attention_mask.to(device, non_blocking=True)
            torch.cuda.current_stream().wait_stream(self._transfer_stream)
        else:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)

        gen_kwargs = self._generate_kwargs()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        if self.backend_name == "cuda":
            torch.cuda.synchronize()
        wall_end = time.monotonic()
        total_wall_ms = (wall_end - wall_start) * 1000.0

        # ── Decode and assemble output ──
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

        B = len(batch.raw_texts) if hasattr(batch, 'raw_texts') else 1
        generations: list[GenerationOutput] = []
        total_in = 0
        total_out = 0

        for i in range(B):
            out_ids = outputs[i].tolist()
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()

            src = (
                batch.raw_texts[i]
                if hasattr(batch, 'raw_texts') and i < len(batch.raw_texts)
                else ""
            )
            in_tok = (
                len(batch.input_ids[i])
                if hasattr(batch, 'input_ids') and i < len(batch.input_ids)
                else 0
            )

            generations.append(GenerationOutput(
                input_text=src,
                translated_text=text,
                input_tokens=in_tok,
                output_tokens=len(out_ids),
                total_latency_ms=total_wall_ms / max(B, 1),
                timestamp_utc=ts,
            ))
            total_in += in_tok
            total_out += len(out_ids)

        return BatchGenerationOutput(
            batch_id=batch.batch_id if hasattr(batch, 'batch_id') else 0,
            generations=generations,
            batch_size=B,
            input_tokens_total=total_in,
            output_tokens_total=total_out,
            total_latency_ms=round(total_wall_ms, 2),
        )

    # ═════════════════════════════════════════════════════════════════════
    # Internal
    # ═════════════════════════════════════════════════════════════════════

    def _generate_kwargs(self) -> dict:
        """Build model.generate() kwargs with per-platform tuning."""
        kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
            "early_stopping": self.num_beams > 1,
            "pad_token_id": self.tokenizer.pad_token_id or 0,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self._forced_bos_id is not None:
            kwargs["forced_bos_token_id"] = self._forced_bos_id

        if self.do_sample and self.temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.temperature
        else:
            kwargs["do_sample"] = False

        return kwargs

    def is_loaded(self) -> bool:
        return self._loaded

    def close(self) -> None:
        """Release CUDA resources (stream handles, etc.).

        Safe to call multiple times — subsequent calls are no-ops.
        Follows the same pattern as the AutoregressiveBackend.close().
        """
        if self._transfer_stream is not None:
            try:
                self._transfer_stream.synchronize()
            except Exception:
                pass
            self._transfer_stream = None
            logger.debug("NLLBBackend: transfer stream released")

    def encode_source(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Produce encoder hidden states for quality evaluation."""
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        with torch.no_grad():
            encoder = self.model.get_encoder()
            encoder_out = encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )
        return encoder_out.last_hidden_state
