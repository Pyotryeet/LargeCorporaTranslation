"""NLLB CUDA optimized translation backend (v3.7).

Facebook's NLLB-200 model family optimized specifically for NVIDIA devices.
Uses torch.compile, memory mapping, stream transfers, and memory budgeting.
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
from benchmark.utils.helpers import local_kwargs as _local_kwargs

logger = logging.getLogger(__name__)


class NLLBCUDABackend(InferenceBackend):
    """NVIDIA CUDA optimized backend for NLLB models."""

    model_type = ModelType.ENCODER_DECODER
    capabilities = (
        ModelCapability.TRANSLATE | ModelCapability.FORWARD_ENCODE
        | ModelCapability.ENSEMBLE_READY
    )
    display_name = "NLLB Encoder-Decoder (CUDA Optimized)"

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
        self.num_beams = extra.get("num_beams", 1)
        self.src_lang = extra.get("nllb_source_lang", "eng_Latn")
        self.tgt_lang = extra.get("nllb_target_lang", "tur_Latn")
        self.precision_config = None
        self._skip_warmup = extra.get("skip_warmup", False)
        self._transfer_stream: Optional[torch.cuda.Stream] = None

    def load(self) -> None:
        load_start = time.monotonic()
        logger.info("=== NLLB CUDA: loading %s ===", self.model_path)

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.device_info and hasattr(self.device_info, 'device') and self.device_info.device is not None:
            self.devices = [self.device_info.device]
        else:
            n = self.device_info.num_devices if self.device_info else 1
            self.devices = [torch.device(f"cuda:{i}") for i in range(n)]

        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream()

        self.precision_config = get_precision_config(self.backend_name)
        dtype = self.precision_config.master_dtype

        _is_madlad = "madlad" in self.tokenizer_path.lower()
        _tok_kwargs: dict = {"trust_remote_code": False, **_local_kwargs(self.tokenizer_path)}
        if not _is_madlad:
            _tok_kwargs["src_lang"] = self.src_lang
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            **_tok_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tgt_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        if isinstance(tgt_id, list):
            tgt_id = tgt_id[0] if len(tgt_id) > 0 else self.tokenizer.unk_token_id
        if tgt_id is None:
            tgt_id = self.tokenizer.unk_token_id

        if tgt_id == self.tokenizer.unk_token_id:
            logger.warning("Target language '%s' resolved to unknown token.", self.tgt_lang)
            self._forced_bos_id = None
        else:
            self._forced_bos_id = int(tgt_id)

        from benchmark.config.constants import (
            GPU_MEMORY_BUDGET_FRACTION, GPU_MEMORY_RESERVE_BYTES,
        )
        n_devs = self.device_info.num_devices if self.device_info else 1

        single_gpu = False
        _est_bytes = 0.0
        single_gpu_mem = torch.cuda.get_device_properties(self.devices[0]).total_memory
        try:
            from transformers import AutoConfig as _AutoConfig
            _cfg = _AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=False,
                **_local_kwargs(self.model_path),
            )
            _h = getattr(_cfg, "hidden_size", 0) or getattr(_cfg, "d_model", 1024)
            _n = getattr(_cfg, "num_hidden_layers", 0) or getattr(_cfg, "decoder_layers", 12)
            _i = getattr(_cfg, "intermediate_size", 0) or getattr(_cfg, "ffn_dim", 4 * _h)
            _est_params = (4 * _h * _h * _n + 3 * _h * _i * _n)
            _est_bytes = _est_params * 2
            single_gpu = _est_bytes < single_gpu_mem * 0.10
        except Exception:
            single_gpu = False

        if single_gpu or n_devs == 1:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map=None,
                attn_implementation="sdpa" if self.config.use_flash_attention else "eager",
                **_local_kwargs(self.model_path),
            )
            self.model = self.model.to(self.devices[0])
        else:
            max_memory = {}
            for i in range(n_devs):
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
                attn_implementation="sdpa" if self.config.use_flash_attention else "eager",
                **_local_kwargs(self.model_path),
            )

        self.model.eval()

        if self.use_torch_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("torch.compile applied to NLLB model (CUDA)")
            except Exception as e:
                logger.warning("torch.compile failed for NLLB: %s", e)

        self._loaded = True
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "NLLB CUDA model loaded in %.1fs: %.1fM params",
            time.monotonic() - load_start, n_params / 1e6
        )

    def warmup(self, batches: int = 10) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        if os.environ.get("TR_SKIP_WARMUP") == "1":
            logger.info("TR_SKIP_WARMUP=1 — skipping NLLB warmup")
            return

        logger.info("NLLB CUDA warmup (%d batches)...", batches)
        device = self.devices[0]
        warmup_bs = getattr(self, '_configured_batch_size', 1) or 1
        warmup_bs = max(warmup_bs, 1)
        base_texts = [
            "This is a warm-up sentence for translation benchmarking.",
            "Machine translation quality assessment requires careful evaluation.",
        ]
        warmup_texts = (base_texts * ((warmup_bs + 1) // 2))[:warmup_bs]

        enc = self.tokenizer(
            warmup_texts, return_tensors="pt",
            padding="max_length",
            truncation=True, max_length=self.max_input_tokens,
        ).to(device)

        gen_kwargs = self._generate_kwargs()
        ws = time.monotonic()

        if self._transfer_stream is not None:
            with torch.cuda.stream(self._transfer_stream):
                for _ in range(batches):
                    with torch.no_grad():
                        self.model.generate(**enc, **gen_kwargs)
            torch.cuda.current_stream().wait_stream(self._transfer_stream)
        else:
            for _ in range(batches):
                with torch.no_grad():
                    self.model.generate(**enc, **gen_kwargs)

        torch.cuda.synchronize()
        logger.info("NLLB CUDA warmup complete (%.1fs)", time.monotonic() - ws)

    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        device = self.devices[0]
        wall_start = time.monotonic()

        if self._transfer_stream is not None:
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

        torch.cuda.synchronize()
        wall_end = time.monotonic()
        total_wall_ms = (wall_end - wall_start) * 1000.0

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

        B = len(batch.raw_texts) if hasattr(batch, 'raw_texts') else 1
        generations: list[GenerationOutput] = []
        total_in = 0
        total_out = 0

        for i in range(B):
            out_ids = outputs[i].tolist()
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()

            src = batch.raw_texts[i] if hasattr(batch, 'raw_texts') and i < len(batch.raw_texts) else ""
            in_tok = (
                int(batch.attention_mask[i].sum().item())
                if hasattr(batch, 'attention_mask') and i < len(batch.attention_mask)
                else (len(batch.input_ids[i]) if hasattr(batch, 'input_ids') and i < len(batch.input_ids) else 0)
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

    def _generate_kwargs(self) -> dict:
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
        if self._transfer_stream is not None:
            try:
                self._transfer_stream.synchronize()
            except Exception:
                pass
            self._transfer_stream = None
            logger.debug("NLLBCUDABackend: transfer stream released")

    def encode_source(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        with torch.no_grad():
            encoder = self.model.get_encoder()
            encoder_out = encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )
        return encoder_out.last_hidden_state
