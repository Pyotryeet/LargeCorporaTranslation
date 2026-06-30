"""NLLB MPS (Apple Silicon Metal) optimized translation backend (v3.7).

Facebook's NLLB-200 model family optimized specifically for Apple Silicon devices.
Uses direct-to-Metal weight loading and skips compilation warmup loops to preserve VRAM.
"""

from __future__ import annotations

import logging
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


class NLLBMPSBackend(InferenceBackend):
    """Apple Silicon MPS optimized backend for NLLB models."""

    model_type = ModelType.ENCODER_DECODER
    capabilities = (
        ModelCapability.TRANSLATE | ModelCapability.FORWARD_ENCODE
        | ModelCapability.ENSEMBLE_READY
    )
    display_name = "NLLB Encoder-Decoder (MPS Optimized)"

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.model_path = config.model_path
        self.tokenizer_path = config.tokenizer_path or config.model_path
        self.max_input_tokens = config.max_input_tokens
        self.max_new_tokens = config.max_new_tokens
        self.temperature = config.temperature

        extra = config.extra
        self.do_sample = extra.get("do_sample", False)
        self.use_flash_attention = config.use_flash_attention
        self.use_torch_compile = config.use_torch_compile
        self.num_beams = extra.get("num_beams", 1)
        self.src_lang = extra.get("nllb_source_lang", "eng_Latn")
        self.tgt_lang = extra.get("nllb_target_lang", "tur_Latn")
        self.precision_config = None

    def load(self) -> None:
        load_start = time.monotonic()
        logger.info("=== NLLB MPS: loading %s ===", self.model_path)

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

        self.devices = [torch.device("mps")]
        self.precision_config = get_precision_config(self.backend_name)
        dtype = self.precision_config.master_dtype

        _is_madlad = "madlad" in self.tokenizer_path.lower()
        _tok_kwargs: dict = {"trust_remote_code": False, **_local_kwargs(self.tokenizer_path)}
        if not _is_madlad:
            _tok_kwargs["src_lang"] = self.src_lang
            tokenizer_cls = AutoTokenizer
        else:
            tokenizer_cls = T5Tokenizer

        self.tokenizer = tokenizer_cls.from_pretrained(
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

        model_cls = T5ForConditionalGeneration if _is_madlad else AutoModelForSeq2SeqLM

        # T5ForConditionalGeneration/MADLAD does not support SDPA in transformers yet.
        # Always use eager attention for MADLAD to avoid ValueError.
        is_sdpa = self.config.use_flash_attention and not _is_madlad
        attn_impl = "sdpa" if is_sdpa else "eager"

        try:
            self.model = model_cls.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
                device_map={"": "mps"},
                attn_implementation=attn_impl,
                **_local_kwargs(self.model_path),
            )
        except (ImportError, ValueError, RuntimeError) as e:
            logger.warning("MPS device_map failed (%s) — falling back to CPU load", e)
            self.model = model_cls.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
                **_local_kwargs(self.model_path),
            )
            self.model = self.model.to(self.devices[0])

        if _is_madlad:
            self.model.shared.weight = self.model.decoder.embed_tokens.weight
            self.model.encoder.embed_tokens.weight = self.model.decoder.embed_tokens.weight

        self.model.eval()
        self._loaded = True
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "NLLB MPS model loaded in %.1fs: %.1fM params",
            time.monotonic() - load_start, n_params / 1e6
        )

    def warmup(self, batches: int = 1) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        logger.info("NLLB warmup: MPS — single compile step (skipping loop)")
        device = self.devices[0]
        gen_kwargs = self._generate_kwargs()
        enc = self.tokenizer(
            ["Warmup."], return_tensors="pt", padding=True,
            truncation=True, max_length=6,
        ).to(device)
        with torch.no_grad():
            self.model.generate(**enc, **gen_kwargs)

    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        device = self.devices[0]
        wall_start = time.monotonic()

        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)

        gen_kwargs = self._generate_kwargs()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

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
        logger.debug("NLLBMPSBackend: close called")

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
