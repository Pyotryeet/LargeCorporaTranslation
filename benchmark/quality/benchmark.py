"""Quality benchmark orchestrator — model-agnostic translation + metrics (v3.0).

Uses the ``InferenceBackend`` protocol so quality evaluation works identically
for autoregressive, diffusion, and custom model backends.

v3.0 changes:
- No direct ``engine.model.generate()`` calls — uses ``engine.backend.translate_batch()``.
- The backend protocol ensures translation, confidence, and scoring work
  regardless of the underlying model architecture.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Methodology limitations for academic publication:
# - Single reference per source (research standard is 3+ references)
# - No bootstrap confidence intervals on quality scores
# - No paired significance testing between runs
# - 10-pair golden reference set is statistically undersized (standard is 500-2000)
# - COMET-22 human correlation is moderate (tau=0.48), not strong
# These are known gaps and should be addressed before publishing results.
# ═══════════════════════════════════════════════════════════════════════════════

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import PreTrainedTokenizerBase
from benchmark.config.constants import (
    DEFAULT_TRUNCATION_LENGTH,
    END_OF_TURN_TOKEN_ID,
    METRICS_PARALLEL_WORKERS,
    QUALITY_BATCH_SIZE,
    QUALITY_BLEU_TARGET,
    QUALITY_CHRF_TARGET,
    QUALITY_COMET_TARGET,
    QUALITY_MAX_NEW_TOKENS,
)
from benchmark.quality.references import ReferenceLoader
from benchmark.quality.metrics_bleu import compute_bleu
from benchmark.quality.metrics_chrf import compute_chrf
from benchmark.quality.metrics_comet import compute_comet

logger = logging.getLogger(__name__)

# ── Aliases kept for backward compatibility with existing internal references ──
DEFAULT_BATCH_SIZE = QUALITY_BATCH_SIZE
DEFAULT_MAX_LENGTH = DEFAULT_TRUNCATION_LENGTH
MAX_METRIC_WORKERS = METRICS_PARALLEL_WORKERS
BLEU_TARGET_MIN = QUALITY_BLEU_TARGET
CHRF_TARGET_MIN = QUALITY_CHRF_TARGET
COMET_TARGET_MIN = QUALITY_COMET_TARGET


@dataclass
class QualityResults:
    comet: dict = field(default_factory=dict)
    comet_kiwi: dict = field(default_factory=dict)
    bertscore: dict = field(default_factory=dict)
    num_references: int = 0
    num_translated: int = 0
    duration_seconds: float = 0.0
    backend_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"comet": self.comet,
                 "comet_kiwi": self.comet_kiwi, "bertscore": self.bertscore,
                 "num_references": self.num_references,
                 "num_translated": self.num_translated,
                 "duration_seconds": round(self.duration_seconds, 1),
                 "backend_info": self.backend_info}

    @property
    def scores_meet_targets(self) -> bool:
        # BERTScore (reference-free neural metric) is the primary quality gate.
        # >= 0.55 indicates the translation preserves meaning acceptably.
        # >= 0.70 is strong (competitive with commercial systems).
        bs = self.bertscore.get("system_score", 0) or 0.0
        return bs >= 0.55


def _build_batch(
    sources: list[str],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    engine=None,  # InferenceEngine — needed for backend-aware prompting
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Tokenize source sentences for quality benchmark translation.

    For **autoregressive / instruction-tuned** models (TranslateGemma, Gemma 4,
    Ministral etc.): wraps source text in the model's chat template so the
    model knows it's a translation task.

    For **encoder-decoder** models (NLLB, T5 etc.): prefixes with the NLLB
    language token and uses raw tokenization — no chat template needed.
    """

    # ── Encoder-decoder path: NLLB-style forced-language prompting ──
    try:
        mt = engine.model_type if engine is not None else None
    except Exception:
        mt = None
    if mt in ("encoder_decoder", "encoder-decoder"):
        # Build simple "translate: source → target" prefixes.
        # NLLB uses forced decoder language tokens set in generate(),
        # so just tokenize source as-is with a translation prefix.
        prompted_sources: list[str] = []
        for text in sources:
            prompted_sources.append(f"English: {text}\nTurkish:")
        enc = tokenizer(
            prompted_sources,
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_LENGTH,
            return_tensors="pt",
            return_length=True,
        )
        input_ids: torch.Tensor = enc["input_ids"].to(device)
        attention_mask: torch.Tensor = enc["attention_mask"].to(device)
        lengths: list[int] = enc["length"].tolist()
        return input_ids, attention_mask, lengths

    # ── Autoregressive path: chat-template wrapping ──
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        prompted_sources: list[str] = []
        for text in sources:
            prompt = None
            try:
                # Try TranslateGemma-style structured content with lang codes.
                msgs = [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "tr",
                        "text": text,
                    }],
                }]
                result = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                prompt = result if isinstance(result, str) else "".join(result)
            except Exception:
                pass

            if prompt is None:
                try:
                    # Fall back to plain-text prompt (works with SmolLM, LLaMA, etc.).
                    fallback_msgs = [{
                        "role": "user",
                        "content": f"Translate the following English text to Turkish. "
                                   f"Output only the translation, nothing else:\n\n{text}",
                    }]
                    result = tokenizer.apply_chat_template(
                        fallback_msgs, tokenize=False, add_generation_prompt=True,
                    )
                    prompt = result if isinstance(result, str) else "".join(result)
                except Exception:
                    pass

            if prompt is None:
                # Last resort: no template wrapping.
                prompt = f"Translate English to Turkish:\n{text}"

            prompted_sources.append(prompt)
    else:
        # No chat template — use a simple EN→TR prefix.
        prompted_sources = [
            f"Translate English to Turkish:\n{t}" for t in sources
        ]

    enc = tokenizer(
        prompted_sources,
        padding=True,
        truncation=True,
        max_length=DEFAULT_MAX_LENGTH,
        return_tensors="pt",
        return_length=True,
    )
    input_ids: torch.Tensor = enc["input_ids"].to(device)
    attention_mask: torch.Tensor = enc["attention_mask"].to(device)
    lengths: list[int] = enc["length"].tolist()
    return input_ids, attention_mask, lengths


def _compute_metrics_parallel(
    hypotheses: list[str],
    references: list[str],
    sources: list[str],
) -> tuple[dict, dict, dict, dict]:
    with ThreadPoolExecutor(max_workers=MAX_METRIC_WORKERS) as pool:
        # Neural quality metrics — reference-based (COMET-22) and
        # reference-free (BERTScore, COMET-Kiwi).  N-gram metrics (BLEU, chrF++)
        # are omitted because they penalise legitimate wording variations and
        # fail to capture semantic quality for morphologically rich EN→TR.
        future_comet = pool.submit(compute_comet, sources, hypotheses, references)
        from benchmark.quality.metrics_comet import compute_comet_kiwi
        from benchmark.quality.metrics_bertscore import compute_bertscore
        future_comet_kiwi = pool.submit(compute_comet_kiwi, sources, hypotheses)
        future_bertscore = pool.submit(compute_bertscore, sources, hypotheses)
        comet = future_comet.result()
        comet_kiwi = future_comet_kiwi.result()
        bertscore = future_bertscore.result()
    return comet, comet_kiwi, bertscore


class QualityBenchmark:
    """Model-agnostic quality benchmark — works with any InferenceBackend."""

    def __init__(self, reference_path: str):
        self.reference_path = reference_path

    def run(self, engine, *, max_references: Optional[int] = None) -> QualityResults:
        """Run quality benchmark using the engine's backend.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine (v3.0 — delegates to backend protocol).
        max_references : Optional[int]
            Cap the number of reference sentences translated.  When None
            (default), all references are used.  Set to a small value
            (e.g. 32) for smoke-test / dry-run mode.

        Returns
        -------
        QualityResults
        """
        logger.info("Starting quality benchmark (backend=%s)...",
                    engine.display_name)
        start = time.monotonic()
        loader = ReferenceLoader(self.reference_path)
        sources, references = loader.load()

        # Truncate for smoke-test / dry-run mode.
        if max_references is not None and max_references < len(sources):
            logger.info(
                "Limiting quality benchmark to %d of %d reference sentences "
                "(max_references=%d)",
                max_references, len(sources), max_references,
            )
            sources = sources[:max_references]
            references = references[:max_references]

        n = len(sources)
        logger.info("Translating %d reference sentences in batches…", n)

        device = engine.devices[0]
        tokenizer = engine.tokenizer
        # Reference sentences average ~42 chars (≈10-20 tokens).  Using the
        # production max_new_tokens (512) would generate 20-50× more than needed.
        max_new = QUALITY_MAX_NEW_TOKENS
        do_sample = engine.decoding_params.do_sample
        num_beams = engine.decoding_params.num_beams
        temperature = engine.decoding_params.temperature

        hypotheses: list[str] = [""] * n
        bs = DEFAULT_BATCH_SIZE

        # Minimal PipelineBatch-compatible object for backend protocol.
        class _MiniBatch:
            pass

        for batch_idx in range(0, n, bs):
            end = min(batch_idx + bs, n)
            batch_sources = sources[batch_idx:end]
            batch_size_actual = len(batch_sources)

            t0 = time.monotonic()
            input_ids, attention_mask, _ = _build_batch(batch_sources, tokenizer, device, engine=engine)

            # ── v3.0: Use backend protocol instead of model.generate() ──
            if hasattr(engine, '_backend') and engine._backend is not None:
                mini_batch = _MiniBatch()
                mini_batch.input_ids = input_ids
                mini_batch.attention_mask = attention_mask
                mini_batch.raw_texts = batch_sources
                mini_batch.batch_id = batch_idx // bs

                try:
                    result = engine._backend.translate_batch(mini_batch)
                except Exception as e:
                    logger.warning(
                        "translate_batch failed for batch %d: %s — skipping",
                        batch_idx // bs, e,
                    )
                    continue

                for i, gen in enumerate(result.generations):
                    hypotheses[batch_idx + i] = gen.translated_text
            else:
                # Legacy fallback: direct model.generate().
                try:
                    with torch.no_grad():
                        gen_kwargs = dict(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new,
                            do_sample=do_sample,
                            num_beams=num_beams,
                            pad_token_id=tokenizer.pad_token_id or 0,
                            eos_token_id=[tokenizer.eos_token_id, END_OF_TURN_TOKEN_ID],
                            use_cache=True,
                        )
                        if do_sample:
                            gen_kwargs["temperature"] = temperature
                        outputs = engine.model.generate(**gen_kwargs)

                    for i, out_ids in enumerate(outputs):
                        in_len = len(input_ids[i])
                        new_toks = out_ids[in_len:]
                        hypotheses[batch_idx + i] = tokenizer.decode(
                            new_toks, skip_special_tokens=True,
                        ).strip()
                except Exception as e:
                    logger.warning(
                        "model.generate() failed for batch %d: %s — skipping",
                        batch_idx // bs, e,
                    )
                    continue

            elapsed_s = time.monotonic() - t0
            logger.info(
                "  batch [%4d:%4d] (%d sentences) — %.1fs",
                batch_idx, end, batch_size_actual, elapsed_s,
            )

        translate_duration = time.monotonic() - start
        logger.info(
            "Batched translation done in %.1f s (%d sentences)",
            translate_duration, n,
        )

        # ── Parallel metrics ──
        logger.info("Computing quality metrics in parallel (COMET | COMET-Kiwi | BERTScore)...")
        metric_start = time.monotonic()
        comet, comet_kiwi, bertscore = _compute_metrics_parallel(hypotheses, references, sources)
        metric_duration = time.monotonic() - metric_start
        logger.info("All metrics computed in %.1f s (parallel)", metric_duration)

        duration = time.monotonic() - start
        backend_info = {}
        if hasattr(engine, 'get_backend_info'):
            backend_info = engine.get_backend_info()

        results = QualityResults(
            comet=comet, comet_kiwi=comet_kiwi, bertscore=bertscore,
            num_references=len(references),
            num_translated=len([h for h in hypotheses if h]),
            duration_seconds=duration,
            backend_info=backend_info,
        )
        logger.info("Quality benchmark complete in %.1fs", duration)
        logger.info(
            "  BERTScore: %s, COMET-22: %s, COMET-Kiwi: %s",
            bertscore.get('system_score', 'N/A'),
            comet.get('system_score', 'N/A'),
            comet_kiwi.get('system_score', 'N/A'),
        )
        return results
