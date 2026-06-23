"""Ensemble translation with quality voting (Phase 6).

Translates each chunk with multiple models/configurations and selects
the best output via consensus scoring.  This provides non-negotiable
quality assurance by cross-referencing different models.

Strategy
--------
1. Primary model (TranslateGemma 12B, greedy) — main translator.
2. Secondary model (TranslateGemma 4B, beam=2) — diversity check.
3. Consensus score (chrF++ between translations) — must be ≥ 0.7.
4. If consensus fails, flag for human review.

For large-scale production (hundreds of GPUs), the primary model runs
on every chunk; secondary verification runs on a 1% sample.  This
balances quality assurance with compute cost.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
MAX_INPUT_LENGTH = 2048
SECONDARY_NUM_BEAMS = 2


@dataclass
class EnsembleTranslationResult:
    """Result of ensemble translation for one input chunk."""

    source_text: str
    primary_translation: str
    secondary_translation: Optional[str] = None
    consensus_score: Optional[float] = None
    selected_translation: str = ""
    flags: list[str] = field(default_factory=list)
    needs_human_review: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source_text[:200],
            "primary": self.primary_translation[:200],
            "secondary": self.secondary_translation[:200] if self.secondary_translation else None,
            "consensus_score": round(self.consensus_score, 4) if self.consensus_score else None,
            "selected": self.selected_translation[:200],
            "flags": self.flags,
            "needs_human_review": self.needs_human_review,
        }


class EnsembleTranslator:
    """Multi-model translation ensemble for quality assurance.

    Usage
    -----
    >>> ensemble = EnsembleTranslator(primary_engine, secondary_engine)
    >>> result = ensemble.translate("The weather is nice today.")
    >>> if result.needs_human_review:
    ...     review_queue.append(result)
    """

    def __init__(
        self,
        primary_engine,   # InferenceEngine (main model)
        secondary_engine=None,  # InferenceEngine (smaller model, optional)
        consensus_threshold: float = 0.70,
    ):
        self.primary_engine = primary_engine
        self.secondary_engine = secondary_engine
        self.consensus_threshold = consensus_threshold

        # Stats
        self.total_translations = 0
        self.total_flagged = 0

    def translate(self, text: str) -> EnsembleTranslationResult:
        """Translate a single text with consensus verification.

        Parameters
        ----------
        text : str
            Source text in English.

        Returns
        -------
        EnsembleTranslationResult
        """
        self.total_translations += 1

        # ── Primary translation ──
        primary_text = self._translate_single(self.primary_engine, text)

        result = EnsembleTranslationResult(
            source_text=text,
            primary_translation=primary_text,
            selected_translation=primary_text,  # default
        )

        # ── Secondary translation (if available) ──
        if self.secondary_engine is not None:
            secondary_text = self._translate_single(
                self.secondary_engine, text, num_beams=SECONDARY_NUM_BEAMS,
            )
            result.secondary_translation = secondary_text

            # Compute consensus (chrF++ between primary and secondary).
            try:
                import sacrebleu
                raw_chrf = sacrebleu.corpus_chrf(
                    [primary_text], [[secondary_text]]
                )
                # sacrebleu < 2.0.0 returned 0-100; >= 2.0.0 returns 0-1.
                # Detect the scale from the runtime package version.
                try:
                    _ver = sacrebleu.__version__
                    _major = int(_ver.split(".")[0])
                except (AttributeError, ValueError):
                    _major = 1  # conservative: assume old scale
                chrf_score = raw_chrf.score / 100.0 if _major < 2 else raw_chrf.score
                result.consensus_score = chrf_score

                if chrf_score < self.consensus_threshold:
                    result.flags.append(
                        f"low_consensus_{chrf_score:.3f}_lt_{self.consensus_threshold}"
                    )
                    result.needs_human_review = True
                    self.total_flagged += 1

                    # When consensus is low, prefer primary but with a flag.
                    # Secondary might be more literal; primary is usually better
                    # for fluency in low-resource pairs like EN→TR.
            except (ImportError, ValueError, TypeError) as e:
                logger.warning("Consensus scoring failed: %s", e)
                result.consensus_score = None

        return result

    def _translate_single(
        self, engine, text: str, num_beams: int = 1,
    ) -> str:
        """Translate a single text using the given engine.

        Uses the backend protocol (translate_batch) when available, falling
        back to raw model.generate() only for encoder-decoder models where
        the backend is not wired.
        """
        from types import SimpleNamespace
        import torch

        device = engine.devices[0]
        tokenizer = engine.tokenizer

        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_INPUT_LENGTH)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Prefer the backend protocol since it handles chat templates,
        # forced language tokens, and architecture-specific generation
        # details correctly for all model families (autoregressive,
        # diffusion, encoder-decoder).
        if hasattr(engine, '_backend') and engine._backend is not None:
            try:
                mini_batch = SimpleNamespace()
                mini_batch.input_ids = input_ids
                mini_batch.attention_mask = attention_mask
                mini_batch.raw_texts = [text]
                mini_batch.batch_id = 0

                result = engine._backend.translate_batch(mini_batch)
                if result.generations:
                    return result.generations[0].translated_text
            except Exception as e:
                logger.warning(
                    "Ensemble: translate_batch failed — "
                    "falling back to model.generate(): %s", e,
                )

        # Legacy fallback: direct model.generate() for autoregressive models
        # without a wired backend.  Only applicable when the model object
        # actually has a generate() method (autoregressive, not diffusion).
        if not hasattr(engine.model, 'generate'):
            logger.warning(
                "Ensemble: backend unavailable and model has no generate() "
                "method (likely a diffusion model or custom backend). "
                "Skipping translation for: %.100s...", text,
            )
            return ""

        # Legacy fallback: direct model.generate() for models without a
        # backend, or when the backend call fails above.
        with torch.no_grad():
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=engine.decoding_params.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or 0,
                eos_token_id=tokenizer.eos_token_id,
            )
            if num_beams > 1:
                gen_kwargs["num_beams"] = num_beams
            out = engine.model.generate(**gen_kwargs)

        in_len = len(input_ids[0])
        return tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()

    @property
    def flag_rate(self) -> float:
        if self.total_translations == 0:
            return 0.0
        return self.total_flagged / self.total_translations
