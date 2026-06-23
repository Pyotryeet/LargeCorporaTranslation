"""Token-level confidence estimation for translation quality (Phase 6).

Computes sequence-level confidence scores from token log-probabilities
during greedy decoding.  Low-confidence translations are flagged for
re-translation with beam search or human review.

The confidence metric is:
    confidence = exp(mean(log_prob) / max(1, length_penalty))

where length_penalty = ((5 + N) / 6) ^ alpha  (standard BLEU-style penalty).
This penalizes very short outputs while allowing longer ones.
"""

import logging
import math
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

# ── Length-penalty constants (BLEU-style brevity penalty) ─────────────
# Reference length N is smoothed toward 5 and further smoothed by 6
# to avoid extreme penalties on very short sequences.
LP_REF_LENGTH = 5.0
LP_SMOOTHING = 6.0
CONFIDENCE_CLAMP_MAX = 1.0   # confidence ceiling — values above 1.0 are clamped
CONFIDENCE_LP_FLOOR = 1.0    # length-penalty minimum — prevents division by < 1.0


@dataclass
class TokenConfidence:
    """Per-token confidence information."""

    token_id: int
    token_text: str
    log_prob: float
    probability: float
    rank: int  # 0 = top choice (greedy), >0 = lower-ranked


@dataclass
class SequenceConfidence:
    """Confidence assessment for a complete translation."""

    tokens: list[TokenConfidence] = field(default_factory=list)
    mean_log_prob: float = 0.0
    mean_probability: float = 0.0
    sequence_confidence: float = 0.0
    length_penalty: float = 1.0
    is_low_confidence: bool = False
    low_confidence_tokens: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sequence_confidence": round(self.sequence_confidence, 4),
            "mean_log_prob": round(self.mean_log_prob, 4),
            "mean_probability": round(self.mean_probability, 4),
            "num_tokens": len(self.tokens),
            "is_low_confidence": self.is_low_confidence,
            "low_confidence_positions": self.low_confidence_tokens,
        }


class ConfidenceEstimator:
    """Estimates translation confidence from token log-probabilities.

    Usage
    -----
    >>> estimator = ConfidenceEstimator(threshold=0.7)
    >>> # During generation, collect token log-probs via model outputs.logits
    >>> conf = estimator.evaluate(token_log_probs, token_ids, tokenizer)
    >>> if conf.is_low_confidence:
    ...     flag_for_review(chunk)
    """

    def __init__(
        self,
        threshold: float = 0.65,      # below this → low confidence
        token_threshold: float = 0.1,  # individual token prob below this → flagged
        length_alpha: float = 0.6,    # BLEU-style length penalty exponent
        min_tokens: int = 3,
        seed: int | None = None,        # seed for reproducible pseudo-random draws;
                                        # set to None for non-deterministic sampling.
                                        # NOTE: Currently not consumed by any method
                                        # in this class.  External callers that need
                                        # reproducibility should set torch.manual_seed()
                                        # and numpy.random.seed() directly.
    ):
        self.threshold = threshold
        self.token_threshold = token_threshold
        self.length_alpha = length_alpha
        self.min_tokens = min_tokens
        self.seed = seed

    def evaluate(
        self,
        log_probs: list[float],       # per-token log-probabilities
        token_ids: list[int],          # generated token IDs
        tokenizer,                     # for decoding token text
    ) -> SequenceConfidence:
        """Compute confidence for a generated sequence.

        Parameters
        ----------
        log_probs : list[float]
            Log-probability of each generated token under the model.
            Typically extracted from ``outputs.logits`` after generation.
        token_ids : list[int]
            The generated token IDs.
        tokenizer :
            Tokenizer for decoding token text.

        Returns
        -------
        SequenceConfidence
        """
        n = len(log_probs)
        if n < self.min_tokens:
            return SequenceConfidence(
                is_low_confidence=True,
                sequence_confidence=0.0,
            )

        # ── Per-token analysis ──
        tokens = []
        low_positions = []
        for i, (lp, tid) in enumerate(zip(log_probs, token_ids)):
            prob = math.exp(lp)
            token_text = tokenizer.decode([tid]) if tokenizer else str(tid)
            is_low = prob < self.token_threshold

            tokens.append(TokenConfidence(
                token_id=tid,
                token_text=token_text,
                log_prob=lp,
                probability=prob,
                rank=0,  # greedy → always rank 0
            ))

            if is_low:
                low_positions.append(i)

        # ── Sequence-level aggregation ──
        mean_lp = sum(log_probs) / n
        mean_prob = math.exp(mean_lp)

        # Length penalty (shorter = penalized more).
        lp_val = ((LP_REF_LENGTH + n) / LP_SMOOTHING) ** self.length_alpha

        # Sequence confidence: exp(mean(log_prob) / length_penalty).
        seq_confidence = math.exp(mean_lp / max(lp_val, CONFIDENCE_LP_FLOOR))
        seq_confidence = min(seq_confidence, CONFIDENCE_CLAMP_MAX)  # clamp confidence ceiling to (0.0, 1.0]

        return SequenceConfidence(
            tokens=tokens,
            mean_log_prob=mean_lp,
            mean_probability=mean_prob,
            sequence_confidence=seq_confidence,
            length_penalty=lp_val,
            is_low_confidence=seq_confidence < self.threshold,
            low_confidence_tokens=low_positions,
        )

    def evaluate_from_logits(
        self,
        logits: torch.Tensor,  # [seq_len, vocab_size] or [1, seq_len, vocab_size]
        token_ids: torch.Tensor, # [seq_len]
        tokenizer,
    ) -> SequenceConfidence:
        """Convenience: evaluate from raw logits tensor.

        Extracts the log-probability of each generated token from the
        logits output by the model.
        """
        if logits.dim() == 3:
            logits = logits.squeeze(0)

        log_probs_raw = torch.nn.functional.log_softmax(logits, dim=-1)
        token_ids_flat = token_ids.squeeze()

        # Warn when logits sequence length differs from token_ids length —
        # this usually indicates a generation vs. logits misalignment.
        if log_probs_raw.shape[0] != len(token_ids_flat):
            logger.warning(
                "evaluate_from_logits: logits seq_len=%d != token_ids len=%d — "
                "truncating to min length",
                log_probs_raw.shape[0], len(token_ids_flat),
            )

        # Extract log_prob for the chosen token at each position.
        log_probs = []
        for i in range(min(log_probs_raw.shape[0], len(token_ids_flat))):
            tid = token_ids_flat[i].item()
            lp = log_probs_raw[i, tid].item()
            log_probs.append(lp)

        return self.evaluate(log_probs, token_ids_flat.tolist(), tokenizer)

    @staticmethod
    def batch_confidence_summary(
        results: list[SequenceConfidence],
    ) -> dict:
        """Summarize confidence across a batch."""
        confidences = [r.sequence_confidence for r in results if r.sequence_confidence > 0]
        low_count = sum(1 for r in results if r.is_low_confidence)

        if not confidences:
            return {"count": 0, "low_confidence_rate": 0.0}

        return {
            "count": len(confidences),
            "mean_confidence": round(sum(confidences) / len(confidences), 4),
            "min_confidence": round(min(confidences), 4),
            "max_confidence": round(max(confidences), 4),
            "low_confidence_count": low_count,
            "low_confidence_rate": round(low_count / len(results), 4) if results else 0.0,
        }
