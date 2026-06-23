"""Decoding parameters for translation.

Temperature safety
------------------
``temperature=0.0`` is passed through to HuggingFace ``model.generate()``,
which handles zero-temperature correctly (greedy decoding, no softmax
division).  Callers that apply ``/ temperature`` manually should guard
with ``max(temperature, EPSILON)`` where ``EPSILON = 1e-7``.
"""

from dataclasses import dataclass

# ── Default values for DecodingParams (magic numbers extracted) ────────
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_DO_SAMPLE = False
DEFAULT_NUM_BEAMS = 1
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_REPETITION_PENALTY = 1.0
TEMPERATURE_EPSILON = 1e-7  # guard for manual temp division callers


@dataclass
class DecodingParams:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    do_sample: bool = DEFAULT_DO_SAMPLE
    num_beams: int = DEFAULT_NUM_BEAMS
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY

    def __post_init__(self) -> None:
        """Validate decoding parameters on construction.

        Raises
        ------
        ValueError
            If temperature or top_p is negative.
        """
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature}.  "
                f"Set temperature=0.0 for greedy decoding."
            )
        if self.top_p < 0:
            raise ValueError(f"top_p must be >= 0, got {self.top_p}.  Set top_p=1.0 to disable.")
        # Guard: manual temperature division callers must avoid div-by-0.
        # HuggingFace model.generate() handles temperature=0.0 safely
        # (greedy decoding with no softmax division), but downstream code
        # that directly applies: logits = logits / max(temperature, EPSILON).
        # TEMPERATURE_EPSILON is exported for this pattern.
        if self.temperature == 0.0 and self.do_sample:
            logger = __import__('logging').getLogger(__name__)
            logger.warning(
                "do_sample=True with temperature=0.0 produces deterministic "
                "output (greedy decoding).  Set temperature > 0 for stochastic "
                "sampling."
            )

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature,
                                "do_sample": self.do_sample, "num_beams": self.num_beams}
        # Only include sampling params when do_sample=True to avoid HF warnings.
        if self.do_sample:
            d["top_p"] = self.top_p
            d["top_k"] = self.top_k
            d["repetition_penalty"] = self.repetition_penalty
        return d

    @property
    def generation_kwargs(self) -> dict[str, object]:
        """Return kwargs dict safe for model.generate() — no unused flags."""
        d: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
        }
        if self.do_sample:
            d["temperature"] = self.temperature
            d["top_p"] = self.top_p
            d["top_k"] = self.top_k
        return d
