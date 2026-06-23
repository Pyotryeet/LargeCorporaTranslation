"""Decoding parameters for translation."""

from dataclasses import dataclass

# ── Default values for DecodingParams (magic numbers extracted) ────────
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_DO_SAMPLE = False
DEFAULT_NUM_BEAMS = 1
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_REPETITION_PENALTY = 1.0


@dataclass
class DecodingParams:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    do_sample: bool = DEFAULT_DO_SAMPLE
    num_beams: int = DEFAULT_NUM_BEAMS
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY

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
