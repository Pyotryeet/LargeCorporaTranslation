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
    """Hyperparameters for text generation / decoding.

    This dataclass bundles all generation knobs (beam search, sampling
    temperature, top-p, top-k, repetition penalty) into a single validated
    object.  Construction-time validation (in ``__post_init__``) rejects
    illegal combinations such as ``do_sample=True`` with ``temperature=0.0``.

    Fields
    ------
    max_new_tokens : int
        Maximum number of tokens to generate (default 512).
    temperature : float
        Sampling temperature; 0.0 means greedy decoding (default 0.0).
    do_sample : bool
        Enable multinomial sampling (default False).
    num_beams : int
        Number of beams for beam search; 1 disables beam search (default 1).
    top_p : float
        Nucleus sampling probability threshold (default 1.0 = disabled).
    top_k : int
        Top-k sampling cutoff; 0 disables top-k filtering (default 0).
    repetition_penalty : float
        Repetition penalty factor; 1.0 disables (default 1.0).

    Raises
    ------
    ValueError
        If any field violates its permitted range or if incompatible
        combinations are passed (e.g. sampling with zero temperature).
    """
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
            If temperature, top_p, num_beams, or max_new_tokens are invalid.
        """
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature}.  "
                f"Set temperature=0.0 for greedy decoding."
            )
        if self.top_p < 0:
            raise ValueError(f"top_p must be >= 0, got {self.top_p}.  Set top_p=1.0 to disable.")
        if self.temperature == 0.0 and self.do_sample:
            raise ValueError(
                "do_sample=True with temperature=0.0 is invalid: "
                "sampling requires temperature > 0."
            )
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {self.num_beams}.")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}.")

    @property
    def temperature_is_zero(self) -> bool:
        """True when temperature is exactly 0.0 (greedy decoding)."""
        return self.temperature == 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a dict representation of decoding parameters.

        Sampling-specific keys (``top_p``, ``top_k``, ``repetition_penalty``)
        are only included when ``do_sample`` is ``True``, to avoid triggering
        spurious HuggingFace warnings about unused generation flags.

        Returns
        -------
        dict[str, object]
            Dict with keys ``"max_new_tokens"``, ``"temperature"``,
            ``"do_sample"``, ``"num_beams"``, and conditionally ``"top_p"``,
            ``"top_k"``, ``"repetition_penalty"``.
        """
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
