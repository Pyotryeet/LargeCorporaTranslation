"""SacreBLEU wrapper for reproducible BLEU scoring.

Provides ``compute_bleu()``, a thin wrapper around ``sacrebleu.corpus_bleu``
that handles version-dependent API differences (sacrebleu 2.x vs 3.x tokenize
parameter), normalizes references to the expected list-of-lists shape, guards
against empty inputs, and returns a dict with score, signature, and tokenizer.
"""

import logging
import sacrebleu

from benchmark.config.constants import SACREBLEU_TOKENIZER

logger = logging.getLogger(__name__)

# Detect sacrebleu version for API compatibility.
# sacrebleu >= 3.0 renamed the ``tokenize`` parameter (e.g. "none" → None).
# .score has been on the 0-100 scale through all observed versions (1.x–2.6.0).
# We sniff the major version at import time so compute_bleu() can adapt its
# API calls if sacrebleu ever changes the scale.
_sacrebleu_major: int = 2
try:
    _ver = sacrebleu.__version__
    _sacrebleu_major = int(_ver.split(".")[0])
except (AttributeError, ValueError, IndexError) as e:
    logger.warning(
        "Could not detect sacrebleu version from %r (%s) — assuming v%s API",
        getattr(sacrebleu, '__version__', '<unknown>'), e, _sacrebleu_major,
    )


def compute_bleu(hypotheses: list[str], references: list[list[str]], tokenize: str = SACREBLEU_TOKENIZER) -> dict:
    """Compute sacreBLEU score for hypotheses against references.

    Args:
        hypotheses: List of candidate translation strings.
        references: List of reference translations. Each element may be a
            single string or a list of strings (multi-reference).
        tokenize: Tokenization strategy forwarded to sacrebleu. Defaults to
            the project-wide ``SACREBLEU_TOKENIZER`` constant. Empty string
            or ``"none"`` is mapped to ``None`` on sacrebleu >= 3.

    Returns:
        dict with keys:
            - ``bleu`` (float): BLEU score rounded to 1 decimal.
            - ``score`` (float): Same value, kept for backward compatibility.
            - ``signature`` (str): sacrebleu signature string describing the
              computation details.
            - ``tokenizer`` (str): The tokenizer name used.

        Returns ``{"bleu": 0.0, "score": 0.0}`` (without signature or
        tokenizer keys) when either input list is empty.

    Raises:
        Nothing explicitly; sacrebleu exceptions propagate as-is (e.g.
        ``EOFError`` on mismatched hypothesis/reference counts in certain
        versions).

    Side effects:
        Logs a warning at WARNING level for empty inputs.
        Logs the computed score at INFO level.
    """
    if not hypotheses or not references:
        logger.warning("Empty hypotheses or references for BLEU")
        return {"bleu": 0.0, "score": 0.0}
    refs = [[r] if isinstance(r, str) else r for r in references]
    if refs[0] and isinstance(refs[0][0], list):
        refs = refs[0]

    # sacrebleu 3.x changed the tokenize parameter: "none" → None, etc.
    if _sacrebleu_major >= 3:
        # Map tokenizer string to sacrebleu v3 API
        _tokenize_kwarg = None if tokenize in (None, "none", "") else tokenize
        try:
            result = sacrebleu.corpus_bleu(hypotheses, refs, tokenize=_tokenize_kwarg)
        except TypeError:
            # Fallback: try without the tokenize kwarg
            result = sacrebleu.corpus_bleu(hypotheses, refs)
    else:
        result = sacrebleu.corpus_bleu(hypotheses, refs, tokenize=tokenize)

    logger.info("BLEU: %.1f (tokenizer=%s)", result.score, tokenize)
    return {"bleu": round(result.score, 1), "score": round(result.score, 1),
            "signature": str(result), "tokenizer": tokenize}
