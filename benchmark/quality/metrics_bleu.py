"""SacreBLEU wrapper for reproducible BLEU scoring."""

import logging
import sacrebleu

from benchmark.config.constants import SACREBLEU_TOKENIZER

logger = logging.getLogger(__name__)

# Detect sacrebleu version for API compatibility.
# sacrebleu >= 3.0 renamed the ``tokenize`` parameter (e.g. "none" → None)
# and changed the output .score scale.  We sniff the major version once at
# import time so that compute_bleu() can adapt its API calls.
_sacrebleu_major: int = 2
try:
    _ver = sacrebleu.__version__
    _sacrebleu_major = int(_ver.split(".")[0])
except (AttributeError, ValueError, IndexError) as e:
    logger.warning(
        "Could not detect sacrebleu version from %r (%s) — assuming v%s API",
        getattr(sacrebleu, '__version__', '<unknown>'), e, _sacrebleu_major,
    )

# sacrebleu >= 2.0.0 changed .score from 0-100 to 0-1 scale.
# sacrebleu > 2.0.0 changed the tokenize API (string → string|None).
_sacrebleu_score_is_hundred_scale = _sacrebleu_major < 2


def compute_bleu(hypotheses: list[str], references: list[list[str]], tokenize: str = SACREBLEU_TOKENIZER) -> dict:
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

    logger.info("BLEU: %.1f (scale=%s, tokenizer=%s)",
                result.score, "0-100" if _sacrebleu_score_is_hundred_scale else "0-1", tokenize)
    return {"bleu": round(result.score, 1), "score": round(result.score, 1),
            "signature": str(result), "tokenizer": tokenize,
            "scale": "0-100" if _sacrebleu_score_is_hundred_scale else "0-1"}
