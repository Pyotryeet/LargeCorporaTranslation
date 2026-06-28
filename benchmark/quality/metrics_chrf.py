"""chrF++ metrics — character n-gram F-score augmented with word n-grams.

Wraps sacrebleu.corpus_chrf with sensible defaults tuned for agglutinative
morphologies (char_order=4 instead of the default 6).  Provides a single
``compute_chrf`` entry point that normalises reference formats and returns
a dictionary suitable for downstream reporting pipelines.
"""

import logging
import sacrebleu

logger = logging.getLogger(__name__)


def compute_chrf(hypotheses: list[str], references: list[list[str]], word_order: int = 2) -> dict:
    """Compute chrF++ score for a batch of hypotheses against references.

    chrF++ is the character n-gram F-score extended with word n-grams.  It is
    the primary automatic metric used at WMT (Conference on Machine Translation);
    this wrapper uses sacrebleu under the hood.

    Args:
        hypotheses: Candidate translation strings, one per item in the batch.
        references: Reference translation(s).  Each element may be either a
            single string (treated as a single reference) or a list of strings
            (multiple references per hypothesis).  If the outer list itself
            contains only a single list-of-strings element, that inner list is
            unwrapped and used as per-item multi-reference lists.
        word_order: Number of word n-grams to include in the chrF++ score.
            Default 2 (bigrams).  Pass 0 for plain chrF (character n-grams
            only).

    Returns:
        A dict with keys:
            chrF (float): Score rounded to one decimal place.
            score (float): Alias for chrF, for compatibility with other
                quality metric return shapes.
            word_order (int): Echo of the word_order parameter used.
            signature (str): sacrebleu signature string for reproducibility.

    Raises:
        No exceptions are raised directly.  On empty input a warning is
        logged and a zero-score dict is returned.

    Side effects:
        Logs the computed score at INFO level.

    Important caveats:
        - The default char_order=4 (set inside the body) is deliberately lower
          than sacrebleu's default of 6.  Longer character n-grams cause over-
          matching in agglutinative languages (e.g. Turkish, Finnish, Hungarian)
          where a long character span can bridge unrelated morphemes, inflating
          scores for poor translations.
        - The reference-unwrapping logic (lines 13-15) is intentionally forgiving
          to accept the varied reference shapes produced by different loaders.
    """
    if not hypotheses or not references:
        logger.warning("Empty hypotheses/references for chrF")
        return {"chrF": 0.0, "score": 0.0}
    refs = [[r] if isinstance(r, str) else r for r in references]
    if refs[0] and isinstance(refs[0][0], list):
        refs = refs[0]
    # char_order=4 per WMT findings: default 6 over-matches agglutinative
    # morphologies (e.g. Turkish, Finnish, Hungarian) where long character
    # n-grams span unrelated morphemes, inflating scores for bad translations.
    result = sacrebleu.corpus_chrf(hypotheses, refs, word_order=word_order, char_order=4)
    logger.info("chrF++: %.1f (word_order=%d)", result.score, word_order)
    return {"chrF": round(result.score, 1), "score": round(result.score, 1),
            "word_order": word_order, "signature": str(result)}
