"""chrF++ wrapper — character n-gram F-score with word n-grams."""

import logging
import sacrebleu

logger = logging.getLogger(__name__)


def compute_chrf(hypotheses: list[str], references: list[list[str]], word_order: int = 2) -> dict:
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
