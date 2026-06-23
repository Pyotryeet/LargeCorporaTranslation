"""SacreBLEU wrapper for reproducible BLEU scoring."""

import logging
import sacrebleu

from benchmark.config.constants import SACREBLEU_TOKENIZER

logger = logging.getLogger(__name__)


def compute_bleu(hypotheses: list[str], references: list[list[str]], tokenize: str = SACREBLEU_TOKENIZER) -> dict:
    if not hypotheses or not references:
        logger.warning("Empty hypotheses or references for BLEU")
        return {"bleu": 0.0, "score": 0.0}
    refs = [[r] if isinstance(r, str) else r for r in references]
    if refs[0] and isinstance(refs[0][0], list):
        refs = refs[0]
    result = sacrebleu.corpus_bleu(hypotheses, refs, tokenize=tokenize)
    logger.info("BLEU: %.1f (tokenizer=%s)", result.score, tokenize)
    return {"bleu": round(result.score, 1), "score": round(result.score, 1),
            "signature": str(result), "tokenizer": tokenize}
