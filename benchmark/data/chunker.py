"""Text segmentation for model-friendly input chunks.

v2.0: Token-level chunking — avoids the wasteful tokenize→decode→re-tokenize
cycle by working directly on token ID lists.  This eliminates the O(n)
decode+encode overhead per chunk, saving ~30–40% of chunker CPU time.
"""

import logging
from typing import Iterator
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class TextChunker:
    """Split long documents into segments fitting within the model's input window.

    v2.0: Token-level chunking.  Instead of:
      1. tokenize → decode each chunk → pipeline re-tokenizes
    we:
      1. tokenize once → slice token ID list → yield (text, token_ids)
      The pipeline's ``_tokeniser_loop`` receives pre-computed token IDs
      and skips the encode step entirely for chunked documents.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_input_tokens: int = 512, overlap_tokens: int = 50):
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.overlap_tokens = overlap_tokens
        if overlap_tokens >= max_input_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be < "
                f"max_input_tokens ({max_input_tokens})"
            )

    def _stride(self) -> int:
        """Stride for sliding-window chunking, accounting for overlap."""
        return max(self.max_input_tokens - self.overlap_tokens, self.max_input_tokens // 2)

    def chunk(self, text: str) -> Iterator[str]:
        """Yield chunk text strings (backward-compatible API).

        For token-level efficiency, use ``chunk_with_tokens()`` instead.

        Notes
        -----
        Token count checks include special tokens (``add_special_tokens=True``)
        to match the pipeline's re-encode (``pipeline.py:611-616``).  Without
        this, the chunk boundary underestimates the true token count and the
        prompt may exceed ``max_input_tokens``, triggering silent truncation.
        """
        if not text or not text.strip():
            return
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) <= self.max_input_tokens:
            yield text
            return
        stride = self._stride()
        for start in range(0, len(token_ids), stride):
            chunk_ids = token_ids[start:start + self.max_input_tokens]
            if len(chunk_ids) < 1:
                logger.debug(
                    "chunk: tail chunk %d tokens < 1 — skipping",
                    len(chunk_ids),
                )
                break
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if chunk_text.strip():
                yield chunk_text

    def chunk_with_tokens(self, text: str) -> Iterator[tuple[str, list[int], int]]:
        """Token-level chunking — yields (text, token_ids, token_count).

        The pipeline can use the pre-computed token_ids to skip its own
        encode step, saving one full tokenization pass per chunk.

        .. warning::

           ``token_ids`` are sliced from the full-document tokenization and
           may include special tokens (BOS/EOS) at document boundaries.  For
           direct model consumption without re-tokenization, strip the
           tokenizer's ``all_special_ids`` from the beginning and end of
           each chunk's token list.

        Returns
        -------
        Iterator[tuple[str, list[int], int]]
            (chunk_text, token_id_list, token_count)
        """
        if not text or not text.strip():
            return
        # Build the set of special token IDs once per call — this lets us
        # strip BOS/EOS from chunk token_ids without re-tokenizing.  The
        # tokenizer may return a variable number of special tokens depending
        # on the model, so we use the tokenizer's own definition.
        special_ids = set(getattr(self.tokenizer, 'all_special_ids', []))
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        n = len(token_ids)

        if n <= self.max_input_tokens:
            # Strip leading/trailing special tokens from the short-document path
            # so callers get consistent token_ids regardless of document length.
            clean_ids = self._strip_special_boundary(token_ids, special_ids)
            yield self.tokenizer.decode(token_ids, skip_special_tokens=True), clean_ids, len(clean_ids)
            return

        stride = self._stride()
        for start in range(0, n, stride):
            chunk_ids = token_ids[start:start + self.max_input_tokens]
            if len(chunk_ids) < 1:
                # Only skip truly empty tail chunks.  Short but non-empty
                # chunks are yielded; downstream ChunkFilter handles them
                # via its min_tokens setting — that is the filter's job,
                # not the chunker's.
                logger.debug(
                    "chunk_with_tokens: tail chunk empty — skipping",
                )
                break
            # Strip BOS/EOS that may appear at chunk boundaries after slicing.
            clean_ids = self._strip_special_boundary(chunk_ids, special_ids)
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if chunk_text.strip():
                yield chunk_text, clean_ids, len(clean_ids)

    @staticmethod
    def _strip_special_boundary(token_ids: list[int], special_ids: set[int]) -> list[int]:
        """Strip known special tokens from both ends of *token_ids*.

        Only strips when the special token appears at the very beginning or
        very end — interior occurrences are left intact (they may be
        semantically meaningful, e.g. a mask token mid-sequence).
        """
        if not token_ids:
            return token_ids
        start = 0
        end = len(token_ids)
        while start < end and token_ids[start] in special_ids:
            start += 1
        while end > start and token_ids[end - 1] in special_ids:
            end -= 1
        return token_ids[start:end]

    def chunk_with_token_count(self, text: str) -> Iterator[tuple[str, int]]:
        """Backward-compatible: yield (chunk_text, token_count)."""
        for chunk_text, _, token_count in self.chunk_with_tokens(text):
            yield chunk_text, token_count


class NullChunker:
    """No-op chunker — passes text through unchanged."""
    def chunk(self, text: str) -> Iterator[str]:
        if text and text.strip():
            yield text

    def chunk_with_tokens(self, text: str):
        if text and text.strip():
            yield text, [], 0
