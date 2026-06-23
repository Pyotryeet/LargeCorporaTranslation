"""Tests for text chunking."""

import pytest
from benchmark.data.chunker import TextChunker, NullChunker


class TestTextChunker:
    def test_short_text_passes_through(self, real_tokenizer):
        chunker = TextChunker(real_tokenizer, max_input_tokens=100)
        text = "hello world test"
        chunks = list(chunker.chunk(text))
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_split(self, real_tokenizer):
        chunker = TextChunker(real_tokenizer, max_input_tokens=10, overlap_tokens=2)
        # Generate text long enough to guarantee splitting with any tokenizer:
        # 500 words repeated ensures well beyond the 10-token limit.
        text = " ".join(["abcdefghijklmnopqrstuvwxyz"] * 500)
        chunks = list(chunker.chunk(text))
        assert len(chunks) > 1, (
            f"Expected multiple chunks for {len(text)}-char text with "
            f"max_input_tokens=10, got {len(chunks)}"
        )
        # Content-based assertions: every chunk must be non-empty.
        # We cannot assert exact word-level subset because subword tokenizers
        # may split mid-word, and decode-after-slice produces partial tokens
        # that differ from the original text.
        for i, chunk in enumerate(chunks):
            assert len(chunk) > 0, f"Chunk {i} is empty"
            assert any(c.isalpha() for c in chunk), (
                f"Chunk {i} contains no alphabetic content: {chunk[:80]!r}"
            )

    def test_empty_text_yields_nothing(self, real_tokenizer):
        chunker = TextChunker(real_tokenizer)
        chunks = list(chunker.chunk(""))
        assert len(chunks) == 0

    def test_overlap_preserves_context(self, real_tokenizer):
        chunker = TextChunker(real_tokenizer, max_input_tokens=10, overlap_tokens=4)
        # Generate a long enough text that the real tokenizer will need to split.
        text = " ".join(["This is a longer piece of text that should definitely produce enough tokens for chunking to split it into multiple pieces."] * 50)
        chunks = list(chunker.chunk(text))
        # May or may not split depending on tokenizer; at minimum, no crash.
        assert len(chunks) >= 1
        # When chunking splits the text, verify overlap by checking that
        # content from the end of one chunk appears near the start of the next.
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                # Get the last few words of chunk[i] — they should overlap into chunk[i+1].
                tail_words = set(chunks[i].split()[-5:])
                head_words = set(chunks[i + 1].split()[:5])
                assert tail_words & head_words, (
                    f"Chunks {i} and {i+1} share no overlapping content. "
                    f"Tail of chunk {i}: {list(tail_words)[:3]}, "
                    f"Head of chunk {i+1}: {list(head_words)[:3]}"
                )


class TestNullChunker:
    def test_passes_through(self):
        chunker = NullChunker()
        chunks = list(chunker.chunk("hello world"))
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_empty_returns_nothing(self):
        chunker = NullChunker()
        assert list(chunker.chunk("")) == []
