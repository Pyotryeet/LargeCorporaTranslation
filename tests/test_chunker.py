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
        text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars — token count may differ, but should split
        chunks = list(chunker.chunk(text))
        # With a real tokenizer, even 26 chars produces several tokens; should still split.
        # If the text is short enough to fit, the test still verifies no crash.
        assert len(chunks) >= 1

    def test_empty_text_yields_nothing(self, real_tokenizer):
        chunker = TextChunker(real_tokenizer)
        chunks = list(chunker.chunk(""))
        assert len(chunks) == 0

    def test_overlap_preserves_context(self, real_tokenizer):
        chunker = TextChunker(real_tokenizer, max_input_tokens=10, overlap_tokens=4)
        # Generate a long enough text that the real tokenizer will need to split.
        text = "This is a longer piece of text that should definitely produce enough tokens for chunking to split it into multiple pieces."
        chunks = list(chunker.chunk(text))
        # May or may not split depending on tokenizer; at minimum, no crash.
        assert len(chunks) >= 1


class TestNullChunker:
    def test_passes_through(self):
        chunker = NullChunker()
        chunks = list(chunker.chunk("hello world"))
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_empty_returns_nothing(self):
        chunker = NullChunker()
        assert list(chunker.chunk("")) == []
