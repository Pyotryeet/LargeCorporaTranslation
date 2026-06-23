"""Tests for golden reference set loading."""

import json

from benchmark.quality.references import ReferenceLoader


class TestReferenceLoader:
    def test_load_references(self, golden_references_path):
        loader = ReferenceLoader(golden_references_path)
        sources, references = loader.load()
        # Real data (data/references/golden_en_tr.jsonl) has ~1960 lines;
        # the fallback fixture has 10.  Either way, sources and references
        # must be equal in length and non-empty.
        assert len(sources) > 0, "Expected at least one reference"
        assert len(references) > 0, "Expected at least one reference"
        assert len(sources) == len(references)

    def test_validate_pair(self):
        assert ReferenceLoader.validate_pair("Hello", "Merhaba") is True
        assert ReferenceLoader.validate_pair("", "Merhaba") is False
        assert ReferenceLoader.validate_pair("Hello", "") is False
        assert ReferenceLoader.validate_pair("", "") is False

    def test_missing_file_raises(self):
        loader = ReferenceLoader("/nonexistent/path.jsonl")
        import pytest
        with pytest.raises(FileNotFoundError):
            loader.load()

    # ── Edge case tests ──

    def test_load_and_validate_pair_edge_cases(self):
        """validate_pair handles edge cases correctly."""
        # Unicode / non-ASCII Turkish characters
        assert ReferenceLoader.validate_pair("Hello world", "Merhaba dünya") is True
        # Very long strings
        long_en = "The " + "quick brown fox " * 1000
        long_tr = "Hızlı " + "kahverengi tilki " * 1000
        assert ReferenceLoader.validate_pair(long_en, long_tr) is True
        # Whitespace-only strings
        assert ReferenceLoader.validate_pair("   ", "Merhaba") is False
        assert ReferenceLoader.validate_pair("Hello", "   ") is False
        # Control characters only
        assert ReferenceLoader.validate_pair("\x00\x01", "Merhaba") is False

    def test_validate_pair_strips_whitespace(self):
        """validate_pair applies strip() before checking emptiness."""
        assert ReferenceLoader.validate_pair("  Hello  ", "  Merhaba  ") is True
        assert ReferenceLoader.validate_pair("  ", "hello") is False

    def test_load_with_non_jsonl_lines_raises(self, tmp_path):
        """Lines that are not valid JSON should raise an error."""
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text("this is not json\n")
        loader = ReferenceLoader(str(bad_file))
        import pytest
        with pytest.raises((json.JSONDecodeError, ValueError)):
            loader.load()
