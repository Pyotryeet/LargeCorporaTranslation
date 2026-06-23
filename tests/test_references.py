"""Tests for golden reference set loading."""

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
