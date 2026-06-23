"""Tests for JSONL streaming loader — in-memory + external sort shuffle."""

import json
import os
import tempfile
from pathlib import Path

import pytest
from benchmark.data.loader import JSONLLoader


class TestJSONLLoader:
    def test_loads_sample_file(self, sample_jsonl_path):
        loader = JSONLLoader([sample_jsonl_path], shuffle=False)
        docs = list(loader.iter_documents())
        # Real data or pre-generated fixtures -- at minimum we need > 1 doc.
        assert len(docs) > 1, f"Expected more than 1 document, got {len(docs)}"
        doc_id, file_name, text = docs[0]
        assert doc_id == 0
        # Guard: data should not be the known synthetic "THIS DATA IS NOT REAL"
        # placeholder text.  If this assertion fires, tests are running on
        # auto-generated garbage — provide real data at data/input/ or
        # tests/fixtures/.
        assert "NOT REAL" not in text, (
            "Test is running on auto-generated synthetic data. "
            "Provide real data at data/input/ or tests/fixtures/."
        )

    def test_shuffle_preserves_count(self, sample_jsonl_path):
        loader = JSONLLoader([sample_jsonl_path], shuffle=True, seed=42)
        docs = list(loader.iter_documents())
        assert len(docs) > 1

    def test_deterministic_shuffle(self, sample_jsonl_path):
        loader1 = JSONLLoader([sample_jsonl_path], shuffle=True, seed=42)
        loader2 = JSONLLoader([sample_jsonl_path], shuffle=True, seed=42)
        texts1 = [t for _, _, t in loader1.iter_documents()]
        texts2 = [t for _, _, t in loader2.iter_documents()]
        assert texts1 == texts2  # Same seed = same order

    def test_different_seeds_different_order(self, sample_jsonl_path):
        loader1 = JSONLLoader([sample_jsonl_path], shuffle=True, seed=42)
        loader2 = JSONLLoader([sample_jsonl_path], shuffle=True, seed=999)
        texts1 = [t for _, _, t in loader1.iter_documents()]
        texts2 = [t for _, _, t in loader2.iter_documents()]
        # Different seeds should likely produce different orders
        # (very unlikely to be identical for more than a few items)
        # Note: for very small datasets (< 3 items), two different seeds can
        # coincidentally produce the same order.
        assert texts1 != texts2, (
            f"Different seeds (42 vs 999) produced identical order "
            f"({len(texts1)} docs). This is statistically possible but "
            f"exceedingly unlikely for non-trivial datasets."
        )

    def test_missing_pattern_does_not_crash(self):
        """Missing glob patterns warn instead of crashing — pipeline drains empty."""
        loader = JSONLLoader(["/nonexistent/pattern_*.jsonl.gz"])
        docs = list(loader.iter_documents())
        assert len(docs) == 0  # empty pipeline, no crash

    def test_gz_file(self, sample_jsonl_gz_path):
        loader = JSONLLoader([sample_jsonl_gz_path], shuffle=False)
        docs = list(loader.iter_documents())
        # Real data or pre-generated fixtures -- at minimum we need > 1 doc.
        assert len(docs) > 1, f"Expected more than 1 document, got {len(docs)}"

    # ── External sort tests ─────────────────────────────────────────

    def test_external_shuffle_preserves_count(self, sample_jsonl_path):
        """External sort yields the same number of documents as sequential."""
        loader_seq = JSONLLoader([sample_jsonl_path], shuffle=False)
        seq_count = len(list(loader_seq.iter_documents()))

        # Force external sort with a tiny 1 KB memory budget.
        loader_ext = JSONLLoader(
            [sample_jsonl_path], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,  # ~1 KB — forces external sort
        )
        ext_docs = list(loader_ext.iter_documents())
        assert len(ext_docs) == seq_count

    def test_external_shuffle_deterministic(self, sample_jsonl_path):
        """Same seed + same input → identical output order with external sort."""
        loader1 = JSONLLoader(
            [sample_jsonl_path], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        )
        loader2 = JSONLLoader(
            [sample_jsonl_path], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        )
        texts1 = [t for _, _, t in loader1.iter_documents()]
        texts2 = [t for _, _, t in loader2.iter_documents()]
        assert texts1 == texts2

    def test_external_shuffle_different_seeds(self, sample_jsonl_path):
        """Different seeds → different output order with external sort."""
        loader1 = JSONLLoader(
            [sample_jsonl_path], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        )
        loader2 = JSONLLoader(
            [sample_jsonl_path], shuffle=True, seed=999,
            max_shuffle_memory_gb=0.000001,
        )
        texts1 = [t for _, _, t in loader1.iter_documents()]
        texts2 = [t for _, _, t in loader2.iter_documents()]
        assert texts1 != texts2, (
            f"Different seeds should produce different orders "
            f"(seed 42 vs 999, {len(texts1)} docs)"
        )

    def test_external_shuffle_respects_seek(self, sample_jsonl_path):
        """seek_to(N) skips N documents from the shuffled output."""
        total = len(list(JSONLLoader(
            [sample_jsonl_path], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        ).iter_documents()))

        if total < 3:
            pytest.skip("Need at least 3 documents to test seek")

        # Get the full shuffled order.
        loader_full = JSONLLoader(
            [sample_jsonl_path], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        )
        # Collect all (doc_id, file_name, text).
        full_docs = list(loader_full.iter_documents())

        # Seek to skip the first 2 documents of the shuffled output.
        loader_seek = JSONLLoader(
            [sample_jsonl_path], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        )
        loader_seek.seek_to(2)
        after_seek = list(loader_seek.iter_documents())

        assert len(after_seek) == total - 2
        # The seek'd output should match positions 2+ of the full shuffled
        # order (deterministic via seed — the shuffle order is identical).
        assert after_seek == full_docs[2:]

    def test_fast_path_still_works(self, sample_jsonl_path):
        """Small dataset with default budget uses in-memory Fisher-Yates."""
        loader = JSONLLoader(
            [sample_jsonl_path], shuffle=True, seed=42,
            # Default budget is 2 GiB — dataset should fit.
        )
        docs = list(loader.iter_documents())
        assert len(docs) > 1
        # Verify documents are well-formed: each must have a doc_id, file_name, and
        # non-empty text.  This catches the case where synthetic auto-generated data
        # silently produces structurally valid but meaningless output.
        for doc in docs[:3]:
            assert len(doc) == 3, f"Expected (id, file, text), got {len(doc)}-tuple"
            doc_id, file_name, text = doc
            assert isinstance(doc_id, int)
            assert isinstance(file_name, str)
            assert isinstance(text, str) and len(text) > 0, (
                f"Document {doc_id} has empty text — possible fixture corruption"
            )

    def test_temp_files_cleaned_up(self, sample_jsonl_path):
        """Binary run files are cleaned up after external sort completes."""
        import tempfile as tmp

        with tmp.TemporaryDirectory() as td:
            loader = JSONLLoader(
                [sample_jsonl_path], shuffle=True, seed=42,
                max_shuffle_memory_gb=0.000001,
                shuffle_temp_dir=td,
            )
            # Exhaust the iterator to complete the external sort.
            docs = list(loader.iter_documents())
            assert len(docs) > 0

            # Check that no .bin files remain.
            bin_files = list(Path(td).glob("*.bin"))
            assert len(bin_files) == 0, (
                f"Temp files not cleaned up: {bin_files}"
            )

    @pytest.mark.xfail(
        reason=(
            "Generator finalization via __del__ is not guaranteed to run "
            "immediately in CPython. Residual .bin files may remain until the "
            "TemporaryDirectory context manager exits. This test documents "
            "expected behavior but __del__ timing is implementation-defined."
        ),
        strict=False,
    )
    def test_temp_files_cleaned_up_on_error(self, sample_jsonl_path):
        """Temp files are cleaned up even if the iterator is not exhausted."""
        import tempfile as tmp
        import gc

        with tmp.TemporaryDirectory() as td:
            loader = JSONLLoader(
                [sample_jsonl_path], shuffle=True, seed=42,
                max_shuffle_memory_gb=0.000001,
                shuffle_temp_dir=td,
            )
            # Start iterating but don't exhaust — simulate early exit.
            it = loader.iter_documents()
            _first_few = [next(it) for _ in range(min(5, 100))]
            # The generator is garbage-collected here; the finally
            # block in _external_shuffle_iter runs via generator.close().
            del it
            del loader
            gc.collect()  # Encourage immediate finalization

            # Check that .bin files were cleaned up by generator finalization.
            bin_files = list(Path(td).glob("*.bin"))
            # Generator finalization via __del__ is not guaranteed to run
            # immediately, so we mark this as xfail (non-strict).
            assert len(bin_files) == 0, (
                f"Temp files not cleaned up: {bin_files}"
            )

    def test_gz_external_shuffle(self, sample_jsonl_gz_path):
        """External sort works with gzip-compressed input."""
        loader_seq = JSONLLoader([sample_jsonl_gz_path], shuffle=False)
        seq_count = len(list(loader_seq.iter_documents()))

        loader_ext = JSONLLoader(
            [sample_jsonl_gz_path], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        )
        ext_docs = list(loader_ext.iter_documents())
        assert len(ext_docs) == seq_count
        assert len(ext_docs) > 1

    def test_empty_dataset(self, tmp_path):
        """Both shuffle paths handle zero documents gracefully."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        loader = JSONLLoader([str(empty_file)], shuffle=True, seed=42)
        docs = list(loader.iter_documents())
        assert len(docs) == 0

        # Also test with external sort forced.
        loader2 = JSONLLoader(
            [str(empty_file)], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        )
        docs2 = list(loader2.iter_documents())
        assert len(docs2) == 0

    def test_external_shuffle_multi_file(self, tmp_path):
        """External sort works correctly with multiple input files."""
        # Create 3 files with 20 documents each.
        rng = __import__('random').Random(42)
        files = []
        for fi in range(3):
            fpath = tmp_path / f"sample_{fi}.jsonl"
            with open(fpath, "w", encoding="utf-8") as f:
                for i in range(20):
                    text = " ".join(
                        f"word_{rng.randint(0, 1000)}"
                        for _ in range(rng.randint(5, 40))
                    )
                    f.write(json.dumps({"text": text}) + "\n")
            files.append(str(fpath))

        # Sequential count from all files.
        loader_seq = JSONLLoader(files, shuffle=False)
        seq_docs = list(loader_seq.iter_documents())
        assert len(seq_docs) == 60

        # External sort — must preserve document count.
        loader_ext = JSONLLoader(
            files, shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        )
        ext_docs = list(loader_ext.iter_documents())
        assert len(ext_docs) == 60

        # All original texts should be present.
        seq_texts = {t for _, _, t in seq_docs}
        ext_texts = {t for _, _, t in ext_docs}
        assert seq_texts == ext_texts

    def test_external_shuffle_texts_with_non_ascii(self, tmp_path):
        """Documents with non-ASCII (tr characters) survive the round-trip."""
        fpath = tmp_path / "turkish.jsonl"
        turkish_texts = [
            "İstanbul'da güneşli bir gün",
            "Türkçe karakterler: ğ, ü, ş, ı, ö, ç",
            "Ankara'nın bağları",
            "Bugün hava çok sıcak",
            "Merhaba dünya!",
        ]
        with open(fpath, "w", encoding="utf-8") as f:
            for t in turkish_texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

        loader = JSONLLoader(
            [str(fpath)], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        )
        texts = {t for _, _, t in loader.iter_documents()}
        assert texts == set(turkish_texts)

    def test_both_paths_produce_same_texts(self, sample_jsonl_path):
        """In-memory and external shuffle produce the same set of texts."""
        # In-memory path.
        loader_mem = JSONLLoader([sample_jsonl_path], shuffle=True, seed=42)
        mem_texts = {t for _, _, t in loader_mem.iter_documents()}

        # External sort path.
        loader_ext = JSONLLoader(
            [sample_jsonl_path], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.000001,
        )
        ext_texts = {t for _, _, t in loader_ext.iter_documents()}

        assert mem_texts == ext_texts
        assert len(mem_texts) > 0

    def test_external_shuffle_multi_pass_merge(self, tmp_path):
        """Multi-pass merge works when runs > SHUFFLE_MAX_OPEN_RUNS.

        Uses an extremely tiny budget to generate many small run files,
        ensuring the multi-pass merge codepath is exercised.
        """
        # Generate 500+ documents so with a 500-byte budget we get many runs.
        fpath = tmp_path / "many.jsonl"
        with open(fpath, "w", encoding="utf-8") as f:
            for i in range(600):
                text = f"Document number {i} with some extra text to push past the tiny budget. " * 3
                f.write(json.dumps({"text": text}) + "\n")

        loader_seq = JSONLLoader([str(fpath)], shuffle=False)
        seq_count = len(list(loader_seq.iter_documents()))
        assert seq_count == 600

        # 100-byte budget → many run files.
        loader_ext = JSONLLoader(
            [str(fpath)], shuffle=True, seed=42,
            max_shuffle_memory_gb=0.0000001,  # ~100 bytes
        )
        ext_docs = list(loader_ext.iter_documents())
        assert len(ext_docs) == 600

        # Verify all texts are present.
        seq_texts = {t for _, _, t in loader_seq.iter_documents()}
        ext_texts = {t for _, _, t in ext_docs}
        assert seq_texts == ext_texts
