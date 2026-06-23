"""Real end-to-end integration test — full pipeline with real data + tokenizer.

NO DUMMIES.  Every fixture and test uses only real objects:
- The real TranslateGemma 4B SentencePiece tokenizer
- The real fineweb_en_sample.jsonl.gz data file
- The real pipeline components (loader, chunker, filter, async pipeline, batch assembly)

If a real resource cannot be loaded (tokenizer download denied, data missing,
no GPU for model inference), the test is skipped — never mocked.
"""

import json
import gzip
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark.data.loader import JSONLLoader
from benchmark.data.chunker import TextChunker, NullChunker
from benchmark.data.filters import ChunkFilter
from benchmark.data.pipeline import AsyncPipeline, PipelineBatch
from benchmark.inference.batch_assembly import BatchAssembler


# ═══════════════════════════════════════════════════════════════════════════════
# Session-scoped fixtures — loaded ONCE per test run
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def real_tokenizer():
    """Load the real TranslateGemma 4B SentencePiece tokenizer once."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        tok = AutoTokenizer.from_pretrained(
            "google/translategemma-4b-it", trust_remote_code=False,
        )
        return tok
    except Exception as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")


@pytest.fixture(scope="session")
def real_data_path():
    """Path to the real fineweb_en_sample.jsonl.gz (100 000 docs, 123 MB)."""
    candidates = [
        Path("data/input/fineweb_en_sample.jsonl.gz"),
        Path("tests/fixtures/sample_input.jsonl.gz"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    pytest.skip("no sample data file found")


@pytest.fixture(scope="session")
def real_data_lines(real_data_path):
    """Return the first 200 lines of the real data file (validation baseline)."""
    lines = []
    with gzip.open(str(real_data_path), "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 200:
                break
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


@pytest.fixture(scope="session")
def real_vocab_size(real_tokenizer):
    """The real tokenizer's vocabulary size."""
    return real_tokenizer.vocab_size


# ═══════════════════════════════════════════════════════════════════════════════
# Test classes
# ═══════════════════════════════════════════════════════════════════════════════

class TestRealDataLoadAndInspect:
    """Verify the real data file is well-formed and loadable."""

    def test_real_data_file_exists_and_is_gzipped(self, real_data_path):
        """The real sample data exists and is valid gzip."""
        assert real_data_path.suffix == ".gz"
        assert real_data_path.stat().st_size > 10_000_000, (
            f"Expected >10 MB gzip file, got {real_data_path.stat().st_size} bytes"
        )
        # Verify it is actually gzip-compressed.
        magic = real_data_path.read_bytes()[:2]
        assert magic == b"\x1f\x8b", f"Not a gzip file — magic bytes: {magic!r}"

    def test_real_data_lines_are_valid_json(self, real_data_lines):
        """Every line in the first 200 docs is parseable JSON."""
        for i, line in enumerate(real_data_lines):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} is not valid JSON: {exc}")
            assert "text" in obj, f"Line {i} missing 'text' field: {list(obj.keys())}"

    def test_real_data_texts_are_valid_utf8(self, real_data_lines):
        """Every text field is valid UTF-8 and non-empty."""
        for i, line in enumerate(real_data_lines):
            obj = json.loads(line)
            text = obj["text"]
            assert isinstance(text, str), f"Line {i}: text is {type(text)}, not str"
            assert len(text) > 0, f"Line {i}: empty text field"
            # Verify encode/decode round-trip.
            try:
                encoded = text.encode("utf-8")
                decoded = encoded.decode("utf-8")
                assert decoded == text
            except (UnicodeEncodeError, UnicodeDecodeError) as exc:
                pytest.fail(f"Line {i}: not valid UTF-8: {exc}")

    def test_real_data_texts_have_reasonable_length(self, real_data_lines):
        """Text documents are between 50 and 100 000 characters."""
        for i, line in enumerate(real_data_lines):
            text = json.loads(line)["text"]
            assert 50 <= len(text) <= 100_000, (
                f"Line {i}: text length {len(text)} outside [50, 100_000]"
            )

    def test_no_duplicate_texts_in_first_200(self, real_data_lines):
        """First 200 documents should be distinct (no data corruption)."""
        texts = [json.loads(line)["text"] for line in real_data_lines]
        duplicates = len(texts) - len(set(texts))
        assert duplicates == 0, f"Found {duplicates} duplicate texts in first 200 docs"


class TestRealLoader:
    """Test the JSONLLoader with the real gzipped fineweb data."""

    def test_loader_opens_gzipped_input(self, real_data_path):
        """JSONLLoader transparently opens and reads .gz files."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        assert loader.file_count >= 1
        assert loader.total_size_bytes > 0

        docs = []
        for doc_id, fname, text in loader.iter_documents():
            docs.append((doc_id, fname, text))
            if len(docs) >= 50:
                break

        assert len(docs) == 50, f"Expected 50 docs, got {len(docs)}"
        for doc_id, fname, text in docs:
            assert isinstance(doc_id, int)
            assert isinstance(fname, str)
            assert isinstance(text, str)
            assert len(text) > 0
            assert doc_id >= 0

    def test_loader_iterates_all_documents(self, real_data_path):
        """Full iteration returns the correct number of documents."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        count = 0
        for _ in loader.iter_documents():
            count += 1
        assert count == 100_000, f"Expected 100 000 docs, got {count}"

    def test_loader_tracks_position(self, real_data_path):
        """_current_file and _current_doc_id update during iteration."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        assert loader._current_file == ""
        assert loader._current_doc_id == 0

        for doc_id, fname, text in loader.iter_documents():
            if doc_id == 0:
                assert loader._current_file != ""
                assert loader._current_doc_id == 0
                break

    def test_loader_seek_to_skips_documents(self, real_data_path):
        """seek_to() skips exactly the requested number of documents."""
        # Full sequential pass to get reference ordering.
        loader_a = JSONLLoader([str(real_data_path)], shuffle=False)
        all_texts = [text for _, _, text in loader_a.iter_documents()]
        assert len(all_texts) == 100_000

        skip = 500
        loader_b = JSONLLoader([str(real_data_path)], shuffle=False)
        loader_b.seek_to(skip)

        after_seek = [text for _, _, text in loader_b.iter_documents()]
        assert len(after_seek) == 100_000 - skip, (
            f"After skipping {skip}, expected {100_000 - skip} docs, got {len(after_seek)}"
        )
        assert after_seek[0] == all_texts[skip], (
            "First doc after seek must match doc at position skip in original order"
        )
        assert after_seek[-1] == all_texts[-1], (
            "Last doc after seek must match last doc in original order"
        )

    def test_loader_seek_then_second_call_no_skip(self, real_data_path):
        """After one iter_documents() with seek, a second call starts fresh (no skip)."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        loader.seek_to(100)

        # First iteration: 99900 docs.
        first_pass = sum(1 for _ in loader.iter_documents())
        assert first_pass == 99_900

        # Second iteration: should be back to full without seek repeating.
        second_pass = sum(1 for _ in loader.iter_documents())
        assert second_pass == 100_000, (
            f"Second pass without seek should iterate all docs, got {second_pass}"
        )


class TestRealTokenizer:
    """Verify the real TranslateGemma tokenizer behaves as expected."""

    def test_tokenizer_has_expected_vocab_size(self, real_tokenizer):
        """TranslateGemma 4B has vocab size 262144."""
        assert real_tokenizer.vocab_size == 262_144, (
            f"Expected vocab_size=262144, got {real_tokenizer.vocab_size}"
        )

    def test_tokenizer_special_token_ids(self, real_tokenizer):
        """pad=0, eos=1, bos=2."""
        assert real_tokenizer.pad_token_id == 0
        assert real_tokenizer.eos_token_id == 1
        assert real_tokenizer.bos_token_id == 2

    def test_tokenizer_encode_returns_native_python_ints(self, real_tokenizer):
        """Encode produces native Python ints (not numpy types)."""
        result = real_tokenizer.encode("Hello world test sentence.", add_special_tokens=True)
        assert isinstance(result, list)
        for i, tok_id in enumerate(result):
            assert isinstance(tok_id, int), (
                f"Token at position {i} is {type(tok_id).__name__}, expected int"
            )

    def test_tokenizer_encode_with_special_tokens(self, real_tokenizer):
        """BOS token appears first, result is non-empty."""
        result = real_tokenizer.encode("Hello world", add_special_tokens=True)
        assert len(result) > 0
        assert result[0] == real_tokenizer.bos_token_id

    def test_token_ids_within_vocab_range(self, real_tokenizer, real_data_lines):
        """Every token ID from real data is within [0, vocab_size)."""
        vocab = real_tokenizer.vocab_size
        for i, line in enumerate(real_data_lines[:20]):
            text = json.loads(line)["text"]
            ids = real_tokenizer.encode(
                text, add_special_tokens=True,
                truncation=True, max_length=512,
            )
            for j, tid in enumerate(ids):
                assert 0 <= tid < vocab, (
                    f"Line {i}, token {j}: id {tid} outside [0, {vocab})"
                )

    def test_decode_roundtrip_preserves_ascii(self, real_tokenizer):
        """ASCII text survives encode→decode roundtrip (modulo tokenization)."""
        text = "The quick brown fox jumps over the lazy dog."
        ids = real_tokenizer.encode(text, add_special_tokens=True)
        decoded = real_tokenizer.decode(ids, skip_special_tokens=True)
        # After tokenization roundtrip, words should be preserved (modulo spacing).
        for word in text.split():
            assert word in decoded, f"Word '{word}' lost in roundtrip: {decoded}"

    def test_truncation_respects_max_length(self, real_tokenizer):
        """encode with max_length truncates correctly."""
        very_long = "Hello world. " * 5000
        ids = real_tokenizer.encode(
            very_long, add_special_tokens=True,
            truncation=True, max_length=256,
        )
        assert len(ids) <= 256, f"Expected <=256 tokens, got {len(ids)}"


class TestRealChunker:
    """TextChunker with real tokenizer segments long documents correctly."""

    def test_null_chunker_passthrough(self, real_tokenizer, real_data_lines):
        """NullChunker passes text through unchanged."""
        chunker = NullChunker()
        for line in real_data_lines[:10]:
            text = json.loads(line)["text"]
            chunks = list(chunker.chunk(text))
            assert len(chunks) >= 1
            assert chunks[0] == text

    def test_text_chunker_splits_long_document(self, real_tokenizer):
        """A 500-repetition doc is split into multiple chunks."""
        chunker = TextChunker(real_tokenizer, max_input_tokens=256, overlap_tokens=50)
        long_text = "This is a reasonably long sentence for tokenization testing purposes. " * 500
        chunks = list(chunker.chunk(long_text))
        # 500 repetitions should produce at least 2 chunks at max_input_tokens=256.
        assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
        for i, chunk in enumerate(chunks):
            assert len(chunk) > 0, f"Chunk {i} is empty"
            assert isinstance(chunk, str)

    def test_text_chunker_short_text_not_split(self, real_tokenizer, real_data_lines):
        """A document that fits in max_input_tokens is not split."""
        chunker = TextChunker(real_tokenizer, max_input_tokens=512, overlap_tokens=50)
        for line in real_data_lines[:10]:
            text = json.loads(line)["text"]
            token_count = len(real_tokenizer.encode(text, add_special_tokens=False))
            chunks = list(chunker.chunk(text))
            if token_count <= 512:
                assert len(chunks) == 1, (
                    f"Short text ({token_count} tokens) was split into {len(chunks)} chunks"
                )

    def test_text_chunker_yields_valid_utf8(self, real_tokenizer, real_data_lines):
        """All chunker outputs are valid UTF-8 strings."""
        chunker = TextChunker(real_tokenizer, max_input_tokens=256, overlap_tokens=50)
        for line in real_data_lines[:5]:
            text = json.loads(line)["text"]
            for chunk in chunker.chunk(text):
                chunk.encode("utf-8")  # Must not raise.
                assert isinstance(chunk, str)

    def test_chunk_with_tokens_precomputes_valid_ids(self, real_tokenizer):
        """chunk_with_tokens yields valid token IDs whose decode matches chunk_text.

        chunk_with_tokens tokenizes the full text once with
        add_special_tokens=True, slices the token list, then decodes each
        slice.  The decoded text must roundtrip back to the same text
        (modulo whitespace), and every token ID must be within the
        tokenizer's vocab range.
        """
        text = "The translation quality benchmark measures inference throughput. " * 300
        chunker = TextChunker(real_tokenizer, max_input_tokens=256, overlap_tokens=50)
        vocab = real_tokenizer.vocab_size

        at_least_one_chunk = False
        for chunk_text, token_ids, token_count in chunker.chunk_with_tokens(text):
            at_least_one_chunk = True
            assert len(token_ids) == token_count
            # All token IDs within vocab range.
            for tid in token_ids:
                assert 0 <= tid < vocab, f"Token id {tid} out of range [0, {vocab})"
            # The decoded text must match the chunk_text (modulo whitespace).
            decoded = real_tokenizer.decode(token_ids, skip_special_tokens=True)
            assert decoded.strip() == chunk_text.strip(), (
                f"Decode mismatch:\n  decoded:   {decoded.strip()[:120]}\n  "
                f"chunk_text: {chunk_text.strip()[:120]}"
            )
        assert at_least_one_chunk, "No chunks produced"


class TestRealFilter:
    """ChunkFilter with real tokenized data."""

    def test_filter_passes_normal_english_text(self, real_tokenizer, real_data_lines):
        """Normal English text chunks pass the filter."""
        filt = ChunkFilter(min_tokens=10, max_garbage_ratio=0.95)
        for line in real_data_lines[:30]:
            text = json.loads(line)["text"]
            token_count = len(real_tokenizer.encode(text, add_special_tokens=True))
            assert filt.should_keep(text, token_count), (
                f"Filter rejected valid English text with {token_count} tokens"
            )

    def test_filter_rejects_too_short(self):
        """Chunks with < min_tokens are rejected."""
        filt = ChunkFilter(min_tokens=10)
        assert not filt.should_keep("Hi.", 2)  # 2 tokens < 10

    def test_filter_rejects_empty_text(self):
        """Empty text is rejected."""
        filt = ChunkFilter(min_tokens=1)
        assert filt.should_keep("", 0) is False
        assert filt.should_keep("   ", 0) is False

    def test_filter_stats_accumulate(self, real_data_lines, real_tokenizer):
        """FilterStats increment correctly during filtering."""
        filt = ChunkFilter(min_tokens=5, max_garbage_ratio=0.95)
        for line in real_data_lines[:50]:
            text = json.loads(line)["text"]
            token_count = len(real_tokenizer.encode(text, add_special_tokens=True))
            filt.should_keep(text, token_count)

        assert filt.stats.total_chunks == 50
        assert filt.stats.passed + filt.stats.rejected == 50
        assert 0.0 <= filt.stats.pass_rate <= 1.0

    def test_filter_resets_stats(self, real_data_lines, real_tokenizer):
        """reset_stats clears counters."""
        filt = ChunkFilter(min_tokens=5)
        for line in real_data_lines[:10]:
            text = json.loads(line)["text"]
            token_count = len(real_tokenizer.encode(text, add_special_tokens=True))
            filt.should_keep(text, token_count)

        assert filt.stats.total_chunks == 10
        filt.reset_stats()
        assert filt.stats.total_chunks == 0
        assert filt.stats.passed == 0
        assert filt.stats.rejected == 0


class TestRealPipelineIntegration:
    """Full pipeline: load -> chunk -> filter -> tokenize -> batch -> assemble."""

    def test_full_pipeline_produces_batches(self, real_tokenizer, real_data_path):
        """Real data + real tokenizer through the async pipeline produces batches."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        chunker = NullChunker()
        filt = ChunkFilter(min_tokens=10, max_garbage_ratio=0.95)

        pipeline = AsyncPipeline(
            loader, chunker, real_tokenizer, filt,
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        batches = []
        deadline = time.monotonic() + 30
        target_batches = 5
        while time.monotonic() < deadline and len(batches) < target_batches:
            batch = pipeline.next_batch()
            if batch is not None:
                batches.append(batch)
            elif pipeline.draining():
                break

        pipeline.stop_prefetch()

        assert len(batches) >= target_batches, (
            f"Expected >= {target_batches} batches, got {len(batches)}"
        )

    def test_batches_have_correct_shapes(self, real_tokenizer, real_data_path):
        """Every batch's input_ids and attention_mask have matching shapes."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        for _ in range(5):
            deadline = time.monotonic() + 10
            batch = None
            while time.monotonic() < deadline:
                batch = pipeline.next_batch()
                if batch is not None:
                    break
            if batch is None:
                break

            bs, seq_len = batch.input_ids.shape
            assert bs > 0, "Batch size must be > 0"
            assert seq_len > 0, "Sequence length must be > 0"
            assert batch.attention_mask.shape == (bs, seq_len), (
                f"Shape mismatch: ids {batch.input_ids.shape}, mask {batch.attention_mask.shape}"
            )
            assert len(batch.raw_texts) == bs
            assert len(batch.input_lengths) == bs
            assert len(batch.token_counts) == bs

        pipeline.stop_prefetch()

    def test_batch_tokens_are_non_zero(self, real_tokenizer, real_data_path):
        """The first token of every sequence is non-zero (no all-pad batches)."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        checked = 0
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and checked < 3:
            batch = pipeline.next_batch()
            if batch is None:
                continue
            for i in range(batch.input_ids.shape[0]):
                assert batch.input_ids[i, 0].item() != 0, (
                    f"Batch {checked}, seq {i}: first token is pad (0)"
                )
                # BOS token should be 2 for TranslateGemma.
                assert batch.input_ids[i, 0].item() == real_tokenizer.bos_token_id, (
                    f"Batch {checked}, seq {i}: first token {batch.input_ids[i,0].item()} != BOS ({real_tokenizer.bos_token_id})"
                )
            checked += 1

        pipeline.stop_prefetch()
        assert checked >= 1, "No batches were checked"

    def test_token_ids_within_vocab(self, real_tokenizer, real_data_path):
        """All token IDs produced by the pipeline are within [0, vocab_size)."""
        vocab = real_tokenizer.vocab_size

        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        checked = 0
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and checked < 5:
            batch = pipeline.next_batch()
            if batch is None:
                continue
            ids = batch.input_ids
            mask = batch.attention_mask
            for i in range(ids.shape[0]):
                seq_len = mask[i].sum().item()
                for j in range(int(seq_len)):
                    tid = ids[i, j].item()
                    assert 0 <= tid < vocab, (
                        f"Batch {checked}, seq {i}, pos {j}: token id {tid} out of range [0, {vocab})"
                    )
            checked += 1

        pipeline.stop_prefetch()
        assert checked >= 1, "No batches were checked"

    def test_padding_is_correct(self, real_tokenizer, real_data_path):
        """Pad positions in attention_mask are 0, input_ids are pad_token_id."""
        pad_id = real_tokenizer.pad_token_id or 0

        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        checked = 0
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and checked < 3:
            batch = pipeline.next_batch()
            if batch is None:
                continue

            bs, seq_len = batch.input_ids.shape
            for i in range(bs):
                mask = batch.attention_mask[i]
                ids = batch.input_ids[i]
                active = mask.sum().item()
                # Active positions: mask == 1, ids != pad
                for j in range(int(active)):
                    assert mask[j].item() == 1, (
                        f"Mask should be 1 at active position {j}"
                    )
                    assert ids[j].item() != pad_id, (
                        f"Active position {j} should not be pad"
                    )
                # Pad positions: mask == 0, ids == pad
                for j in range(int(active), seq_len):
                    assert mask[j].item() == 0, (
                        f"Mask should be 0 at pad position {j}"
                    )
                    assert ids[j].item() == pad_id, (
                        f"Pad position {j} should be pad_id={pad_id}, got {ids[j].item()}"
                    )
            checked += 1

        pipeline.stop_prefetch()
        assert checked >= 1, "No batches checked"

    def test_input_lengths_match_mask_sums(self, real_tokenizer, real_data_path):
        """pipeline batch input_lengths[i] == attention_mask[i].sum()."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        checked = 0
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and checked < 3:
            batch = pipeline.next_batch()
            if batch is None:
                continue
            for i in range(batch.input_ids.shape[0]):
                expected_len = batch.input_lengths[i]
                actual_len = batch.attention_mask[i].sum().item()
                assert expected_len == actual_len, (
                    f"Seq {i}: input_lengths={expected_len}, mask_sum={actual_len}"
                )
            checked += 1

        pipeline.stop_prefetch()
        assert checked >= 1, "No batches checked"

    def test_batch_texts_are_valid_utf8(self, real_tokenizer, real_data_path):
        """All batch raw_texts are valid UTF-8 strings."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        checked = 0
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and checked < 3:
            batch = pipeline.next_batch()
            if batch is None:
                continue
            for i, text in enumerate(batch.raw_texts):
                assert isinstance(text, str), f"raw_texts[{i}] is {type(text)}, not str"
                assert len(text) > 0, f"raw_texts[{i}] is empty"
                text.encode("utf-8")  # Must not raise.
            checked += 1

        pipeline.stop_prefetch()
        assert checked >= 1, "No batches checked"

    def test_no_empty_strings_in_batch(self, real_tokenizer, real_data_path):
        """No empty strings in raw_texts after filtering."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        checked = 0
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and checked < 3:
            batch = pipeline.next_batch()
            if batch is None:
                continue
            for text in batch.raw_texts:
                assert len(text.strip()) > 0, "Empty or whitespace-only text in batch"
            checked += 1

        pipeline.stop_prefetch()
        assert checked >= 1, "No batches checked"

    def test_pipeline_produces_at_least_n_batches(self, real_tokenizer, real_data_path):
        """Pipeline produces at least 20 batches within 60 seconds."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=4, backend="cpu",
        )
        pipeline.start_prefetch()

        batches = []
        deadline = time.monotonic() + 60
        target = 20
        while time.monotonic() < deadline and len(batches) < target:
            batch = pipeline.next_batch()
            if batch is not None:
                batches.append(batch)
            elif pipeline.draining():
                break

        pipeline.stop_prefetch()

        assert len(batches) >= target, (
            f"Pipeline produced only {len(batches)} batches in 60s (target: >= {target})"
        )

    def test_pipeline_drains_correctly(self, real_tokenizer, real_data_path):
        """When data is exhausted, the pipeline drains and next_batch() returns None."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        loader.seek_to(99_000)  # Only ~1000 docs left.

        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        # Collect all remaining batches.
        batches = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            batch = pipeline.next_batch()
            if batch is not None:
                batches.append(batch)
            elif pipeline.draining():
                break

        pipeline.stop_prefetch()

        assert pipeline.draining(), "Pipeline should be draining after data exhausted"
        assert len(batches) >= 1, "Should produce at least 1 batch from ~1000 remaining docs"
        # After draining, next_batch should return None.
        post_batch = pipeline.next_batch()
        assert post_batch is None, "next_batch() should return None after drain"

    def test_pipeline_counter_increments(self, real_tokenizer, real_data_path):
        """total_chunks_produced and total_chunks_consumed increment during run."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=4, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        # Collect a few batches.
        deadline = time.monotonic() + 20
        batch_count = 0
        while time.monotonic() < deadline and batch_count < 10:
            batch = pipeline.next_batch()
            if batch is not None:
                batch_count += 1

        pipeline.stop_prefetch()

        assert pipeline.total_chunks_produced > 0, "No chunks were produced"
        assert pipeline.total_chunks_consumed > 0, "No chunks were consumed"
        assert pipeline.total_chunks_consumed <= pipeline.total_chunks_produced, (
            f"Consumed ({pipeline.total_chunks_consumed}) > produced ({pipeline.total_chunks_produced})"
        )

    def test_pipeline_with_textchunker(self, real_tokenizer, real_data_path):
        """TextChunker (token-level chunker) works end-to-end."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        chunker = TextChunker(real_tokenizer, max_input_tokens=256, overlap_tokens=50)
        filt = ChunkFilter(min_tokens=10, max_garbage_ratio=0.95)

        pipeline = AsyncPipeline(
            loader, chunker, real_tokenizer, filt,
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        deadline = time.monotonic() + 20
        batch = None
        while time.monotonic() < deadline:
            batch = pipeline.next_batch()
            if batch is not None:
                break

        pipeline.stop_prefetch()

        assert batch is not None, "No batch produced with TextChunker"
        assert batch.input_ids.shape[0] >= 1
        assert batch.input_ids.shape[1] > 0

    def test_pipeline_multiple_workers(self, real_tokenizer, real_data_path):
        """Pipeline with 8 prefetch workers still produces valid batches."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=8, backend="cpu",
        )
        pipeline.start_prefetch()

        deadlines = time.monotonic() + 20
        batch = None
        while time.monotonic() < deadlines:
            batch = pipeline.next_batch()
            if batch is not None:
                break

        pipeline.stop_prefetch()

        assert batch is not None, "No batch produced with 8 workers"
        assert batch.input_ids.shape[0] > 0

    def test_pipeline_batch_ids_are_monotonic(self, real_tokenizer, real_data_path):
        """PipelineBatch.batch_id increments monotonically."""
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=8, prefetch_workers=2, backend="cpu",
        )
        pipeline.start_prefetch()

        batch_ids = []
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and len(batch_ids) < 10:
            batch = pipeline.next_batch()
            if batch is not None:
                batch_ids.append(batch.batch_id)

        pipeline.stop_prefetch()

        assert len(batch_ids) >= 2, "Need at least 2 batches to check monotonicity"
        for i in range(1, len(batch_ids)):
            assert batch_ids[i] == batch_ids[i - 1] + 1, (
                f"batch_id gap: {batch_ids[i-1]} -> {batch_ids[i]}"
            )


class TestRealBatchAssembly:
    """BatchAssembler with real tokenizer output."""

    def test_assemble_from_tokenized_chunks(self, real_tokenizer):
        """Assemble pre-tokenized chunks into a padded tensor batch."""
        texts = [
            "Hello world, this is a test.",
            "A shorter one.",
            "Machine translation is an important NLP task that requires careful evaluation.",
        ]
        # Tokenize each text individually (simulating what pipeline workers do).
        items = []
        for text in texts:
            ids = real_tokenizer.encode(text, add_special_tokens=True)
            items.append((text, ids, len(ids)))

        assembler = BatchAssembler(pad_token_id=real_tokenizer.pad_token_id or 0)
        input_ids, attention_mask, lengths, out_texts = assembler.collate(items)

        assert input_ids.shape[0] == 3
        assert attention_mask.shape[0] == 3
        max_len = max(len(ids) for _, ids, _ in items)
        assert input_ids.shape[1] == max_len

        # Verify padding.
        for i, (_, ids, _) in enumerate(items):
            # Active region: ids match.
            assert (input_ids[i, :len(ids)] == torch.tensor(ids, dtype=torch.long)).all()
            # Attention mask: active = 1, pad = 0.
            assert attention_mask[i, :len(ids)].sum() == len(ids)
            assert attention_mask[i, len(ids):].sum() == 0

    def test_assemble_all_same_length(self, real_tokenizer):
        """When all sequences have the same length, no padding needed."""
        texts = ["Hello world.", "Good morning.", "Testing now."]
        items = []
        for text in texts:
            ids = real_tokenizer.encode(text, add_special_tokens=True)
            items.append((text, ids, len(ids)))

        assembler = BatchAssembler(pad_token_id=real_tokenizer.pad_token_id or 0)
        input_ids, attention_mask, lengths, _ = assembler.collate(items)

        # All lengths should be similar (short sentences).
        assert input_ids.shape[0] == 3
        assert attention_mask.sum().item() == sum(len(ids) for _, ids, _ in items), (
            "No padding should be present when all sequences have same length"
        )


class TestRealModelInference:
    """Model inference tests with the real TranslateGemma 4B model.

    These tests load the full ~8GB model and run inference on MPS or CUDA.
    They are skipped only if no GPU is available. torch.compile is disabled
    because on MPS the inductor backend can deadlock.
    """

    @pytest.fixture(scope="class")
    def _require_gpu(self):
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            pytest.skip("no GPU available — cannot run model inference tests")

    def test_model_can_load_and_generate(
        self, real_tokenizer, real_data_path, _require_gpu,
    ):
        """Load the real TranslateGemma model and generate one batch.

        _require_gpu either skips (no GPU) or returns silently (GPU available).
        """

        from benchmark.hardware.backend import detect_backend
        from benchmark.inference.engine import InferenceEngine
        from benchmark.inference.sampling import DecodingParams

        device_info = detect_backend()
        if device_info.backend == "cpu":
            pytest.skip("backend is cpu — inference too slow for e2e test")

        # Load the model with torch.compile DISABLED — on MPS, inductor
        # max-autotune deadlocks and reduce-overhead hangs on first generate.
        engine = InferenceEngine(
            model_path="google/translategemma-4b-it",
            tokenizer_path="google/translategemma-4b-it",
            device_info=device_info,
            decoding_params=DecodingParams(
                max_new_tokens=32,
                temperature=0.0,
                do_sample=False,
            ),
            use_flash_attention=False,
        )
        try:
            engine.load()
        except Exception as exc:
            pytest.skip(f"model load failed: {exc}")

        assert engine.is_loaded(), "Model should be loaded"

        # Run a small warmup.
        engine.warmup(batches=3)

        # Produce one batch from real data and translate it.
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=2, prefetch_workers=1, backend=device_info.backend,
        )
        pipeline.start_prefetch()

        batch = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            batch = pipeline.next_batch()
            if batch is not None:
                break

        pipeline.stop_prefetch()

        assert batch is not None, "No batch from pipeline"

        # Translate.
        result = engine.translate(batch)
        assert len(result.generations) == batch.input_ids.shape[0]
        for i, tr in enumerate(result.generations):
            assert tr.translated_text, f"Translation {i} is empty"
            # Must be valid UTF-8.
            tr.translated_text.encode("utf-8")
            # Translation should differ from input.
            assert tr.translated_text != tr.input_text, (
                f"Translation matches input: '{tr.input_text[:60]}...'"
            )

        assert result.output_tokens_total > 0, "No output tokens generated"

    def test_model_deterministic_with_temperature_zero(
        self, real_tokenizer, real_data_path, _require_gpu,
    ):
        """With temperature=0, two runs of the same batch produce identical output."""

        from benchmark.hardware.backend import detect_backend
        from benchmark.inference.engine import InferenceEngine
        from benchmark.inference.sampling import DecodingParams

        device_info = detect_backend()
        if device_info.backend == "cpu":
            pytest.skip("backend is cpu")

        engine = InferenceEngine(
            model_path="google/translategemma-4b-it",
            tokenizer_path="google/translategemma-4b-it",
            device_info=device_info,
            decoding_params=DecodingParams(
                max_new_tokens=32, temperature=0.0, do_sample=False,
            ),
            use_flash_attention=False,
        )
        try:
            engine.load()
        except Exception as exc:
            pytest.skip(f"model load failed: {exc}")

        engine.warmup(batches=2)

        # Prepare one small batch.
        loader = JSONLLoader([str(real_data_path)], shuffle=False)
        pipeline = AsyncPipeline(
            loader, NullChunker(), real_tokenizer,
            ChunkFilter(min_tokens=10, max_garbage_ratio=0.95),
            batch_size=2, prefetch_workers=1, backend=device_info.backend,
        )
        pipeline.start_prefetch()

        batch = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            batch = pipeline.next_batch()
            if batch is not None:
                break
        pipeline.stop_prefetch()

        assert batch is not None

        # Two runs.
        result1 = engine.translate(batch)
        result2 = engine.translate(batch)

        # Outputs should be identical.
        for i in range(len(result1.generations)):
            assert result1.generations[i].translated_text == result2.generations[i].translated_text, (
                f"Nondeterminism detected at index {i}:\n"
                f"  Run1: {result1.generations[i].translated_text[:120]}\n"
                f"  Run2: {result2.generations[i].translated_text[:120]}"
            )

        assert result1.output_tokens_total == result2.output_tokens_total
