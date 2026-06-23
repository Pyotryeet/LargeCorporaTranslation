"""Pipeline integration test: load → tokenize → batch → assemble → padding."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from benchmark.data.loader import JSONLLoader
from benchmark.data.chunker import NullChunker
from benchmark.data.filters import ChunkFilter
from benchmark.data.pipeline import AsyncPipeline, PinnedBufferPool


class TestPipelineE2E:
    def test_load_to_batch_flow(self, real_tokenizer):
        """Data flows correctly: JSONL → chunk → filter → tokenize → batch."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "test.jsonl"
            texts = [
                "This is a sample English sentence for translation testing.",
                "Another example text to verify the pipeline works correctly.",
                "Short.",
                "Machine translation quality assessment requires careful evaluation.",
            ]
            inp.write_text("\n".join(json.dumps({"text": t}) for t in texts))

            loader = JSONLLoader([str(inp)], shuffle=False)
            chunker = NullChunker()
            filt = ChunkFilter(min_tokens=2, max_garbage_ratio=0.95)

            pipeline = AsyncPipeline(
                loader, chunker, real_tokenizer, filt,
                batch_size=2, prefetch_workers=1, backend="cpu",
            )
            pipeline.start_prefetch()

            batches = []
            for _ in range(3):
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    batch = pipeline.next_batch()
                    if batch is not None:
                        break
                    time.sleep(0.05)
                if batch is not None:
                    batches.append(batch)

            pipeline.stop_prefetch()

            assert len(batches) >= 1, "No batches produced"
            for batch in batches:
                assert batch.input_ids.shape[0] > 0, "Batch should have inputs"
                assert batch.attention_mask.shape[0] > 0
                assert len(batch.raw_texts) > 0

    def test_batch_padding_correct(self, real_tokenizer):
        """Shorter sequences are padded to max_len within a batch."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "test.jsonl"
            inp.write_text(
                json.dumps({"text": "Short."}) + "\n" +
                json.dumps({"text": "A much longer text that should produce more tokens."}) + "\n"
            )

            loader = JSONLLoader([str(inp)], shuffle=False)
            chunker = NullChunker()
            filt = ChunkFilter(min_tokens=1, max_garbage_ratio=0.95)

            pipeline = AsyncPipeline(
                loader, chunker, real_tokenizer, filt,
                batch_size=2, prefetch_workers=1, backend="cpu",
            )
            pipeline.start_prefetch()

            deadline = time.monotonic() + 5
            batch = None
            while time.monotonic() < deadline:
                batch = pipeline.next_batch()
                if batch is not None:
                    break
                time.sleep(0.05)
            pipeline.stop_prefetch()

            assert batch is not None
            assert batch.input_ids.shape[0] == 2
            max_len = batch.input_ids.shape[1]

            # Verify: shorter sequence should have pad tokens beyond its length.
            short_tokens = len(real_tokenizer.encode(batch.raw_texts[0]))
            long_tokens = len(real_tokenizer.encode(batch.raw_texts[1]))
            if short_tokens < long_tokens:
                short_idx = 0
            else:
                short_idx = 1

            # After the actual tokens, the rest should be pad (0).
            pinned = batch.attention_mask[short_idx]
            assert pinned.sum() < max_len or short_tokens == long_tokens, (
                f"Short sequence should be padded; got mask sum = {pinned.sum().item()} / max_len = {max_len}"
            )

    def test_draining_stops_eventually(self, real_tokenizer):
        """Pipeline stops when data is exhausted."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "small.jsonl"
            inp.write_text(json.dumps({"text": "Only one document."}) + "\n")

            loader = JSONLLoader([str(inp)], shuffle=False)
            chunker = NullChunker()
            filt = ChunkFilter(min_tokens=1, max_garbage_ratio=0.95)

            pipeline = AsyncPipeline(
                loader, chunker, real_tokenizer, filt,
                batch_size=4, prefetch_workers=1, backend="cpu",
            )
            pipeline.start_prefetch()

            # Should produce a small batch then drain.
            deadline = time.monotonic() + 5
            batch = None
            while time.monotonic() < deadline:
                batch = pipeline.next_batch()
                if batch is not None:
                    break
                time.sleep(0.05)
            assert batch is not None

            # Next call should return None since data is drained.
            deadline2 = time.monotonic() + 5
            batch2 = None
            while time.monotonic() < deadline2:
                batch2 = pipeline.next_batch()
                if batch2 is not None or pipeline.draining():
                    break
                time.sleep(0.05)
            assert batch2 is None or pipeline.draining()

            pipeline.stop_prefetch()

    def test_loader_tracks_doc_id(self):
        """JSONLLoader tracks _current_file and _current_doc_id."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "test.jsonl"
            inp.write_text(json.dumps({"text": "Test document."}) + "\n")
            loader = JSONLLoader([str(inp)], shuffle=False)
            for doc_id, fname, text in loader.iter_documents():
                assert text == "Test document."
                assert loader._current_file != ""
                assert loader._current_doc_id >= 0
                break

    def test_loader_seek_skips(self):
        """seek_to skips the specified number of documents."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            f1 = tmpdir / "file_a.jsonl"
            f2 = tmpdir / "file_b.jsonl"
            f1.write_text(
                json.dumps({"text": "Skip this one."}) + "\n"
            )
            f2.write_text(
                json.dumps({"text": "First in B."}) + "\n" +
                json.dumps({"text": "Second in B."}) + "\n"
            )
            loader = JSONLLoader([str(f1), str(f2)], shuffle=False)

            # Seek past first document.
            loader.seek_to(1)
            results = list(loader.iter_documents())
            assert len(results) == 2, f"Expected 2 docs from file_b, got {len(results)}"
            _, fname0, text0 = results[0]
            assert "First in B" in text0
            assert fname0 == "file_b.jsonl"


class TestPinnedBufferPool:
    import torch

    def test_pool_acquire_and_release(self):
        pool = PinnedBufferPool(max_batch_size=8, max_seq_len=512, pool_size=2)
        ids1, mask1 = pool.acquire()
        assert ids1.shape == (8, 512)
        assert mask1.shape == (8, 512)
        # Pinning only works on CUDA; on MPS/CPU pinned memory is not guaranteed.
        if TestPinnedBufferPool.torch.cuda.is_available():
            assert ids1.is_pinned(), "Pinned memory should be enabled on CUDA"

        pool.release(ids1, mask1)
        ids2, mask2 = pool.acquire()
        # Should be the SAME tensor (reused).
        assert ids2.data_ptr() == ids1.data_ptr()

    def test_pool_hit_rate(self):
        pool = PinnedBufferPool(max_batch_size=4, max_seq_len=64, pool_size=2)
        ids, mask = pool.acquire()
        pool.release(ids, mask)
        ids, mask = pool.acquire()  # hit
        assert pool.hit_rate == 0.5  # 1 miss, 1 hit


class TestModelMaxLengthOverflow:
    """The pipeline must cap max_input_tokens to _DEFAULT_MAX_SEQ_LEN (2048)
    even when the tokenizer reports a sentinel value (~10^30) for
    "unlimited context".  Passing a multi-decillion value to the C++
    SentencePiece extension causes OverflowError: int too big to convert.
    """

    # The exact sentinel used by some tokenizers (2^100).
    SENTINEL_MAX_LENGTH = 1000000000000000019884624838656

    def test_sentinel_capped_during_init(self, real_tokenizer):
        """max_input_tokens is clamped to _DEFAULT_MAX_SEQ_LEN at __init__."""
        from benchmark.data.loader import JSONLLoader
        from benchmark.data.chunker import NullChunker
        from benchmark.data.filters import ChunkFilter

        # Monkey-patch the tokenizer to report a sentinel model_max_length.
        original_max_length = real_tokenizer.model_max_length
        real_tokenizer.model_max_length = self.SENTINEL_MAX_LENGTH

        try:
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                inp = tmpdir / "overflow_test.jsonl"
                inp.write_text(
                    '{"text": "Quick sanity check for overflow guard."}' + "\n"
                )

                loader = JSONLLoader([str(inp)], shuffle=False)
                chunker = NullChunker()
                filt = ChunkFilter(min_tokens=1, max_garbage_ratio=0.95)

                pipeline = AsyncPipeline(
                    loader, chunker, real_tokenizer, filt,
                    batch_size=1, prefetch_workers=1, backend="cpu",
                )

                # The pipeline must cap to _DEFAULT_MAX_SEQ_LEN (2048),
                # NOT the sentinel value of ~10^30.
                from benchmark.config.constants import DEFAULT_MAX_SEQ_LEN
                assert pipeline.max_input_tokens == DEFAULT_MAX_SEQ_LEN, (
                    f"max_input_tokens should be capped to {DEFAULT_MAX_SEQ_LEN}, "
                    f"got {pipeline.max_input_tokens}"
                )

                # Verify the pipeline actually works without overflowing.
                pipeline.start_prefetch()
                import time
                deadline = time.monotonic() + 10
                batch = None
                while time.monotonic() < deadline:
                    batch = pipeline.next_batch()
                    if batch is not None:
                        break
                    time.sleep(0.05)
                pipeline.stop_prefetch()

                assert batch is not None, (
                    "Pipeline failed to produce a batch with sentinel model_max_length"
                )
                assert batch.input_ids.shape[1] <= DEFAULT_MAX_SEQ_LEN, (
                    f"Batch seq_len {batch.input_ids.shape[1]} exceeds "
                    f"DEFAULT_MAX_SEQ_LEN {DEFAULT_MAX_SEQ_LEN}"
                )
        finally:
            real_tokenizer.model_max_length = original_max_length

    def test_sentinel_cannot_overflow_c_extension(self, real_tokenizer):
        """Sending the sentinel value to the C extension would overflow.
        Verify that when max_input_tokens is capped, tok.encode() succeeds
        without OverflowError."""
        from benchmark.config.constants import DEFAULT_MAX_SEQ_LEN

        original_max_length = real_tokenizer.model_max_length
        real_tokenizer.model_max_length = self.SENTINEL_MAX_LENGTH

        try:
            import copy
            tok = copy.deepcopy(real_tokenizer)

            # Simulate what the pipeline does: cap before passing to encode.
            capped = min(tok.model_max_length, DEFAULT_MAX_SEQ_LEN)
            token_ids = tok.encode(
                "Test input that should not overflow.",
                add_special_tokens=True,
                truncation=True,
                max_length=capped,
            )
            # Conversion to native ints must also succeed.
            safe = [int(t) for t in token_ids]
            assert len(safe) > 0
        finally:
            real_tokenizer.model_max_length = original_max_length
