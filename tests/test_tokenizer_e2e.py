"""E2E test that validates the pipeline with the REAL HuggingFace tokenizer.

Tests real-world scenarios: numpy-backed token IDs, token ID conversion,
insane model_max_length sentinel, and corrupt/damaged input files.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from benchmark.data.loader import JSONLLoader
from benchmark.data.chunker import NullChunker
from benchmark.data.filters import ChunkFilter
from benchmark.data.pipeline import AsyncPipeline

# numpy is needed only for TokenIdConversion tests — import is guarded so
# the test file remains importable even in minimal environments.
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


class TestPipelineHandlesRealTokenizer:
    """The pipeline must produce correct batches using the real tokenizer."""

    def test_real_tokenizer_produces_batches(self, real_tokenizer):
        """Real HF tokenizer → pipeline produces valid batches with proper shapes."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "test.jsonl"
            inp.write_text(
                json.dumps({"text": "Hello world test sentence."}) + "\n" +
                json.dumps({"text": "Another example for translation."}) + "\n" +
                json.dumps({"text": "Third text to fill a batch."}) + "\n" +
                json.dumps({"text": "Fourth example goes here."}) + "\n"
            )

            loader = JSONLLoader([str(inp)], shuffle=False)
            chunker = NullChunker()
            filt = ChunkFilter(min_tokens=1, max_garbage_ratio=0.95)

            pipeline = AsyncPipeline(
                loader, chunker, real_tokenizer, filt,
                batch_size=2, prefetch_workers=1, backend="cpu",
            )
            pipeline.start_prefetch()

            deadline = time.monotonic() + 10
            batch = None
            while time.monotonic() < deadline:
                batch = pipeline.next_batch()
                if batch is not None:
                    break
                time.sleep(0.05)

            pipeline.stop_prefetch()

            assert batch is not None, "Pipeline should produce batches with real tokenizer"
            # Verify tokens are stored correctly in tensor.
            assert batch.input_ids.shape[0] >= 1
            assert batch.input_ids.shape[1] > 0
            # First token should not be pad (0) for a real tokenizer.
            assert batch.input_ids[0, 0].item() != 0, "First token should not be pad"


class TestTokenIdConversion:
    """Token IDs from any source must be convertible to native Python ints."""

    def test_real_tokenizer_returns_native_ints(self, real_tokenizer):
        """Verify that the real HF tokenizer returns native Python ints."""
        result = real_tokenizer.encode("Hello world", add_special_tokens=True)
        assert isinstance(result, list)
        assert len(result) > 0
        for t in result:
            assert isinstance(t, int), (
                f"Token {t} has type {type(t)}, expected int"
            )

    def test_numpy_int32_converts_to_python_int(self):
        """int() on numpy.int32 must produce native Python int."""
        pytest.importorskip("numpy")
        import numpy as np
        ids = [np.int32(42), np.int32(0), np.int32(256_000)]
        native = [int(t) for t in ids]
        assert all(isinstance(t, int) for t in native)
        assert native == [42, 0, 256_000]

    def test_numpy_int64_converts_to_python_int(self):
        """int() on numpy.int64 must produce native Python int."""
        pytest.importorskip("numpy")
        import numpy as np
        ids = [np.int64(2**40), np.int64(2**50)]
        native = [int(t) for t in ids]
        assert all(isinstance(t, int) for t in native)
        assert native == [2**40, 2**50]

    def test_mixed_numpy_and_python_ints(self):
        """Mixed list of numpy and Python ints must all convert to Python int."""
        pytest.importorskip("numpy")
        import numpy as np
        ids = [np.int32(10), 20, np.int64(30), np.int32(40)]
        native = [int(t) for t in ids]
        assert native == [10, 20, 30, 40]
        assert all(isinstance(t, int) for t in native)


class TestPipelineWithCorruptData:
    """The pipeline must handle corrupt/malformed data gracefully."""

    def test_corrupt_jsonl_line_is_handled(self, real_tokenizer):
        """A JSONL file with a malformed line should not crash the pipeline."""
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "corrupt.jsonl"
            # Write a valid line, then a corrupt line, then another valid line.
            inp.write_text(
                json.dumps({"text": "Valid line one."}) + "\n" +
                "this is not valid json at all\n" +
                json.dumps({"text": "Valid line two."}) + "\n"
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

            # Pipeline should not crash and should produce at least the valid lines.
            assert batch is not None, "Pipeline should handle corrupt JSONL gracefully"


class TestPipelineMaxLengthOverflow:
    """The exact crash from production: tokenizer.model_max_length = 10^30.

    TranslateGemma 4B reports model_max_length = 1000000000000000019884624838656
    (a sentinel for "unlimited context").  The pipeline MUST cap this before
    passing it as max_length to the tokenizer C extension, which overflows.
    """

    def test_insane_model_max_length_does_not_crash_pipeline(self, real_tokenizer):
        """10^30 model_max_length → pipeline must cap it and produce batches."""

        # The real tokenizer's model_max_length IS 10^30 — verify that fact.
        insane_value = real_tokenizer.model_max_length
        assert insane_value > 10_000_000, (
            f"Expected insane model_max_length, got {insane_value}"
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            inp = tmpdir / "gemma.jsonl"
            inp.write_text(
                json.dumps({"text": "Hello world."}) + "\n" +
                json.dumps({"text": "Another sentence for translation."}) + "\n"
            )

            loader = JSONLLoader([str(inp)], shuffle=False)
            chunker = NullChunker()
            filt = ChunkFilter(min_tokens=1, max_garbage_ratio=0.95)

            pipeline = AsyncPipeline(
                loader, chunker, real_tokenizer, filt,
                batch_size=2, prefetch_workers=1, backend="cpu",
            )
            # Verify the pipeline capped the insane value.
            assert pipeline.max_input_tokens <= 2048, (
                f"max_input_tokens must be capped, got {pipeline.max_input_tokens}"
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

            assert batch is not None, (
                "Pipeline should produce batches even with 10^30 model_max_length"
            )
            assert batch.input_ids.shape[0] >= 1
