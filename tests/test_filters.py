"""Tests for chunk quality filters."""

from benchmark.data.filters import ChunkFilter, FilterStats


class TestChunkFilter:
    def test_valid_chunk_passes(self):
        f = ChunkFilter(min_tokens=10)
        assert f.should_keep("This is a valid English input text for testing.", 15) is True
        assert f.stats.passed == 1

    def test_too_short_chunk_rejected(self):
        f = ChunkFilter(min_tokens=10)
        assert f.should_keep("hi", 2) is False
        assert f.stats.rejected_too_short == 1

    def test_too_long_chunk_rejected(self):
        f = ChunkFilter(max_tokens=100)
        assert f.should_keep("x " * 200, 150) is False
        # NOTE: ChunkFilter tracks both "too short" and "too long" under the
        # single `rejected_too_short` counter — the filter lumps all token-count
        # rejections (outside [min_tokens, max_tokens]) into one bucket.
        assert f.stats.rejected_too_short == 1

    def test_garbage_text_rejected(self):
        f = ChunkFilter(max_garbage_ratio=0.3)
        # Mostly non-ASCII characters will exceed 0.3
        garbage = "\ufffd" * 100 + "aa"  # 100 unicode replacement chars + 2 ASCII = 100/102 > 0.3
        # NOTE: token_count (20) is caller-provided; the filter trusts it rather
        # than recomputing. This avoids coupling the filter to a tokenizer.
        assert f.should_keep(garbage, 20) is False
        assert f.stats.rejected_garbage == 1

    def test_normal_text_not_garbage(self):
        f = ChunkFilter(max_garbage_ratio=0.5)
        assert f.should_keep("Hello world, this is normal ASCII text.", 15) is True

    def test_filter_stats_tracks_correctly(self):
        f = ChunkFilter(min_tokens=10)
        f.should_keep("hi", 2)
        f.should_keep("hello world test text here", 15)
        f.should_keep("hello world test text here two", 16)
        assert f.stats.total_chunks == 3
        assert f.stats.passed == 2
        assert f.stats.rejected_too_short == 1
        assert round(f.stats.pass_rate, 2) == round(2/3, 2)

    def test_stats_to_dict(self):
        f = ChunkFilter()
        d = f.stats.to_dict()
        assert "total_chunks" in d
        assert "passed" in d
        assert "pass_rate" in d


class TestFilterStats:
    def test_default_stats(self):
        s = FilterStats()
        assert s.pass_rate == 0.0
        assert s.rejected == 0

    def test_rejected_sum(self):
        s = FilterStats(rejected_too_short=3, rejected_garbage=2, rejected_language=1)
        assert s.rejected == 6

    # ── Edge case tests ──

    def test_empty_chunk_sentinel_token_count(self):
        """A zero-length input with token_count=0 should be rejectable."""
        f = ChunkFilter(min_tokens=1)
        assert f.should_keep("", 0) is False
        assert f.stats.rejected_too_short == 1

    def test_filter_with_default_constructor_accepts_all(self):
        """ChunkFilter() with defaults accepts text with sufficient tokens."""
        f = ChunkFilter()
        # Default min_tokens=10 — text with >=10 tokens should pass.
        assert f.should_keep("anything at all", 10) is True

    def test_max_garbage_ratio_zero_allows_pure_ascii_only(self):
        """max_garbage_ratio=0 rejects any text with non-ASCII chars."""
        f = ChunkFilter(min_tokens=1, max_garbage_ratio=0.0)
        assert f.should_keep("PureASCII", 5) is True
        # Text with non-ASCII chars should be rejected when ratio is 0.
        assert f.should_keep("\U00010000 extra text padding padding", 10) is False

    def test_stats_reset(self):
        """reset_stats zeros all counters."""
        f = ChunkFilter(min_tokens=5)
        f.should_keep("hello world test", 10)
        f.should_keep("x", 1)
        assert f.stats.total_chunks == 2
        f.reset_stats()
        assert f.stats.total_chunks == 0
        assert f.stats.passed == 0
        assert f.stats.rejected == 0
