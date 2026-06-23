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
        assert f.stats.rejected_too_short == 1

    def test_garbage_text_rejected(self):
        f = ChunkFilter(max_garbage_ratio=0.3)
        # Mostly non-ASCII characters will exceed 0.3
        garbage = "\ufffd" * 100 + "aa"  # 100 unicode replacement chars + 2 ASCII = 100/102 > 0.3
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
