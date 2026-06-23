"""Tests for batch assembly."""

from benchmark.inference.batch_assembly import BatchAssembler


class TestBatchAssembler:
    def test_collate_pads_to_max_length(self):
        a = BatchAssembler(pad_token_id=0)
        items = [
            ("hello world", [1, 2, 3], 3),
            ("hi", [4, 5], 2),
            ("longer text here", [6, 7, 8, 9], 4),
        ]
        input_ids, attention_mask, lengths, texts = a.collate(items)
        assert input_ids.shape == (3, 4)  # 3 items, max_len=4
        assert attention_mask.shape == (3, 4)

    def test_clamp_batch_size(self):
        a = BatchAssembler(max_batch_size=64)
        assert a.clamp(32) == 32
        assert a.clamp(128) == 64
        assert a.clamp(0) == 1
