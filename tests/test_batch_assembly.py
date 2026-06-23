"""Tests for batch assembly."""

import torch
import pytest
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
        # Strengthened assertions: verify content, not just shape.
        # item 0: [1,2,3] padded to len 4, mask [1,1,1,0]
        assert input_ids[0].tolist() == [1, 2, 3, 0]
        assert attention_mask[0].tolist() == [1, 1, 1, 0]
        # item 1: [4,5] padded to len 4, mask [1,1,0,0]
        assert input_ids[1].tolist() == [4, 5, 0, 0]
        assert attention_mask[1].tolist() == [1, 1, 0, 0]
        # item 2: [6,7,8,9] no padding needed, mask [1,1,1,1]
        assert input_ids[2].tolist() == [6, 7, 8, 9]
        assert attention_mask[2].tolist() == [1, 1, 1, 1]
        # lengths match input_lengths
        assert lengths == [3, 2, 4]
        # texts are passed through
        assert texts == ["hello world", "hi", "longer text here"]

    def test_collate_different_pad_token(self):
        """Pad token id other than 0 is respected."""
        a = BatchAssembler(pad_token_id=7)
        items = [("test", [1, 2, 3, 4, 5], 5), ("ok", [8, 9], 2)]
        input_ids, attention_mask, lengths, _ = a.collate(items)
        # First sequence padded with 7, not 0.
        assert input_ids[1].tolist() == [8, 9, 7, 7, 7]

    def test_clamp_batch_size(self):
        a = BatchAssembler(max_batch_size=64)
        assert a.clamp(32) == 32
        assert a.clamp(128) == 64
        assert a.clamp(0) == 1

    def test_collate_single_item(self):
        """Single item requires no padding."""
        a = BatchAssembler(pad_token_id=0)
        items = [("solo", [10, 20, 30], 3)]
        input_ids, attention_mask, lengths, texts = a.collate(items)
        assert input_ids.shape == (1, 3)
        assert input_ids[0].tolist() == [10, 20, 30]
        assert attention_mask.sum().item() == 3

    def test_collate_empty_raises(self):
        """Collating an empty item list should raise ValueError."""
        a = BatchAssembler(pad_token_id=0)
        with pytest.raises(ValueError, match="Empty item list"):
            a.collate([])
