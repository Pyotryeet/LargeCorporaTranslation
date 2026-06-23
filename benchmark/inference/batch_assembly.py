"""Dynamic batch collation."""

from typing import Any

import torch

# ── Constants ──────────────────────────────────────────────────────────
DEFAULT_PAD_TOKEN_ID = 0
DEFAULT_MAX_BATCH_SIZE = 64


class BatchAssembler:
    def __init__(self, pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
                 max_batch_size: int = DEFAULT_MAX_BATCH_SIZE):
        self.pad_token_id = pad_token_id
        self.max_batch_size = max_batch_size

    def collate(self, items: list[tuple[str, list[int], int]]) -> tuple[Any, Any, list[int], list[str]]:
        """Collate a list of (text, token_ids, length) tuples into batch tensors.

        Returns (input_ids, attention_mask, lengths, texts).

        Raises
        ------
        ValueError
            If ``items`` is empty — an empty batch cannot be collated into
            valid model input tensors and indicates a pipeline bug upstream.
        """
        if not items:
            raise ValueError(
                "BatchAssembler.collate() received an empty items list.  "
                "An empty batch cannot be collated into valid model input "
                "tensors.  Check the pipeline for a bug that produces "
                "zero-element batches."
            )

        texts = [i[0] for i in items]
        token_lists = [i[1] for i in items]
        lengths = [i[2] for i in items]
        max_len = max(lengths)
        bs = len(items)
        input_ids = torch.full((bs, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        for i, tids in enumerate(token_lists):
            input_ids[i, :len(tids)] = torch.tensor(tids, dtype=torch.long)
            attention_mask[i, :len(tids)] = 1
        return input_ids, attention_mask, lengths, texts

    def clamp(self, batch_size: int) -> int:
        return max(1, min(batch_size, self.max_batch_size))
