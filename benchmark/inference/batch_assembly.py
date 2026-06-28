"""Dynamic batch collation for inference workloads.

Provides BatchAssembler, which pads variable-length token sequences to a
common length for batched model inference.  All padding uses a configurable
pad-token id and the batch size is clamped to a configurable maximum.
"""

from typing import Any

import torch

# ── Constants ──────────────────────────────────────────────────────────
DEFAULT_PAD_TOKEN_ID = 0
DEFAULT_MAX_BATCH_SIZE = 64


class BatchAssembler:
    """Collate variable-length token sequences into padded batch tensors.

    Pads shorter sequences with ``pad_token_id`` and produces an
    attention mask so the model ignores padding positions.  Batch sizes are
    clamped to ``max_batch_size`` by the ``clamp`` helper.

    Attributes
    ----------
    pad_token_id : int
        Token id used to fill padding positions (default 0).
    max_batch_size : int
        Upper bound for batch sizes emitted by ``clamp`` (default 64).
    """

    def __init__(self, pad_token_id: int = DEFAULT_PAD_TOKEN_ID,
                 max_batch_size: int = DEFAULT_MAX_BATCH_SIZE):
        """Initialize the assembler.

        Parameters
        ----------
        pad_token_id : int
            Token id used to pad shorter sequences to a uniform length.
        max_batch_size : int
            Maximum number of items the ``clamp`` method will allow.
        """
        self.pad_token_id = pad_token_id
        self.max_batch_size = max_batch_size

    def collate(self, items: list[tuple[str, list[int], int]]) -> tuple[Any, Any, list[int], list[str]]:
        """Collate a list of (text, token_ids, length) tuples into batch tensors.

        Parameters
        ----------
        items : list[tuple[str, list[int], int]]
            Each tuple contains:
            - ``text`` (str): The original input string.
            - ``token_ids`` (list[int]): The tokenized sequence.
            - ``length`` (int): The number of tokens in the sequence.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, list[int], list[str]]
            - ``input_ids``: Padded tensor of shape ``(batch_size, max_len)``
              with ``pad_token_id`` filling the trailing positions.
            - ``attention_mask``: Binary tensor of shape ``(batch_size, max_len)``
              where 1 marks real tokens and 0 marks padding.
            - ``lengths``: Original sequence lengths (list[int]).
            - ``texts``: Original input strings (list[str]).

        Raises
        ------
        ValueError
            If ``items`` is empty.  An empty batch cannot be collated into valid
            model input tensors and indicates a pipeline bug upstream.
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
        """Clamp a requested batch size to the allowed range [1, max_batch_size].

        Parameters
        ----------
        batch_size : int
            The desired batch size (may be 0 or negative).

        Returns
        -------
        int
            A batch size guaranteed to be between 1 and ``self.max_batch_size``
            inclusive.
        """
        return max(1, min(batch_size, self.max_batch_size))
