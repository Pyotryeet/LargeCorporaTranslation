"""CUDA Graph capture and replay for the decode loop (C1-CUDA).

.. warning::
   **DEPRECATED MODULE (2026-06-23).**  The CUDA graph infrastructure in this
   module is captured but **never replayed** on the hot path.  The decode loop
   (``AutoregressiveBackend._extreme_decode``) intentionally uses the standard
   model forward pass because the graphs do not support accumulated KV-cache
   inputs.  This module is retained for reference and future use once static
   KV-cache buffers are implemented.  Do NOT add new callers — any code that
   instantiates ``CUDAGraphDecoder`` or ``CUDAGraphPool`` will pay the capture
   cost without ever benefiting from the graph.

Eliminates per-token kernel launch overhead by capturing the entire
single-step decode forward pass as a CUDA graph.  Once captured, each
decode iteration becomes a single ``graph.replay()`` call instead of
hundreds of individual kernel launches.

Supports up to ``max_batch_size`` sequences and ``max_seq_len`` tokens.
Sequences shorter than the max are handled via a padding mask.

Reference:
  - NVIDIA CUDA Graphs documentation
  - vLLM CUDA graph pool design
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deprecation guard — emit once per process
# ---------------------------------------------------------------------------

_DEPRECATION_WARNED = False


def _warn_deprecated() -> None:
    global _DEPRECATION_WARNED
    if not _DEPRECATION_WARNED:
        _DEPRECATION_WARNED = True
        warnings.warn(
            "benchmark.hardware.cuda_graphs is DEPRECATED.  The CUDA graph "
            "infrastructure is captured but NEVER replayed on the hot path "
            "(see AutoregressiveBackend._extreme_decode).  Instantiating "
            "CUDAGraphDecoder or CUDAGraphPool incurs the capture cost "
            "without any benefit.  This module will be removed in a future "
            "version once static KV-cache buffers are implemented.",
            FutureWarning,  # FutureWarning is displayed by default (unlike DeprecationWarning)
            stacklevel=2,
        )


# Emit deprecation warning on import so that importers see it even if they
# never instantiate a class (e.g. static inspection, linting, autodoc).
_DEPRECATION_IMPORT_WARNED = False
if not _DEPRECATION_IMPORT_WARNED:
    _DEPRECATION_IMPORT_WARNED = True
    warnings.warn(
        "Importing benchmark.hardware.cuda_graphs is DEPRECATED.  "
        "This module will be removed once static KV-cache buffers are implemented.",
        FutureWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Module-level constants (used as constructor defaults and warm-up params)
# ---------------------------------------------------------------------------

DEFAULT_MAX_BATCH_SIZE: int = 128
DEFAULT_MAX_SEQ_LEN: int = 2048
DEFAULT_HIDDEN_SIZE: int = 3840       # TranslateGemma 4B
DEFAULT_NUM_KV_HEADS: int = 4         # TranslateGemma 4B GQA
DEFAULT_HEAD_DIM: int = 256           # 3840 / 16 heads
WARMUP_ITERATIONS: int = 3            # Number of warm-up forward passes before capture

# vLLM-style batch-size progression: powers of 2 plus useful intermediates.
CUDA_GRAPH_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


class CUDAGraphDecoder:
    """CUDA graph–backed decoder for token-by-token generation.

    Captures one forward pass through the model (a single decode step)
    and replays it for each token.  Requires CUDA >= 11.4 and a model
    that can be traced (no data-dependent control flow in forward).

    .. warning::

       **KV-cache limitation**: The captured graph does NOT include
       ``past_key_values`` as a static input.  This means:

       - Each ``replay()`` runs a single-token forward pass WITHOUT
         accumulated KV-cache from prior decode steps — the model sees
         only the current token, not the full conversation history.
       - The graph output (``_static_output``) IS a ``CausalLMOutputWithPast``
         and DOES contain ``.past_key_values`` from the single-step forward,
         but those KV entries only cover the current token.
       - For correct multi-step autoregressive decoding, each step MUST
         pass the accumulated ``past_key_values`` from all prior steps
         as input to the model — which the current graph capture does not
         support.

       **To re-enable**: pre-allocate static KV-cache buffers of shape
       ``(batch, num_kv_heads, max_seq_len, head_dim)`` for every layer,
       include them as static inputs during ``capture()``, and use a
       valid-length mask (integer tensor tracking actual sequence length
       per batch element) inside the attention kernel instead of relying
       on dynamic KV-cache tensor shapes.

       Currently the hot path (``AutoregressiveBackend._extreme_decode``)
       intentionally skips this graph and uses the standard model forward
       pass instead.  The graph infrastructure is preserved for future use
       once static KV-cache buffers are implemented.
    """

    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
        head_dim: int = DEFAULT_HEAD_DIM,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        _warn_deprecated()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAGraphDecoder requires CUDA")

        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False
        self._static_input_ids: Optional[torch.Tensor] = None
        self._static_attention_mask: Optional[torch.Tensor] = None
        self._static_position_ids: Optional[torch.Tensor] = None
        # Type annotation — this is set to the full model output object
        # (CausalLMOutputWithPast) after capture/replay, NOT a plain
        # torch.Tensor.  Access .logits and .past_key_values on it.
        self._static_output: Any = None

        # Track capture shapes so we can validate on replay.
        self._capture_batch_size: int = 0
        self._capture_seq_len: int = 0

    @property
    def is_captured(self) -> bool:
        return self._captured

    def capture(self, batch_size: int, seq_len: int, device: torch.device | int = 0) -> None:
        """Capture the CUDA graph for a specific (batch_size, seq_len) shape.

        Parameters
        ----------
        batch_size : int
            Number of sequences to support.  Must not exceed ``max_batch_size``.
        seq_len : int
            Total sequence length (prompt + generated so far).  Must not
            exceed ``max_seq_len``.
        device :
            CUDA device for the graph (default ``cuda:0``).
        """
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size {batch_size} exceeds max_batch_size {self.max_batch_size}"
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        device = torch.device(device) if isinstance(device, int) else device
        logger.info(
            "Capturing CUDA graph for batch_size=%d, seq_len=%d...", batch_size, seq_len,
        )

        # -- Allocate static buffers --
        self._static_input_ids = torch.full(
            (batch_size, 1), 0, dtype=torch.long, device=device,
        )
        self._static_attention_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.long, device=device,
        )
        self._static_position_ids = torch.arange(
            seq_len - 1, seq_len, dtype=torch.long, device=device,
        ).unsqueeze(0).expand(batch_size, -1)

        try:
            # -- Warm-up (several iterations to prime CUDA caches) --
            for _ in range(WARMUP_ITERATIONS):
                with torch.no_grad():
                    self.model(
                        input_ids=self._static_input_ids,
                        attention_mask=self._static_attention_mask,
                        position_ids=self._static_position_ids,
                        use_cache=True,
                    )
            torch.cuda.synchronize()

            # -- Capture --
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                with torch.no_grad():
                    self._static_output = self.model(
                        input_ids=self._static_input_ids,
                        attention_mask=self._static_attention_mask,
                        position_ids=self._static_position_ids,
                        use_cache=True,
                    )

            self._captured = True
            self._capture_batch_size = batch_size
            self._capture_seq_len = seq_len
            logger.info("CUDA graph captured successfully (bs=%d, seq=%d)", batch_size, seq_len)

        except Exception:
            # Capture failed — release static buffers to prevent GPU
            # memory leak.  The graph object may also hold GPU memory.
            logger.warning(
                "CUDA graph capture failed — releasing static buffers",
                exc_info=True,
            )
            self._static_input_ids = None
            self._static_attention_mask = None
            self._static_position_ids = None
            self._static_output = None
            if self._graph is not None:
                try:
                    self._graph.reset()
                except Exception:
                    pass  # best-effort CUDA-side cleanup
                del self._graph
                self._graph = None
            self._captured = False
            torch.cuda.empty_cache()
            raise

    def replay(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Any:
        """Replay the captured graph with updated inputs.

        The caller is responsible for copying new token IDs into the
        static buffers BEFORE calling ``replay()``.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape ``[batch_size, 1]`` — the next token for each sequence.
        attention_mask : torch.Tensor
            Shape ``[batch_size, seq_len]`` — updated causal mask.
        position_ids : torch.Tensor, optional
            Shape ``[batch_size, 1]`` — position indices.

        Returns
        -------
        CausalLMOutputWithPast
            The full model output object.  Access ``.logits`` for the
            next-token prediction and ``.past_key_values`` for the
            updated KV-cache.

            .. warning::

               The ``.past_key_values`` from graph replay only covers
               the single token fed into this forward pass.  It does NOT
               contain accumulated KV-cache from prior decode steps
               because the graph does not accept ``past_key_values`` as
               an input.  See the class docstring for details.
        """
        if not self._captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        bs, _ = input_ids.shape
        if bs > self._capture_batch_size:
            raise RuntimeError(
                f"Batch size {bs} exceeds captured size {self._capture_batch_size}"
            )

        # Copy updated inputs into static buffers.
        self._static_input_ids[:bs].copy_(input_ids)
        self._static_attention_mask[:bs, :attention_mask.shape[1]].copy_(attention_mask)
        if position_ids is not None:
            self._static_position_ids[:bs].copy_(position_ids)

        self._graph.replay()
        return self._static_output

    def release(self) -> None:
        """Free the CUDA graph memory."""
        if self._graph is not None:
            del self._graph
        self._graph = None
        self._captured = False
        self._static_input_ids = None
        self._static_attention_mask = None
        self._static_position_ids = None
        self._static_output = None
        logger.info("CUDA graph released")


class CUDAGraphPool:
    """Pool of pre-captured CUDA graphs for different batch sizes.

    vLLM-style: captures a graph for each supported batch size (powers of 2
    plus configurable intermediate sizes) so the decode loop can pick the
    closest one and pad to it.  This avoids paying capture cost mid-run.

    .. warning::
       **DEPRECATED.**  This pool is never used on the hot path.
    """

    def __init__(
        self,
        model: nn.Module,
        batch_sizes: Optional[List[int]] = None,
        max_seq_len: int = 2048,
        **kwargs: Any,
    ) -> None:
        _warn_deprecated()
        self.model: nn.Module = model
        self.max_seq_len: int = max_seq_len
        self.kwargs: Dict[str, Any] = kwargs
        self._graphs: Dict[int, CUDAGraphDecoder] = {}

        if batch_sizes is None:
            batch_sizes = list(CUDA_GRAPH_BATCH_SIZES)
        self.batch_sizes: List[int] = sorted(batch_sizes)

    def get_or_capture(
        self, batch_size: int, seq_len: int, device: torch.device | int = 0,
    ) -> CUDAGraphDecoder:
        """Return a graph that can handle *batch_size*, capturing if needed.

        Pads up to the next supported batch size.
        """
        # Find next supported batch size >= batch_size
        bs_key = 1
        for bs in self.batch_sizes:
            if bs >= batch_size:
                bs_key = bs
                break

        if bs_key not in self._graphs:
            graph = CUDAGraphDecoder(
                self.model,
                max_batch_size=bs_key,
                max_seq_len=self.max_seq_len,
                **self.kwargs,
            )
            graph.capture(batch_size=bs_key, seq_len=seq_len, device=device)
            self._graphs[bs_key] = graph

        return self._graphs[bs_key]

    def release_all(self) -> None:
        for graph in self._graphs.values():
            graph.release()
        self._graphs.clear()

    def cleanup(self) -> None:
        """Reset all captured graphs and empty the cache.

        Unlike release_all(), this calls reset() on each graph's underlying
        CUDA graph object (if available), freeing the CUDA-side resources
        before releasing the Python wrappers and clearing the pool.
        """
        for graph in self._graphs.values():
            if graph._graph is not None:
                graph._graph.reset()
            graph.release()
        self._graphs.clear()
        torch.cuda.synchronize()
        logger.info("CUDAGraphPool cleaned up: all graphs reset and cleared")
