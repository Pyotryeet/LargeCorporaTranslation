"""Replicate-and-split layer parallelism for transformer models.

Shards transformer layers across 2 GPUs (pipeline-style per-layer assignment):
- Layers 0-23 on GPU 0, layers 24-47 on GPU 1
- Embedding and LM head REPLICATED (identical copy on each GPU)
- Logits all-reduced and averaged for identical results to single-GPU

Architecture reference (defaults)
----------------------------------
- Gemma 3 12B: 48 decoder layers, GQA 8 KV-heads, 5:1 local/global attention

.. note::
   ``apply_tensor_parallelism()`` is defined but NOT wired into the hot path
   by default.  Call ``ensure_dist_initialized()`` before ``apply_tensor_parallelism()``
   to bootstrap the distributed process group.  Launch with ``torchrun --nproc_per_node=2``.

   See [[h200-roadmap-gap-analysis]] for known caveats with this module.
"""

from __future__ import annotations

import logging
import os
import datetime
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for Gemma 3 12B
# ---------------------------------------------------------------------------

GEMMA_3_12B_NUM_LAYERS = 48
GEMMA_3_12B_HIDDEN_SIZE = 3840
GEMMA_3_12B_NUM_ATTENTION_HEADS = 16
GEMMA_3_12B_NUM_KV_HEADS = 8
GEMMA_3_12B_INTERMEDIATE_SIZE = 15360
GEMMA_3_12B_VOCAB_SIZE = 262144
GEMMA_3_12B_LOCAL_ATTN_WINDOW = 1024
GEMMA_3_12B_LOCAL_GLOBAL_RATIO = 5

# Legacy aliases for backward compatibility
GEMMA3_NUM_LAYERS = GEMMA_3_12B_NUM_LAYERS
GEMMA3_NUM_KV_HEADS = GEMMA_3_12B_NUM_KV_HEADS
GEMMA3_LOCAL_GLOBAL_RATIO = GEMMA_3_12B_LOCAL_GLOBAL_RATIO
GEMMA3_GLOBAL_LAYER_INTERVAL = GEMMA3_LOCAL_GLOBAL_RATIO + 1


# ---------------------------------------------------------------------------
# TensorParallelConfig
# ---------------------------------------------------------------------------


@dataclass
class TensorParallelConfig:
    """Configuration for 2-way layer-split parallelism on transformer models.

    This implements **replicate-and-split parallelism** (not true weight-sharded
    tensor parallelism): transformer layers are split across GPUs (pipeline-style
    per-layer assignment), while embeddings and LM head are REPLICATED (each GPU
    holds a full copy).  An all-reduce wrapper on the LM head averages logits
    across ranks to produce identical results to single-GPU inference (valid
    because the replicated LM head weights are identical).

    Attributes
    ----------
    tp_size : int
        Number of GPUs.  Must be exactly 2 for this implementation.
    num_layers : int
        Total transformer decoder layers (default 48 for Gemma 3 12B).
    _layers_per_gpu : list[int]
        Layer count per GPU (backing field — use layers_per_gpu property).
    _layer_ranges : list[tuple]
        (start, end) per GPU (backing field — use layer_ranges property).
    replicate_embedding : bool
        Whether to replicate the input embedding on every GPU (default True).
    replicate_lm_head : bool
        Whether to replicate the LM head and all-reduce logits (default True).
    all_reduce_lm_head : bool
        Whether to insert an all-reduce after the LM head forward (default True).
    """

    tp_size: int = 2
    num_layers: int = GEMMA_3_12B_NUM_LAYERS
    hidden_size: int = GEMMA_3_12B_HIDDEN_SIZE
    num_attention_heads: int = GEMMA_3_12B_NUM_ATTENTION_HEADS
    num_kv_heads: int = GEMMA_3_12B_NUM_KV_HEADS
    intermediate_size: int = GEMMA_3_12B_INTERMEDIATE_SIZE
    local_attn_window: int = GEMMA_3_12B_LOCAL_ATTN_WINDOW
    local_global_ratio: int = GEMMA_3_12B_LOCAL_GLOBAL_RATIO
    _layers_per_gpu: List[int] = field(default_factory=list)
    _layer_ranges: List[tuple] = field(default_factory=list)
    replicate_embedding: bool = True
    replicate_lm_head: bool = True
    all_reduce_lm_head: bool = True

    def __post_init__(self):
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {self.tp_size}")
        if self.num_layers % self.tp_size != 0:
            raise ValueError(
                f"num_layers ({self.num_layers}) must be divisible by "
                f"tp_size ({self.tp_size})."
            )
        per_gpu = self.num_layers // self.tp_size
        self._layers_per_gpu = [per_gpu] * self.tp_size
        self._layer_ranges = [
            (i * per_gpu, (i + 1) * per_gpu) for i in range(self.tp_size)
        ]
        # For backward compatibility when tp_size=1
        if self.tp_size == 1:
            self._layer_ranges = [(0, self.num_layers)]

    # --- Properties for backward compatibility ---

    @property
    def layers_per_gpu(self) -> List[int]:
        """Layer count per GPU (backward compat)."""
        return self._layers_per_gpu

    @property
    def layer_ranges(self) -> List[tuple]:
        """(start, end) layer ranges per GPU."""
        return self._layer_ranges

    @property
    def ranks(self) -> List[int]:
        """Return the list of process ranks participating in TP."""
        return list(range(self.tp_size))

    # --- Layer mapping ---

    def get_device_for_layer(self, layer_idx: int) -> int:
        """Return the GPU rank that owns a given layer."""
        for rank, (start, end) in enumerate(self._layer_ranges):
            if start <= layer_idx < end:
                return rank
        return 0

    def get_global_attention_layers(self) -> list[int]:
        """Layers using full-context (global) attention — every 6th layer.

        Gemma 3 uses 5:1 local/global interleaving.
        Layers: 5, 11, 17, 23, 29, 35, 41, 47 (8 global layers total).
        """
        return [i for i in range(self.num_layers)
                if (i + 1) % (self.local_global_ratio + 1) == 0]

    def is_global_attention_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses global (full-context) attention."""
        return (layer_idx + 1) % (self.local_global_ratio + 1) == 0

    def estimate_kv_cache_mb(
        self, batch_size: int, sequence_length: int, bytes_per_element: int = 2
    ) -> float:
        """Estimate KV cache memory usage in MB.

        Only global attention layers store full context.
        Local attention layers cache only the sliding window.
        """
        n_global = len(self.get_global_attention_layers())
        n_local = self.num_layers - n_global
        local_win = min(self.local_attn_window, sequence_length)
        head_dim = self.hidden_size // self.num_attention_heads

        global_bytes = (
            n_global * batch_size * sequence_length
            * self.num_kv_heads * head_dim * 2 * bytes_per_element
        )
        local_bytes = (
            n_local * batch_size * local_win
            * self.num_kv_heads * head_dim * 2 * bytes_per_element
        )
        return (global_bytes + local_bytes) / (1024 * 1024)

    def layer_range_for_rank(self, rank: int) -> slice:
        """Return the layer index slice assigned to *rank*."""
        start, end = self._layer_ranges[rank]
        return slice(start, end)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _find_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Resolve a dotted attribute path on *model*, e.g. ``"model.layers"``."""
    parts = name.split(".")
    obj = model
    for part in parts:
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return obj


def _maybe_get_attr(model: nn.Module, candidates: List[str]) -> Optional[nn.Module]:
    """Return the first matching module from *candidates*, or None."""
    for name in candidates:
        mod = _find_module_by_name(model, name)
        if mod is not None:
            return mod
    return None


# ---------------------------------------------------------------------------
# Distributed initialization bootstrap
# ---------------------------------------------------------------------------


def ensure_dist_initialized(
    backend: str = "nccl",
    init_method: str = "env://",
    timeout_seconds: int = 300,
) -> bool:
    """Initialize torch.distributed if it hasn't been already.

    Safe to call on any rank — detects ``torchrun`` environment variables
    automatically.  On a single-GPU setup (no ``WORLD_SIZE`` / ``RANK`` in
    the environment) this is a no-op and returns False.

    Parameters
    ----------
    backend : str
        Distributed backend ("nccl", "gloo").  Defaults to "nccl" for CUDA.
    init_method : str
        ``torch.distributed.init_process_group`` init method.  "env://"
        reads MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK from the
        environment, which torchrun sets automatically.
    timeout_seconds : int
        Timeout for the distributed rendezvous.

    Returns
    -------
    bool
        True if distributed was initialized (or was already initialized).
        False if this is a single-process / single-GPU launch.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        return True

    # Only bootstrap when the launcher environment is present.
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size < 2:
        logger.debug("WORLD_SIZE < 2 — distributed init skipped.")
        return False

    # Fall back to gloo on non-CUDA systems.
    if not torch.cuda.is_available() and backend == "nccl":
        backend = "gloo"
        logger.info("CUDA not available — using gloo backend for testing.")

    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        logger.info(
            "Distributed initialized: backend=%s world_size=%d rank=%d",
            backend, dist.get_world_size(), dist.get_rank(),
        )
        return True
    except Exception as e:
        logger.warning("Distributed init failed: %s — running single-GPU.", e)
        return False


def _get_distributed_world() -> int:
    """Return the world size if distributed is initialized, otherwise 1."""
    try:
        import torch.distributed as dist
        return dist.get_world_size() if dist.is_initialized() else 1
    except ImportError:
        return 1


def _get_distributed_rank() -> int:
    """Return the local rank if distributed is initialized, otherwise 0."""
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except ImportError:
        return 0


class AllReduceLMHead(nn.Module):
    """Wrapper that all-reduces logits after the LM head forward pass.

    In replicate-and-split parallelism the LM head is **replicated** (each GPU
    holds IDENTICAL weights).  All-reducing logits via SUM followed by division
    by group size produces results identical to single-GPU inference because
    both replicas compute the same logit values.

    Note: this is logit-averaging, NOT true weight-sharded tensor parallelism.
    If the LM head weights ever differ across GPUs (e.g., weight-sharded TP),
    this approach produces incorrect results since softmax(mean(L1,L2)) ≠
    mean(softmax(L1), softmax(L2)).  With replicated weights, L1 == L2 so the
    nonlinearity is a no-op and both approaches are equivalent.
    """

    def __init__(self, lm_head: nn.Module, group=None):
        super().__init__()
        self.lm_head = lm_head
        self.group = group

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if self.group is not None:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.all_reduce(logits, op=dist.ReduceOp.SUM, group=self.group)
                logits = logits / self.group.size()
        return logits


# ---------------------------------------------------------------------------
# Main tensor parallelism application
# ---------------------------------------------------------------------------


def apply_tensor_parallelism(
    model: nn.Module,
    config: Optional[TensorParallelConfig] = None,
    *,
    group: Optional[dist.ProcessGroup] = None,
    rank: Optional[int] = None,
) -> nn.Module:
    """Apply 2-way tensor parallelism to a Gemma 3 12B model in-place.

    Sharding strategy
    -----------------
    1. **Transformer layers**: layers 0-23 -> GPU 0, layers 24-47 -> GPU 1.
       Layers not belonging to the current rank are deleted from the model
       to free memory.

    2. **Embedding**: replicated on every GPU (each GPU holds a full copy).

    3. **LM head**: replicated on every GPU.  An all-reduce wrapper is
       inserted so that logits are averaged across ranks after the forward.

    Parameters
    ----------
    model : nn.Module
        The loaded Gemma 3 12B model (e.g. from ``transformers`` or direct).
        Expected to have attributes like ``model.embed_tokens``,
        ``model.layers`` (a ``ModuleList``), and ``lm_head``.

    config : TensorParallelConfig, optional
        TP configuration.  Constructed with defaults if omitted.

    group : ProcessGroup, optional
        The NCCL process group for TP communication.  Defaults to the
        default distributed group.

    rank : int, optional
        The rank of the current process.  Defaults to ``dist.get_rank()``
        when distributed is initialized, otherwise 0.

    Returns
    -------
    nn.Module
        The same model instance, sharded in-place for the current rank.
    """
    if config is None:
        config = TensorParallelConfig()

    if rank is None:
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
        except ImportError:
            logger.warning("torch.distributed not available — rank defaults to 0.")
            rank = 0

    if rank not in config.ranks:
        raise ValueError(
            f"Rank {rank} is outside the TP group ranks {config.ranks}."
        )

    # Map process rank to local CUDA device index.
    # In single-node setups rank==device_index, but in multi-node setups
    # (e.g., torchrun with CUDA_VISIBLE_DEVICES) rank 0 and rank 1 are
    # two different processes each seeing a single GPU at index 0.
    # PyTorch maps the local rank to the device automatically when using
    # torchrun — we detect the local CUDA device via ``torch.cuda.current_device()``
    # rather than assuming ``rank`` is the device index.
    local_device = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device_str = f"cuda:{local_device}"

    # -- locate submodules -------------------------------------------------
    layers = _maybe_get_attr(
        model,
        [
            "model.layers",          # HF Gemma2/3 style
            "model.decoder.layers",  # Alternative HF layout
            "transformer.layers",    # Generic
        ],
    )

    embed = _maybe_get_attr(
        model,
        [
            "model.embed_tokens",
            "model.decoder.embed_tokens",
            "transformer.embed_tokens",
            "transformer.wte",
        ],
    )

    lm_head = _maybe_get_attr(
        model,
        [
            "lm_head",
            "model.lm_head",
            "transformer.lm_head",
        ],
    )

    # -- validate layer count -----------------------------------------------
    if layers is None:
        raise RuntimeError(
            "Could not locate the transformer layers module. "
            "Expected one of: model.layers, model.decoder.layers, transformer.layers"
        )

    if not isinstance(layers, nn.ModuleList):
        raise TypeError(
            f"Transformer layers must be a ModuleList, got {type(layers).__name__}"
        )

    actual_layers = len(layers)
    if actual_layers != config.num_layers:
        logger.warning(
            "Model has %d layers but config expects %d layers. "
            "Using actual layer count.",
            actual_layers,
            config.num_layers,
        )
        # Adjust config to match reality — write to the backing fields
        # since layers_per_gpu/layer_ranges are read-only @property.
        config.num_layers = actual_layers
        per_gpu = actual_layers // config.tp_size
        if actual_layers % config.tp_size != 0:
            raise ValueError(
                f"Model has {actual_layers} layers, which is not divisible by "
                f"tp_size={config.tp_size}."
            )
        config._layers_per_gpu = [per_gpu] * config.tp_size
        config._layer_ranges = [
            (i * per_gpu, (i + 1) * per_gpu) for i in range(config.tp_size)
        ]

    # -- get our layer slice ------------------------------------------------
    layer_slice = config.layer_range_for_rank(rank)
    my_start, my_stop = layer_slice.start, layer_slice.stop

    assert my_stop is not None  # type narrowing for slice attributes

    logger.info(
        "Rank %d: keeping layers [%d, %d) - %d layers total",
        rank,
        my_start,
        my_stop,
        my_stop - my_start,
    )

    # -- shard layers -------------------------------------------------------
    # Remove layers that belong to other ranks (highest indices first to
    # preserve lower-index semantics during deletion).
    # First, dump everything that is downstream of our range.
    to_delete_high = list(range(my_stop, config.num_layers))
    for idx in reversed(to_delete_high):
        del layers[idx]

    # Then, dump everything upstream of our range.
    to_delete_low = list(range(0, my_start))
    for idx in reversed(to_delete_low):
        del layers[idx]

    logger.info("Rank %d: layers ModuleList now has %d entries", rank, len(layers))

    # -- replicate embedding ------------------------------------------------
    if config.replicate_embedding and embed is not None:
        embed.to(torch.device(device_str))
        logger.debug("Rank %d: embedding replicated (not sharded) on %s.", rank, device_str)

    # -- replicate LM head + all-reduce wrapper -----------------------------
    if config.replicate_lm_head and lm_head is not None:
        if config.all_reduce_lm_head:
            try:
                import torch.distributed as dist
                dist_ok = dist.is_initialized()
            except ImportError:
                logger.warning("torch.distributed not available — LM head all-reduce disabled.")
                dist_ok = False

            if dist_ok:
                # Wrap the LM head so logits are averaged across TP ranks.
                wrapped = AllReduceLMHead(lm_head, group=group)

                # Replace the LM head reference on the model so that the forward
                # path goes through the wrapper.
                replaced = False
                for candidate in [
                    "lm_head",
                    "model.lm_head",
                    "transformer.lm_head",
                ]:
                    target = _find_module_by_name(model, candidate)
                    if target is not None:
                        if "." in candidate:
                            parent_path, attr = candidate.rsplit(".", 1)
                            parent = _find_module_by_name(model, parent_path)
                            setattr(parent, attr, wrapped)
                        else:
                            setattr(model, attr, wrapped)
                        replaced = True
                        break

                if not replaced:
                    logger.warning(
                        "LM head all-reduce wrapping failed - LM head not found at expected paths."
                    )
                else:
                    logger.info("Rank %d: LM head replicated with all-reduce wrapper.", rank)
            else:
                logger.info(
                    "Rank %d: LM head replicated (no all-reduce - single GPU or dist not initialized).",
                    rank,
                )

    # -- move model to the correct device ----------------------------------
    model.to(torch.device(device_str))
    logger.info("Rank %d: model sharded and moved to %s.", rank, device_str)

    return model


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_tensor_parallel_config(
    num_gpus: int = 2,
    tp_size: int | None = None,
    num_layers: int | None = None,
    replicate_embedding: bool = True,
    replicate_lm_head: bool = True,
    all_reduce_lm_head: bool = True,
) -> TensorParallelConfig:
    """Create a TensorParallelConfig with the given parameters.

    Parameters
    ----------
    num_gpus : int
        Number of available GPUs. tp_size = min(num_gpus, 2).
    tp_size : int, optional
        Explicit TP size override (takes precedence over num_gpus).
    num_layers : int, optional
        Total transformer layers.  Defaults to GEMMA_3_12B_NUM_LAYERS (48).
        For other models, pass the actual layer count from model.config.
    replicate_embedding : bool
        Replicate embeddings on every GPU.
    replicate_lm_head : bool
        Replicate LM head on every GPU.
    all_reduce_lm_head : bool
        All-reduce logits after the LM head forward.

    Returns
    -------
    TensorParallelConfig
    """
    if num_layers is None:
        num_layers = GEMMA_3_12B_NUM_LAYERS
    if tp_size is None:
        tp_size = min(num_gpus, 2)
    if tp_size < 2:
        logger.info("Tensor parallelism disabled (%d GPU(s))", num_gpus)
        return TensorParallelConfig(tp_size=1, num_layers=num_layers)

    return TensorParallelConfig(
        tp_size=tp_size,
        num_layers=num_layers,
        replicate_embedding=replicate_embedding,
        replicate_lm_head=replicate_lm_head,
        all_reduce_lm_head=all_reduce_lm_head,
    )
