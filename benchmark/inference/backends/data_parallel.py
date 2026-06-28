"""Replicated data parallelism — load independent model copies on each GPU.

Tier 1 of the TPS Acceleration Plan.  No cross-GPU communication — each GPU
runs an independent autoregressive backend on half the batch.  2× throughput
with zero Amdahl's overhead when the model fits on a single GPU.

TranslateGemma 4B (~8GB BF16) × 2 copies = ~16GB in 140GB H200. Trivial.
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Any, Optional

import torch

from benchmark.inference.backends.registry import ModelRegistry
from benchmark.inference.backends.protocol import (
    BackendConfig, BatchGenerationOutput, GenerationOutput,
    InferenceBackend, ModelCapability, ModelType,
)

logger = logging.getLogger(__name__)


def _make_gpu_config(original: BackendConfig, gpu_idx: int) -> BackendConfig:
    """Build a BackendConfig scoped to a single GPU.

    Creates a fresh DeviceInfo for the specified GPU index and returns a new
    BackendConfig that shares all model-level settings (paths, token limits,
    temperature, dtype, attention/compile flags, extra kwargs) from the original
    but points at a single device.

    Parameters
    ----------
    original : BackendConfig
        The original config carrying model-path and inference settings.
        Its ``extra`` dict is shallow-copied so callers cannot accidentally
        mutate the canonical config.
    gpu_idx : int
        Zero-based CUDA device index (e.g. 0 for cuda:0).

    Returns
    -------
    BackendConfig
        A new config whose ``device_info`` is bound to ``cuda:{gpu_idx}``.

    Side effects
    ------------
    Imports ``DeviceInfo`` from ``benchmark.hardware.backend`` at call time
    (trade-off: keeps the module-level import surface small).
    """
    device = torch.device(f"cuda:{gpu_idx}")
    from benchmark.hardware.backend import DeviceInfo
    gpu_info = DeviceInfo(
        backend="cuda", device=device, num_devices=1,
        name=torch.cuda.get_device_name(gpu_idx),
        total_memory_gb=torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3),
    )
    return BackendConfig(
        model_path=original.model_path,
        tokenizer_path=original.tokenizer_path,
        device_info=gpu_info,
        max_input_tokens=original.max_input_tokens,
        max_new_tokens=original.max_new_tokens,
        temperature=original.temperature,
        dtype=original.dtype,
        use_flash_attention=original.use_flash_attention,
        use_torch_compile=original.use_torch_compile,
        extra=dict(original.extra),
    )


class DataParallelBackend(InferenceBackend):
    """Wraps N AutoregressiveBackend instances — one per GPU.

    Splits each batch into N sub-batches, dispatches to independent
    backends via Python threads for true concurrent GPU execution.
    Zero cross-GPU communication.
    """

    model_type = ModelType.AUTOREGRESSIVE
    capabilities = (
        ModelCapability.TRANSLATE | ModelCapability.FORWARD_ENCODE
        | ModelCapability.QUANTIZABLE_KV
        | ModelCapability.SPECULATIVE | ModelCapability.ENSEMBLE_READY
    )
    display_name = "Data-Parallel Autoregressive (N×GPU)"

    def __init__(self, config: BackendConfig, num_gpus: int = 2):
        """Construct a data-parallel wrapper over N GPU replicas.

        Parameters
        ----------
        config : BackendConfig
            Shared model configuration.  Each GPU replica will receive its own
            device-targeted copy created via :func:`_make_gpu_config`.
        num_gpus : int, optional
            Number of GPUs to use.  Clamped to ``torch.cuda.device_count()``.
            Defaults to 2.

        Raises
        ------
        ValueError
            If fewer than 2 CUDA-capable GPUs are visible.

        Notes
        -----
        Call ``load()`` after construction to instantiate the per-GPU backends.
        """
        super().__init__(config)
        self._num_gpus = min(num_gpus, torch.cuda.device_count())
        if self._num_gpus < 2:
            raise ValueError(
                f"DataParallelBackend requires ≥2 GPUs, found {self._num_gpus}. "
                f"Use AutoregressiveBackend directly for single-GPU."
            )
        self._backends: list[InferenceBackend] = []
        self._devices: list[torch.device] = []
        self.devices = self._devices
        self.backend_name = "cuda"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Create and load an independent model replica on each GPU.

        For each GPU:
        1. Builds a device-specific ``BackendConfig`` via :func:`_make_gpu_config`.
        2. Instantiates a backend through the ``ModelRegistry``.
        3. Calls ``backend.load()`` on that GPU's CUDA context.

        After loading all replicas, the wrapper inherits ``tokenizer``, ``model``,
        ``precision_config``, ``model_type``, ``capabilities``, and ``display_name``
        from the first replica (GPU 0).

        Side effects
        ------------
        - ``self._loaded`` is set to ``True`` on success.
        - CUDA memory is allocated on each GPU.
        - Logs per-GPU memory usage and total wall-clock load time.
        """
        logger.info("=== DataParallelBackend: loading %d GPUs ===", self._num_gpus)
        load_start = time.monotonic()
        registry = ModelRegistry()

        for gpu_idx in range(self._num_gpus):
            self._devices.append(torch.device(f"cuda:{gpu_idx}"))
            with torch.cuda.device(gpu_idx):
                gpu_config = _make_gpu_config(self.config, gpu_idx)
                backend = registry.create_backend(gpu_config)
                backend.load()
            self._backends.append(backend)
            logger.info("  GPU %d: model loaded (%.1f GB allocated)",
                        gpu_idx, torch.cuda.memory_allocated(gpu_idx) / (1024**3))

        self._loaded = True
        load_duration = time.monotonic() - load_start
        b0 = self._backends[0]
        self.tokenizer = b0.tokenizer
        self.model = b0.model
        self.precision_config = b0.precision_config

        # Dynamically inherit properties from the wrapped sub-backends
        self.model_type = b0.model_type
        self.capabilities = b0.capabilities
        self.display_name = f"Data-Parallel {b0.display_name}"

        logger.info("=== DataParallelBackend: all %d GPUs loaded in %.1fs ===",
                    self._num_gpus, load_duration)

    def warmup(self, batches: int = 20) -> None:
        """Run warm-up batches concurrently across all GPUs.

        Evenly distributes ``batches`` across GPUs (minimum 5 per GPU), then
        launches one Python thread per GPU.  Each thread sets its own CUDA device
        and invokes the replica's ``warmup()``.  All threads are joined before
        returning.

        Parameters
        ----------
        batches : int, optional
            Total warm-up iterations to perform.  Split across GPUs.
            Defaults to 20.

        Raises
        ------
        RuntimeError
            If the model has not been loaded, or if any per-GPU warmup fails.
            The exception message includes the failing GPU index.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        n_each = max(batches // self._num_gpus, 5)
        errors: list[Optional[Exception]] = [None] * self._num_gpus

        def _warmup_one(idx: int):
            try:
                torch.cuda.set_device(self._devices[idx])
                self._backends[idx].warmup(batches=n_each)
            except Exception as e:
                errors[idx] = e

        threads = [threading.Thread(target=_warmup_one, args=(i,), name=f"dp-warmup-{i}")
                   for i in range(self._num_gpus)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(f"GPU {i} warmup failed: {err}") from err
        logger.info("DataParallel warmup: all GPUs ready")

    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        """Split a batch evenly across GPUs and run inference concurrently.

        Chunking strategy
        -----------------
        The batch is partitioned into ``N`` sub-batches using ceiling division,
        where ``N == self._num_gpus``.  If the batch size is not evenly divisible
        some GPUs may receive one fewer item.  Tensors are moved to each GPU with
        ``non_blocking=True`` to overlap transfer with compute.

        Each sub-batch is dispatched to its GPU via a Python thread.  Because every
        GPU runs its own CUDA context, threads execute in parallel with zero
        cross-GPU communication.

        Merge strategy
        --------------
        After all threads join, individual ``GenerationOutput`` results are
        collected.  Every generation's ``total_latency_ms`` is overwritten with the
        wall-clock duration of the slowest GPU (i.e. the end-to-end batch latency).
        Phase timings take the per-key maximum across replicas, and two metadata
        fields are added:

        - ``phase_timings["method"]``: set to ``"data_parallel_<N>gpu"``.

        Parameters
        ----------
        batch : PipelineBatch-like object
            Must expose ``input_ids``, ``attention_mask`` (both tensors on CPU or
            any GPU), and optionally ``raw_texts`` and ``batch_id``.

        Returns
        -------
        BatchGenerationOutput
            Merged output from all GPU replicas.  The ``generations`` list is
            concatenated and counts are summed.

        Raises
        ------
        RuntimeError
            If the model has not been loaded, or if any GPU thread raises an
            exception during translation.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        N = self._num_gpus
        bs = batch.input_ids.shape[0]
        chunk_size = (bs + N - 1) // N

        # Pre-compute sub-batches, moving tensors to target GPU
        sub_batches: list[Any] = []
        for gpu_idx in range(N):
            start = gpu_idx * chunk_size
            end = min(start + chunk_size, bs)
            if start >= bs:
                sub_batches.append(None)
                continue
            sub = type("SubBatch", (), {})()  # lightweight object for attributes
            sub.input_ids = batch.input_ids[start:end].to(self._devices[gpu_idx], non_blocking=True)
            sub.attention_mask = batch.attention_mask[start:end].to(self._devices[gpu_idx], non_blocking=True)
            sub.raw_texts = (
                batch.raw_texts[start:end]
                if hasattr(batch, 'raw_texts') else [""] * (end - start)
            )
            sub.batch_id = batch.batch_id if hasattr(batch, 'batch_id') else 0
            sub_batches.append(sub)

        # ── Concurrent GPU dispatch via Python threads ──
        # Each thread sets its own cuda: device, then runs translate_batch.
        # Threads execute in parallel because each GPU is an independent
        # device with separate CUDA contexts and memory spaces.
        sub_results: list[Optional[BatchGenerationOutput]] = [None] * N
        errors: list[Optional[Exception]] = [None] * N
        barriers = [threading.Event() for _ in range(N)]

        def _gpu_worker(idx: int):
            try:
                sub = sub_batches[idx]
                if sub is None:
                    return
                with torch.cuda.device(self._devices[idx]):
                    sub_results[idx] = self._backends[idx].translate_batch(sub)
            except Exception as e:
                errors[idx] = e
                logger.exception("GPU %d translate_batch crashed", idx)
            finally:
                barriers[idx].set()

        wall_start = time.monotonic()
        for gpu_idx in range(N):
            if sub_batches[gpu_idx] is None:
                barriers[gpu_idx].set()
                continue
            t = threading.Thread(
                target=_gpu_worker, args=(gpu_idx,),
                name=f"dp-gpu-{gpu_idx}", daemon=False,
            )
            t.start()

        # Await all threads
        for barrier in barriers:
            barrier.wait()
        wall_end = time.monotonic()
        total_wall_ms = (wall_end - wall_start) * 1000.0

        # Check for errors
        for gpu_idx, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(f"GPU {gpu_idx} failed: {err}") from err

        # ── Merge results ──
        all_generations: list[GenerationOutput] = []
        total_in = total_out = 0
        phase_timings: dict[str, float] = {}

        for result in sub_results:
            if result is None:
                continue
            for gen in result.generations:
                gen.total_latency_ms = total_wall_ms
                all_generations.append(gen)
            total_in += result.input_tokens_total
            total_out += result.output_tokens_total
            for k, v in result.phase_timings.items():
                if isinstance(v, (int, float)):
                    phase_timings[k] = max(phase_timings.get(k, 0.0), v)

        return BatchGenerationOutput(
            batch_id=batch.batch_id if hasattr(batch, 'batch_id') else 0,
            generations=all_generations,
            batch_size=len(all_generations),
            input_tokens_total=total_in,
            output_tokens_total=total_out,
            total_latency_ms=round(total_wall_ms, 2),
            phase_timings={
                **phase_timings,
                "method": f"data_parallel_{self._num_gpus}gpu",
            },
        )

    # ── Protocol compliance ───────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self._loaded

    def encode_source(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Produce encoder hidden states by delegating to the GPU-0 replica.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenized source sequence(s).
        attention_mask : torch.Tensor
            Attention mask for the source sequence(s).

        Returns
        -------
        torch.Tensor
            Hidden states of shape ``[batch, src_len, hidden_size]`` from the
            first GPU replica.

        Raises
        ------
        RuntimeError
            If the model has not been loaded.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        return self._backends[0].encode_source(input_ids, attention_mask)

    @property
    def kv_cache_config(self) -> dict[str, Any]:
        if not self._backends:
            return {}
        return self._backends[0].kv_cache_config

    # tokenizer, model, devices, backend_name, precision_config are plain
    # attributes set in __init__/load().  Do NOT define them as @property
    # — InferenceBackend.__init__ assigns them and would crash.

    @property
    def _configured_batch_size(self) -> int:
        """The (possibly tuned) batch size, read from GPU 0 and broadcast to all replicas.

        Getter
        ------
        Returns the ``_configured_batch_size`` of the first sub-backend, or 1
        if no backends have been loaded yet.

        Setter
        ------
        Propagates the given value to every sub-backend so all replicas agree
        on the max batch size (e.g. after auto-tuning).
        """
        if not self._backends:
            return 1
        return self._backends[0]._configured_batch_size

    @_configured_batch_size.setter
    def _configured_batch_size(self, value: int):
        for backend in self._backends:
            backend._configured_batch_size = value

    def close(self) -> None:
        """Release GPU resources for all replicas.

        Calls ``close()`` on every sub-backend (which typically frees CUDA
        allocations and clears GPU caches), then sets ``self._loaded = False``.

        Side effects
        ------------
        After this call the wrapper is no longer usable; ``load()`` must be
        called again to re-initialize.
        """
        for backend in self._backends:
            backend.close()
        self._loaded = False
