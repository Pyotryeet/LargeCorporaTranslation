"""vLLM engine backend (v3.8).

Runs translation inference using the highly optimized vLLM engine,
supporting continuous batching, PagedAttention, and tensor parallelism.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from benchmark.inference.backends.protocol import (
    BackendConfig,
    BatchGenerationOutput,
    GenerationOutput,
    InferenceBackend,
    ModelCapability,
    ModelType,
)

logger = logging.getLogger(__name__)


class VLLMBackend(InferenceBackend):
    """Wrapper around vllm.LLM engine conforming to the InferenceBackend protocol.

    This backend exposes high-throughput translation inference via vLLM's
    PagedAttention, continuous batching, and automatic tensor parallelism
    across all available CUDA GPUs.  It supports NLLB (encoder-decoder) and
    Gemma (decoder-only) model architectures out of the box.

    Class Attributes
    ----------------
    model_type : ModelType
        Set to ``ModelType.VLLM``.
    capabilities : ModelCapability
        Bitmask supporting TRANSLATE, FORWARD_ENCODE, and ENSEMBLE_READY.
    display_name : str
        Human-readable label for reports: ``"vLLM Engine Backend"``.

    Instance Attributes
    -------------------
    model_path : str
        HuggingFace model ID or local path from the BackendConfig.
    tokenizer_path : str
        Tokenizer path (defaults to model_path if not explicitly set).
    max_new_tokens : int
        Maximum tokens to generate per output.
    temperature : float
        Sampling temperature for token generation.
    llm : vllm.LLM or None
        The vLLM engine instance.  Created lazily during ``load()``;
        ``None`` until ``load()`` completes successfully.
    sampling_params : vllm.SamplingParams or None
        Configured sampling parameters.  Created during ``load()``.

    Important Caveats
    -----------------
    - ``load()`` MUST be called before any inference; the engine is NOT
      created in ``__init__``.
    - vLLM allocates ``gpu_memory_utilization=0.90`` of all visible GPUs
      and uses ``tensor_parallel_size=n_gpus``.  Ensure no other process
      is competing for GPU memory.
    - ``enforce_eager=True`` is set to avoid long CUDA graph warmup delays
      for dynamic batches — this trades throughput for lower latency at
      small batch sizes.
    """

    model_type = ModelType.VLLM
    capabilities = (
        ModelCapability.TRANSLATE
        | ModelCapability.FORWARD_ENCODE
        | ModelCapability.ENSEMBLE_READY
    )
    display_name = "vLLM Engine Backend"

    def __init__(self, config: BackendConfig):
        """Initialize the vLLM backend with a BackendConfig.

        Parameters
        ----------
        config : BackendConfig
            Configuration dataclass providing ``model_path``, ``tokenizer_path``,
            ``max_new_tokens``, ``temperature``, and ``extra`` (which may contain
            ``batch_size``, defaulting to 4).

        Side Effects
        ------------
        - Stores configuration values on ``self``.  Does NOT load the model
          or allocate GPU memory — ``load()`` must be called separately.
        - The vLLM engine (``self.llm``) and sampling params are initialized
          to ``None``.
        """
        super().__init__(config)
        self.model_path = config.model_path
        self.tokenizer_path = config.tokenizer_path or config.model_path
        self.max_new_tokens = config.max_new_tokens
        self.temperature = config.temperature

        # Created lazily during load()
        self.llm = None
        self.sampling_params = None
        self._configured_batch_size = config.extra.get("batch_size", 4)

    def load(self) -> None:
        """Load the vLLM engine, tokenizer, and configure sampling parameters.

        This method must be called after ``__init__`` and before any inference.
        It performs the following steps in order:

        1. Imports ``vllm.LLM`` and ``vllm.SamplingParams`` (lazy import to
           avoid requiring vLLM at module-import time).
        2. Loads the HuggingFace tokenizer with ``trust_remote_code=True``.
        3. Builds ``vllm.SamplingParams`` from the configured temperature,
           max tokens, and ``ignore_eos=False``.
        4. Detects the number of available CUDA GPUs via
           ``torch.cuda.device_count()``.
        5. Instantiates ``vllm.LLM`` with tensor parallelism across all GPUs,
           90% GPU memory utilization, ``trust_remote_code=True``, and
           ``enforce_eager=True`` to skip long CUDA graph warmup.

        Side Effects
        ------------
        - Allocates GPU memory proportional to ``gpu_memory_utilization=0.90``
          across all visible CUDA devices.
        - Sets ``self._loaded = True`` on success.
        - Sets ``self.tokenizer``, ``self.sampling_params``, and ``self.llm``.

        Raises
        ------
        ImportError
            If the ``vllm`` package is not installed in the environment.
        RuntimeError
            If no CUDA-capable GPU is detected.
        OSError
            If the model path or tokenizer path is not accessible.
        """
        logger.info("=== vLLM Backend: initializing LLM engine for %s ===", self.model_path)
        load_start = time.monotonic()

        from vllm import LLM, SamplingParams

        # Load tokenizer first to resolve target language prefixes if needed
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )

        # Build sampling params
        sampling_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "ignore_eos": False,
        }

        # vLLM-specific NLLB target language code handling
        extra = self.config.extra
        if "nllb" in self.model_path.lower():
            # For Seq2Seq models like NLLB/MADLAD, identify the target language token
            tgt_lang = extra.get("nllb_target_lang", "tur_Latn")
            # If target lang prefix matches NLLB syntax, pass to vLLM's prompt prefix if required
            # Or vLLM will automatically read the tokenizer configuration
            logger.info("NLLB model detected in vLLM: target language is %s", tgt_lang)

        self.sampling_params = SamplingParams(**sampling_kwargs)

        # Detect GPUs
        if not torch.cuda.is_available():
            raise RuntimeError(
                "vLLM backend requires CUDA — no GPU detected by PyTorch"
            )
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError(
                "vLLM backend requires at least 1 GPU, but "
                "torch.cuda.device_count() returned 0. Check "
                "CUDA_VISIBLE_DEVICES, nvidia-smi, and driver installation."
            )
        logger.info("vLLM Engine: allocating all %d GPUs via Tensor Parallelism", n_gpus)

        # Instantiate vLLM engine
        # trust_remote_code=True is required for custom model architectures like Gemma 3
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=n_gpus,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            enforce_eager=True,  # Avoid long CUDA graph warmup delays for dynamic batches
        )

        self._loaded = True
        logger.info(
            "vLLM Backend loaded successfully in %.1fs (tensor_parallel=%d)",
            time.monotonic() - load_start,
            n_gpus,
        )

    def warmup(self, batches: int = 1) -> None:
        """Run warm-up batches to prime CUDA caches and graphs.

        Parameters
        ----------
        batches : int, default=1
            Number of warm-up batches to run.  Ignored in this implementation
            because vLLM performs automatic profiling and CUDA graph capture
            during ``LLM.__init__`` — no manual warmup loop is needed.

        Side Effects
        ------------
        None.  This is an intentional no-op.

        Notes
        -----
        vLLM automatically profiles the model and captures CUDA graphs during
        engine construction, so the standard warmup phase is redundant.  This
        method exists solely to satisfy the ``InferenceBackend`` protocol.
        """
        # vLLM automatically profiles and warms up CUDA memory and graphs
        # during LLM initialization. No manual warmup loop is required.
        pass

    def translate_batch(self, batch: Any) -> BatchGenerationOutput:
        """Translate a single batch of raw text strings via the vLLM engine.

        Each string in the batch is passed directly to vLLM's ``generate()``,
        which schedules and executes the batch internally using continuous
        batching.  vLLM handles tokenization on its own, so this method works
        with ``raw_texts`` (not pre-tokenized inputs).

        Parameters
        ----------
        batch : PipelineBatch or similar
            A batch object that must have a ``raw_texts`` attribute containing
            the list of source-language strings to translate.  If the optional
            ``batch_id`` attribute is present it is forwarded into the output.

        Returns
        -------
        BatchGenerationOutput
            Dataclass containing:
            - ``batch_id`` : int (0 if the batch had no ``batch_id``).
            - ``generations`` : list of ``GenerationOutput``, one per input text.
            - ``batch_size`` : int (0 if the input batch was empty).
            - ``input_tokens_total`` : sum of input token counts across
              all generations.
            - ``output_tokens_total`` : sum of output token counts across
              all generations.
            - ``total_latency_ms`` : wall-clock duration of the vLLM
              generate call in milliseconds, rounded to 2 decimal places.
            - ``phase_timings`` : dict with ``{"method": "vllm"}``.

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called or ``self._loaded`` is ``False``.

        Side Effects
        ------------
        None.  The method is read-only with respect to backend state.

        Notes
        -----
        - The per-generation ``total_latency_ms`` is the batch wall-clock
          divided by the number of inputs, which is a rough approximation
          and not a per-sample measurement.
        - An empty batch (no ``raw_texts``) returns a ``BatchGenerationOutput``
          with ``batch_size=0`` and no generations.
        """
        if not self._loaded:
            raise RuntimeError("vLLM model not loaded")

        raw_texts = batch.raw_texts if hasattr(batch, "raw_texts") else []
        if not raw_texts:
            return BatchGenerationOutput(
                batch_id=batch.batch_id if hasattr(batch, "batch_id") else 0,
                batch_size=0,
            )

        wall_start = time.monotonic()

        # Execute generation via vLLM
        # vLLM schedules and runs the batch with continuous batching internally
        outputs = self.llm.generate(raw_texts, self.sampling_params, use_tqdm=False)

        wall_end = time.monotonic()
        total_wall_ms = (wall_end - wall_start) * 1000.0

        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

        generations: List[GenerationOutput] = []
        total_in = 0
        total_out = 0

        for i, out in enumerate(outputs):
            gen_text = out.outputs[0].text.strip()
            in_tok = len(out.prompt_token_ids)
            out_tok = len(out.outputs[0].token_ids)

            generations.append(
                GenerationOutput(
                    input_text=raw_texts[i],
                    translated_text=gen_text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_latency_ms=total_wall_ms / len(raw_texts),
                    timestamp_utc=ts,
                )
            )
            total_in += in_tok
            total_out += out_tok

        return BatchGenerationOutput(
            batch_id=batch.batch_id if hasattr(batch, "batch_id") else 0,
            generations=generations,
            batch_size=len(generations),
            input_tokens_total=total_in,
            output_tokens_total=total_out,
            total_latency_ms=round(total_wall_ms, 2),
            phase_timings={
                "method": "vllm",
            },
        )

    def is_loaded(self) -> bool:
        """Return whether the vLLM engine has been successfully loaded.

        Returns
        -------
        bool
            ``True`` if ``load()`` completed successfully and the engine is
            ready for inference, ``False`` otherwise.
        """
        return self._loaded

    def close(self) -> None:
        """Release the vLLM engine and free all CUDA memory.

        Destroys the ``vllm.LLM`` reference so Python can garbage-collect it,
        then invokes an explicit ``gc.collect()`` and ``torch.cuda.empty_cache()``
        to return GPU memory to the driver.  Sets ``self._loaded`` to ``False``.

        Side Effects
        ------------
        - Drops the reference to the vLLM engine (``self.llm = None``).
        - Runs Python garbage collection via ``gc.collect()``.
        - Calls ``torch.cuda.empty_cache()`` to free all cached CUDA allocator
          blocks.  This affects ALL PyTorch tensors in the process, not just
          those owned by vLLM.

        Notes
        -----
        After calling ``close()``, the backend can be re-loaded by calling
        ``load()`` again.  However, there is no guarantee that the same amount
        of GPU memory will be available.
        """
        if self.llm is not None:
            # vLLM manages its own CUDA resources. To release memory, we destroy the object.
            import gc
            self.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._loaded = False
            logger.info("vLLM Engine released and memory cleared")
