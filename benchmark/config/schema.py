"""Pydantic v2 configuration models with strict validation.

Classes:
    ModelConfig: Per-model inference flags (dtype, beam, speculative, quantization, etc.).
    RuntimeConfig: Benchmark duration, checkpointing, observability settings.
    DataConfig: Input paths, preprocessing, shuffle budget, quality reference set.
    ExtrapolationConfig: Throughput/cost projection parameters.
    BenchmarkConfig: Frozen root model composing the four sub-configs.

Functions:
    load_config: Read and validate a YAML config file into a BenchmarkConfig.

All models use model_config = {"extra": "forbid"} so unknown YAML keys
raise pydantic ValidationError. BenchmarkConfig additionally sets
frozen=True, preventing field reassignment after construction."""

import glob
import logging
from pathlib import Path
from typing import Any, Literal, Optional
import yaml
from pydantic import BaseModel, Field, model_validator

from benchmark.config.constants import (
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NOISE_SCHEDULE,
    DEFAULT_TARGET_LENGTH_MULTIPLIER,
)

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Pydantic v2 model for per-model inference configuration.

    Fields:
        model_path: HuggingFace model ID or local path. Defaults to "google/translategemma-4b-it".
        tokenizer_path: Path to tokenizer. Defaults to model_path when empty.
        max_input_tokens: Maximum input token count. Range 1-4096, default 512.
        max_new_tokens: Maximum generated token count. Range 1-2048, default 512.
        temperature: Sampling temperature. Range 0.0-2.0, default 0.0 (greedy).
        do_sample: Enable probabilistic sampling. Default False.
        num_beams: Beam-search width. Range 1-64, default 1.
        dtype: Model precision. "auto" lets the backend decide. Can override with
            "float8_e4m3fn", "bfloat16", "float16", or "float32".
        tensor_parallel_size: Number of GPUs for tensor parallelism. 0 = auto.
        use_flash_attention: Enable FlashAttention-2. Default True.
        backend_type: Inference backend: "auto", "autoregressive", "encoder_decoder",
            "diffusion", "custom", or "vllm". Default "auto".
        diffusion_steps: Number of denoising steps for diffusion backends. 8-4096.
        guidance_scale: Classifier-free guidance scale. 1.0-10.0.
        noise_schedule: Diffusion noise schedule: "cosine", "linear", or "sqrt".
        target_length_multiplier: Target-token multiplier for length-predicting diffusion.
        plugin_name: Name of custom backend plugin when backend_type="custom".
        plugin_config: Arbitrary key-value settings for the custom plugin.
        use_speculative: Enable speculative decoding (v3.4).
        speculative_mode: "self" (self-speculation) or "draft_model".
        speculative_num_tokens: How many tokens to speculate per step (1-16).
        speculative_draft_model: HuggingFace ID of the draft model.
        speculative_num_draft_layers: How many decoder layers to use from draft (0-128).
        use_paged_attention: Enable PagedAttention KV-cache (CUDA only, v3.5).
        use_continuous_batching: Enable continuous/dynamic batching (v3.5).
        quantization: bitsandbytes quantization level: "bf16", "fp16", "int8", or "int4".
        data_parallel_size: Number of data-parallel replicas across GPUs (v3.7).
        nllb_source_lang: Source language code for NLLB translation models.
        nllb_target_lang: Target language code for NLLB translation models.

    Validators:
        validate_model_config: Enforces consistency rules (see method docstring).

    Model config forbids extra fields; unknown keys in YAML will raise ValidationError.
    """
    model_config = {"extra": "forbid"}
    model_path: str = Field(default="google/translategemma-4b-it", max_length=500)
    tokenizer_path: str = Field(default="", max_length=500)
    max_input_tokens: int = Field(default=512, ge=1, le=4096)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    do_sample: bool = False
    num_beams: int = Field(default=1, ge=1, le=64)
    # "auto" resolution: cuda->bfloat16, mps->bfloat16, cpu->float32
    # Set explicitly (float16/bfloat16/float32) to override.
    # NOTE: dtype is auto-detected by the backend (precision.py). This config
    # field is provided for explicit override only; the engine determines the
    # effective dtype from the backend, not from this config field directly.
    dtype: Literal["auto", "float8_e4m3fn", "bfloat16", "float16", "float32"] = "auto"
    tensor_parallel_size: int = Field(default=0, ge=0, le=8)
    use_flash_attention: bool = True
    # v3.0: Model-agnostic dispatch.
    backend_type: Literal["auto", "autoregressive", "encoder_decoder", "diffusion", "custom", "vllm"] = "auto"
    # v3.0: Diffusion-specific parameters.
    diffusion_steps: int = Field(default=DEFAULT_DIFFUSION_STEPS, ge=8, le=4096)
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, ge=1.0, le=10.0)
    noise_schedule: Literal["cosine", "linear", "sqrt"] = DEFAULT_NOISE_SCHEDULE
    target_length_multiplier: float = Field(default=DEFAULT_TARGET_LENGTH_MULTIPLIER, ge=1.0, le=4.0)
    # v3.0: Custom plugin name (when backend_type=custom).
    plugin_name: str = ""
    plugin_config: dict[str, Any] = Field(default_factory=dict, description="Custom plugin key-value settings. Values must be JSON-serializable primitives (str, int, float, bool, list, dict) — no arbitrary objects.")

    # v3.4: Speculative decoding (1.5–3× throughput improvement).
    use_speculative: bool = False
    speculative_mode: Literal["self", "draft_model"] = "self"
    speculative_num_tokens: int = Field(default=3, ge=1, le=16)
    speculative_draft_model: str = ""
    speculative_num_draft_layers: int = Field(default=0, ge=0, le=128)

    # v3.5: PagedAttention KV-cache (40-70% less memory, CUDA only).
    use_paged_attention: bool = False

    # v3.5: Continuous batching (higher throughput via dynamic batch).
    use_continuous_batching: bool = False

    # v3.4: QAT / Quantized model settings.
    # QAT model detection happens automatically in the autoregressive backend
    # via QAT_MODEL_KEYWORDS (constants.py) — no manual config field needed.

    # v3.6: Model quantization level (for bitsandbytes loading).
    quantization: Literal["bf16", "fp16", "int8", "int4"] = "bf16"

    # v3.7: Data parallelism — replicated copies across GPUs.
    data_parallel_size: int = Field(default=1, ge=1, le=8)

    # v3.6: NLLB encoder-decoder translation parameters.
    nllb_source_lang: str = "eng_Latn"
    nllb_target_lang: str = "tur_Latn"

    @model_validator(mode="after")
    def validate_model_config(self):
        """Validate field-level consistency after model construction.

        Performs these checks:
            1. Defaults tokenizer_path to model_path if left empty.
            2. Raises ValueError if do_sample=True with temperature=0.0 (meaningless).
            3. Logs a warning if diffusion-specific parameters are set on a non-diffusion
               backend (they will be silently ignored at runtime).
            4. Raises ValueError if speculative_mode='draft_model' is set without
               providing a speculative_draft_model path.

        Returns:
            self: The validated ModelConfig instance.

        Raises:
            ValueError: On constraint violations described above.

        Side effects:
            Mutates self.tokenizer_path if it was empty.
            Logs warnings for misconfigured diffusion params.
        """
        # 0) Default tokenizer_path to model_path when empty
        if not self.tokenizer_path:
            object.__setattr__(self, "tokenizer_path", self.model_path)

        # 1) do_sample=True with temperature=0.0 is meaningless
        if self.do_sample and self.temperature == 0.0:
            raise ValueError(
                "do_sample=True with temperature=0.0 is invalid: "
                "sampling requires temperature > 0."
            )

        # 2) Diffusion params are only meaningful for diffusion backends
        if self.backend_type != "diffusion":
            diffusion_defaults = {
                "diffusion_steps": DEFAULT_DIFFUSION_STEPS,
                "guidance_scale": DEFAULT_GUIDANCE_SCALE,
                "noise_schedule": DEFAULT_NOISE_SCHEDULE,
                "target_length_multiplier": DEFAULT_TARGET_LENGTH_MULTIPLIER,
            }
            non_default_diffusion = []
            for field_name, default_val in diffusion_defaults.items():
                current_val = getattr(self, field_name)
                if current_val != default_val:
                    non_default_diffusion.append(
                        f"{field_name}={current_val} (default: {default_val})"
                    )
            if non_default_diffusion:
                logger.warning(
                    "Non-default diffusion parameters set on a non-diffusion backend "
                    "(backend_type='%s'): %s. These parameters will be ignored.",
                    self.backend_type, ", ".join(non_default_diffusion),
                )

        # 4) speculative_mode='draft_model' requires speculative_draft_model
        if self.use_speculative and self.speculative_mode == "draft_model" and not self.speculative_draft_model:
            raise ValueError(
                "speculative_mode='draft_model' requires "
                "speculative_draft_model to be set."
            )

        return self


class RuntimeConfig(BaseModel):
    """Pydantic v2 model for runtime and benchmarking configuration.

    Fields:
        target_duration_seconds: How long the benchmark runs. Range 60-86400, default 7200 (2 hr).
        checkpoint_interval_seconds: Seconds between progress checkpoints. Range 30-3600, default 300.
        heartbeat_interval_seconds: Seconds between heartbeat log messages. Range 1-120, default 30.
        metrics_sample_rate_hz: Metrics collection frequency. Range 1-10, default 1.
        seed: Random seed for reproducibility. Must be non-negative, default 42.
        batch_size: Inference batch size. 0 means auto-tune. Range 0-32768.
        observability_enabled: Enable Prometheus observability endpoint. Default False.

    Model config forbids extra fields.
    """
    model_config = {"extra": "forbid"}
    target_duration_seconds: int = Field(default=7200, ge=60, le=86400)
    checkpoint_interval_seconds: int = Field(default=300, ge=30, le=3600)
    heartbeat_interval_seconds: int = Field(default=30, ge=1, le=120)
    metrics_sample_rate_hz: int = Field(default=1, ge=1, le=10)
    seed: int = Field(default=42, ge=0)
    batch_size: int = Field(default=0, ge=0, le=32768)  # 0 = auto-tune
    observability_enabled: bool = False


class DataConfig(BaseModel):
    """Pydantic v2 model for data pipeline and quality-benchmark configuration.

    Fields:
        input_paths: Glob patterns for input data files. Defaults to ["./data/input/*.jsonl.gz"].
        output_dir: Directory for inference output artifacts. Default "./output".
        reference_set_path: Path to golden-reference JSONL used for quality scoring.
        shard_size_mb: Maximum shard size before splitting. Range 10-1024 MB, default 100.
        prefetch_workers: Number of background data-loader workers. Range 1-16, default 4.
            Recommend 8 for H200 (64 cores); 4 is conservative (see M3.2, measured 2026-06-24).
        shuffle: Whether to shuffle input data. Default True.
        min_chunk_tokens: Minimum token count in a chunk for it to be processed. Default 10.
        max_garbage_ratio: Maximum ratio of garbage/token noise tolerated. 0.0-1.0, default 0.95.
        chunk_overlap_tokens: Token overlap between consecutive chunks. Range 0-256, default 50.
        max_references: Maximum number of references to load from the reference set. None = all.
        shuffle_max_memory_gb: RAM budget for in-memory shuffle buffer. When the estimated
            memory exceeds this, falls back to disk-backed external sort. None uses the default
            SHUFFLE_MEMORY_BUDGET_BYTES constant (2 GiB). Range 0.1-512.
        shuffle_temp_dir: Directory for external-sort temporary run files. Empty string
            uses the system temp directory (TMPDIR or /tmp).

    Model config forbids extra fields.
    """
    model_config = {"extra": "forbid"}
    input_paths: list[str] = Field(default_factory=lambda: ["./data/input/*.jsonl.gz"])
    output_dir: str = "./output"
    reference_set_path: str = "./data/references/golden_en_tr.jsonl"
    shard_size_mb: int = Field(default=100, ge=10, le=1024)
    prefetch_workers: int = Field(default=4, ge=1, le=16)  # measured 2026-06-24: recommend 8 for H200 (64 cores), 4 is conservative. See M3.2.
    shuffle: bool = True
    min_chunk_tokens: int = Field(default=10, ge=1)
    max_garbage_ratio: float = Field(default=0.95, ge=0.0, le=1.0)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=256)
    max_references: Optional[int] = Field(default=None, ge=1)
    # External shuffle: maximum RAM budget for the in-memory shuffle buffer.
    # When the estimated text size × 2 (overhead multiplier) exceeds this,
    # the loader switches to a disk-backed external sort.  None = use the
    # SHUFFLE_MEMORY_BUDGET_BYTES constant (2 GiB).
    shuffle_max_memory_gb: Optional[float] = Field(default=None, ge=0.1, le=512.0)
    # Directory for external-sort temporary run files.  Empty string = use
    # the system temp directory (TMPDIR / /tmp).
    shuffle_temp_dir: str = Field(default="", max_length=500)
    _nonlocal_schemes: tuple[str, ...] = ("s3://", "gs://", "hdfs://", "r2://", "wasbs://")


class ExtrapolationConfig(BaseModel):
    """Pydantic v2 model for cost/throughput extrapolation parameters.

    Fields:
        total_clearnet_non_tr_tokens: Estimated total clean-web non-Turkish tokens
            available for processing. Default 200 billion, minimum 1 million.
        gpu_cost_per_hour_usd: Cloud-equivalent GPU hourly cost (e.g., ~$3.00/GPU-hr
            for H200 on-demand, see M0.4). None means cost extrapolation is disabled.

    Model config forbids extra fields.
    """
    model_config = {"extra": "forbid"}
    total_clearnet_non_tr_tokens: int = Field(default=200_000_000_000, ge=1_000_000)
    gpu_cost_per_hour_usd: Optional[float] = None  # cloud-equivalent ~$3.00/GPU-hour for H200 on-demand. See M0.4.


class BenchmarkConfig(BaseModel):
    """Pydantic v2 root configuration model combining all sub-configs.

    Fields:
        backend: Hardware backend to use: "auto" (detect), "cuda", "mps", or "cpu".
            Default "auto".
        model: Per-model configuration (ModelConfig). Defaults to a fresh ModelConfig.
        runtime: Benchmark runtime parameters (RuntimeConfig).
        data: Data pipeline and quality-eval parameters (DataConfig).
        extrapolation: Throughput/cost extrapolation parameters (ExtrapolationConfig).

    Both frozen=True and extra="forbid" are set:
        - Extra keys in YAML will raise ValidationError.
        - Fields cannot be reassigned after construction (explicit freeze).

    Validators:
        validate_benchmark_config: Post-construction consistency checks (see method docstring).
    """
    backend: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    extrapolation: ExtrapolationConfig = Field(default_factory=ExtrapolationConfig)
    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def validate_benchmark_config(self):
        """Validate root-level consistency after all sub-models are built.

        Performs these checks:
            1. When data.input_paths contains only local paths (no s3://, gs://, etc.),
               verifies that at least one file is matched by the glob patterns.
               Raises ValueError if zero files are found.
            2. Checks whether data.reference_set_path exists. If missing, logs a warning
               that quality benchmark will be skipped (but does NOT raise - quality is optional).

        Returns:
            self: The validated BenchmarkConfig instance.

        Raises:
            ValueError: If no local input files are matched by input_paths globs.

        Side effects:
            Logs warnings for missing reference_set_path.
        """
        # 1) Ensure data.input_paths resolves to at least one file
        #    (only for local paths — skip cloud / remote schemes)
        data = self.data
        if data.input_paths:
            all_local = True
            for p in data.input_paths:
                if p.lower().startswith(data._nonlocal_schemes):
                    all_local = False
                    break
            if all_local:
                found = []
                for p in data.input_paths:
                    found.extend(glob.glob(p, recursive=True))
                if not found:
                    raise ValueError(
                        f"No files matched by data.input_paths={data.input_paths}. "
                        "Either fix the glob pattern or set input_paths to a valid path. "
                        "Local paths must resolve to at least one file."
                    )

        # 4) Ensure data.reference_set_path exists (local only)
        ref_path = Path(data.reference_set_path)
        if not ref_path.exists():
            logger.warning(
                "data.reference_set_path='%s' does not exist — "
                "quality benchmark will be skipped.",
                data.reference_set_path,
            )

        return self


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a BenchmarkConfig from a YAML file.

    Args:
        path: Path to a YAML configuration file. Accepts str or pathlib.Path.

    Returns:
        BenchmarkConfig: Validated configuration object. If the file is empty
        or contains only comments, all defaults are used.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValidationError (pydantic): If the YAML contains unknown fields or
        violates field constraints.

    Side effects:
        Logs the loaded path at INFO level.
        Logs a warning and falls back to defaults if the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        logger.warning(
            "Config file '%s' is empty or contains only comments — "
            "falling back to all defaults.", path
        )
        raw = {}
    config = BenchmarkConfig(**raw)
    logger.info(f"Loaded config from {path}")
    return config
