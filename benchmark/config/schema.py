"""Pydantic v2 configuration models with strict validation."""

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
    backend_type: Literal["auto", "autoregressive", "encoder_decoder", "diffusion", "custom"] = "auto"
    # v3.0: Diffusion-specific parameters.
    diffusion_steps: int = Field(default=DEFAULT_DIFFUSION_STEPS, ge=8, le=4096)
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, ge=1.0, le=10.0)
    noise_schedule: Literal["cosine", "linear", "sqrt"] = DEFAULT_NOISE_SCHEDULE
    target_length_multiplier: float = Field(default=DEFAULT_TARGET_LENGTH_MULTIPLIER, ge=1.0, le=4.0)
    # v3.0: Custom plugin name (when backend_type=custom).
    plugin_name: str = ""
    plugin_config: dict[str, Any] = Field(default_factory=dict, description="Custom plugin key-value settings. Values must be JSON-serializable primitives (str, int, float, bool, list, dict) — no arbitrary objects.")

    # v3.3: TensorRT engine optimization (CUDA only).
    use_tensorrt: bool = False
    tensorrt_precision: Literal["fp16", "fp8", "int8"] = "fp16"
    tensorrt_max_batch: int = Field(default=32, ge=1, le=256)
    tensorrt_cache_dir: str = ""

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

    # v3.6: NLLB encoder-decoder translation parameters.
    nllb_source_lang: str = "eng_Latn"
    nllb_target_lang: str = "tur_Latn"

    @model_validator(mode="after")
    def validate_model_config(self):
        # 0) Default tokenizer_path to model_path when empty
        if not self.tokenizer_path:
            object.__setattr__(self, "tokenizer_path", self.model_path)

        # 1) do_sample=True with temperature=0.0 is meaningless
        if self.do_sample and self.temperature == 0.0:
            raise ValueError(
                "do_sample=True with temperature=0.0 is invalid: "
                "sampling requires temperature > 0."
            )

        # 2) Diffusion and TensorRT are mutually exclusive
        if self.backend_type == "diffusion" and self.use_tensorrt:
            raise ValueError(
                "backend_type='diffusion' and use_tensorrt=True are mutually exclusive."
            )

        # 2a) Validate tensorrt_cache_dir when TensorRT is enabled
        if self.use_tensorrt and self.tensorrt_cache_dir:
            cache_path = Path(self.tensorrt_cache_dir)
            if not cache_path.exists():
                try:
                    cache_path.mkdir(parents=True, exist_ok=True)
                    logger.info("Created TensorRT cache directory: %s", cache_path)
                except OSError as exc:
                    raise ValueError(
                        f"tensorrt_cache_dir='{self.tensorrt_cache_dir}' does not exist "
                        f"and could not be created: {exc}"
                    ) from exc

        # 3) FP8 TensorRT requires H200/Hopper (SM 9.0+)
        if self.use_tensorrt and self.tensorrt_precision == "fp8":
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.is_initialized():
                    major, minor = torch.cuda.get_device_capability()
                    if major < 9:
                        logger.warning(
                            f"tensorrt_precision='fp8' requires Hopper/H200 (SM 9.0+), "
                            f"but detected SM {major}.{minor}. FP8 may not be supported."
                        )
                else:
                    logger.debug(
                        "Skipping FP8 GPU capability check: CUDA is not initialized."
                    )
            except (ImportError, RuntimeError, AttributeError) as e:
                logger.debug("Skipping FP8 GPU capability check: %s", e)

        # 3a) Diffusion params are only meaningful for diffusion backends
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
    target_duration_seconds: int = Field(default=7200, ge=60, le=86400)
    checkpoint_interval_seconds: int = Field(default=300, ge=30, le=3600)
    heartbeat_interval_seconds: int = Field(default=30, ge=1, le=120)
    metrics_sample_rate_hz: int = Field(default=1, ge=1, le=10)
    seed: int = Field(default=42, ge=0)
    observability_enabled: bool = False


class DataConfig(BaseModel):
    input_paths: list[str] = Field(default_factory=lambda: ["./data/input/*.jsonl.gz"])
    output_dir: str = "./output"
    reference_set_path: str = "./data/references/golden_en_tr.jsonl"
    shard_size_mb: int = Field(default=100, ge=10, le=1024)
    prefetch_workers: int = Field(default=4, ge=1, le=16)
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
    total_clearnet_non_tr_tokens: int = Field(default=6_230_000_000_000, ge=1_000_000)
    gpu_cost_per_hour_usd: Optional[float] = None


class BenchmarkConfig(BaseModel):
    backend: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    extrapolation: ExtrapolationConfig = Field(default_factory=ExtrapolationConfig)
    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def validate_benchmark_config(self):
        # 1) MPS backend + TensorRT is impossible
        if self.backend == "mps" and self.model.use_tensorrt:
            raise ValueError(
                "backend='mps' is incompatible with model.use_tensorrt=True. "
                "TensorRT requires NVIDIA CUDA GPUs."
            )

        # 2) TensorRT requires CUDA backend
        if self.model.use_tensorrt and self.backend != "cuda":
            raise ValueError(
                f"model.use_tensorrt=True requires backend='cuda', "
                f"got backend='{self.backend}'."
            )

        # 3) Ensure data.input_paths resolves to at least one file
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
