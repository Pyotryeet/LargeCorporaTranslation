"""Pre-flight environment checks — validates hardware and disk space."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_huggingface_hub_id(model_path: str) -> bool:
    """Check if model_path looks like a HuggingFace Hub ID (e.g. 'org/model')."""
    return "/" in model_path and not Path(model_path).exists()


def run_preflight_checks(config, device_info, dry_run: bool = False) -> None:
    issues = []
    total_mem = device_info.total_memory_gb
    if device_info.backend == "cuda":
        if device_info.num_devices < 2:
            logger.warning(
                "CUDA mode: %d GPU(s) detected. >=2 GPUs recommended for "
                "production benchmarking; single-GPU runs are supported for "
                "development and testing.", device_info.num_devices
            )
        per_device = total_mem / device_info.num_devices if device_info.num_devices > 0 else 0
        if per_device < 80:
            logger.warning(f"Per-GPU memory ({per_device:.0f}GB) is below H200 minimum (80GB)")
    elif device_info.backend == "mps":
        if total_mem < 16:
            logger.warning(
                f"MPS: unified memory is {total_mem:.0f}GB — 16GB+ recommended for "
                f"small models (4B); larger models may require 32GB+."
            )
    output_dir = Path(config.data.output_dir)
    try:
        free = shutil.disk_usage(output_dir if output_dir.exists() else Path.cwd()).free / (1024**3)
        # Disk estimates account for model weights + tokenizer + output data.
        # MPS/CPU: models can be 2-30+ GB on disk depending on size (4B to 12B+).
        # CUDA: larger models (7B+) can be 15-30 GB.
        required = 10 if device_info.backend in ("mps", "cpu") else 20
        if dry_run:
            required = 10  # Dry-run needs almost nothing
        if free < required:
            issues.append(f"Insufficient disk space: {free:.0f}GB free, need {required}GB")
    except OSError:
        logger.warning(
            "Could not check disk usage — skipping disk space check. "
            "Ensure at least 10 GB of free space is available or the run may fail "
            "with 'No space left on device'.",
        )
    model_path = str(config.model.model_path)
    if not Path(model_path).exists() and not _is_huggingface_hub_id(model_path):
        issues.append(f"Model path not found: {config.model.model_path}")
    ref_path = Path(config.data.reference_set_path)
    if not ref_path.exists():
        logger.warning(f"Reference set not found: {ref_path} — quality benchmark will be skipped")
    if issues:
        for issue in issues:
            logger.error(f"PREFLIGHT FAIL: {issue}")
        raise RuntimeError(f"Pre-flight checks failed: {'; '.join(issues)}")
    logger.info("Pre-flight checks passed")
