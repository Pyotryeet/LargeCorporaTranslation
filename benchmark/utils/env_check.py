"""Pre-flight environment checks for benchmark runs.

Validates GPU count/memory, disk space, model-path existence, and reference-set
availability before a benchmark harness starts.  The single public entry point is
:func:`run_preflight_checks`, which raises :class:`RuntimeError` when any
hard-blocker check fails; soft warnings are logged for non-fatal conditions.
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_huggingface_hub_id(model_path: str) -> bool:
    """Check whether *model_path* is a remote HuggingFace Hub ID or model-preset name.

    Returns True when the path either (a) contains a ``/`` separator and does NOT
    exist on the local filesystem, or (b) matches a registered model preset from
    the ``benchmark.config.model_presets`` registry.

    Parameters
    ----------
    model_path : str
        A model specifier that may be a local path, a HuggingFace Hub ID (e.g.
        ``"facebook/nllb-200-distilled-600M"``), or a preset name (e.g.
        ``"translategemma-4b-bf16"``).

    Returns
    -------
    bool
        True if *model_path* should be treated as a remote/registered identifier
        rather than a local filesystem path.

    Notes
    -----
    - This function performs a guarded dynamic import of
      ``benchmark.config.model_presets.get_preset_by_name``; import errors are
      silently swallowed and treated as "not a preset".
    - Callers use this to decide whether to skip local-filesystem existence checks
      for models that will be fetched from the Hub at load time.
    """
    if "/" in model_path and not Path(model_path).exists():
        return True
    # Also accept registered model preset names (e.g. 'translategemma-4b-bf16').
    try:
        from benchmark.config.model_presets import get_preset_by_name
        if get_preset_by_name(model_path) is not None:
            return True
    except ImportError:
        pass
    return False


def run_preflight_checks(config, device_info, dry_run: bool = False) -> None:
    """Run a battery of environment checks before starting a benchmark run.

    Checks performed (warnings are logged, hard failures raise RuntimeError):

    * **GPU count** — CUDA only: warns if fewer than 2 GPUs are detected (single-GPU
      runs are supported for dev/test but >1 is recommended for production).
    * **GPU memory per device** — CUDA only: warns when the per-GPU total is below 80 GB
      (the H200 minimum).
    * **Unified memory** — MPS only: warns when total unified memory is below 16 GB.
    * **Disk space** — checks free space on the configured output directory; the
      required amount is 10 GB for MPS/CPU, 20 GB for CUDA, or 10 GB in dry-run mode.
      Falls back to the CWD if the output directory does not exist yet.
    * **Model-path existence** — raises RuntimeError if the path does not exist on
      disk and does not look like a HuggingFace Hub ID or a registered model-preset
      name.
    * **Reference-set existence** — warns (does not fail) if the reference-set file
      is missing; the quality benchmark step will be skipped.

    Parameters
    ----------
    config : BenchmarkConfig
        The fully-resolved benchmark configuration object. Must have ``model.model_path``,
        ``data.output_dir``, and ``data.reference_set_path`` attributes.
    device_info : DeviceInfo
        A named-tuple or object with ``backend`` (str, one of ``"cuda"``, ``"mps"``,
        ``"cpu"``), ``total_memory_gb`` (float), and ``num_devices`` (int) attributes.
    dry_run : bool, default False
        If True, the disk-space threshold is relaxed to 10 GB regardless of backend.

    Raises
    ------
    RuntimeError
        If any hard-blocker check fails (disk space below minimum, model path not
        found). The message includes all failed checks separated by semicolons.

    Notes
    -----
    - The MPS and CUDA memory checks are **warnings only** — they do not prevent the
      run from starting because some models (e.g. NLLB-600M) can run on smaller GPUs.
    - The disk-space check catches OSError and falls back to a logged warning rather
      than crashing, since disk-usage queries can fail on some filesystems.
    """
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
