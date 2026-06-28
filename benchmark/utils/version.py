"""Environment snapshot — captures dependency versions for reproducibility."""

import hashlib
import os
import platform
import sys
from typing import Any

import torch

VERSION = "3.9"

# Canonical PEP 396 attribute — equivalent to VERSION (kept for compatibility).
__version__ = VERSION
__version_info__: tuple[int, ...] = tuple(int(p) for p in VERSION.split("."))

__all__ = ["VERSION", "__version__", "__version_info__", "get_environment_snapshot", "get_model_fingerprint"]

try:
    import transformers
    TRANSFORMERS_VERSION = transformers.__version__
except ImportError:
    TRANSFORMERS_VERSION = "not installed"


def get_model_fingerprint(model_path: str) -> str | None:
    """Compute sha256 fingerprint of a model's config.json for reproducibility.

    For remote/non-local model paths (e.g. HuggingFace Hub IDs like
    "google/gemma-3-12b-it"), returns the model ID string itself as the
    fingerprint so environment snapshots always carry an identifier.
    """
    if not os.path.isdir(model_path):
        # Remote model or non-local path (e.g. HuggingFace Hub ID like
        # "google/gemma-3-12b-it").  Return the model ID string itself
        # as the fingerprint so it is never None in environment snapshots.
        return model_path

    config_file = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_file):
        return None

    hasher = hashlib.sha256()
    with open(config_file, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_environment_snapshot(model_path: str | None = None) -> dict[str, Any]:
    """Build a reproducible dictionary snapshot of the runtime environment.

    Collects Python version, platform info, PyTorch/Transformers versions,
    CUDA and MPS availability, and optionally a model fingerprint.  The
    result is designed to be serialized alongside benchmark results so every
    measurement is traceable.

    Args:
        model_path: Optional path to a local model directory or a HuggingFace
            Hub model ID.  When provided, a ``model_fingerprint`` key is
            included in the returned snapshot (see
            :func:`get_model_fingerprint`).

    Returns:
        A ``dict[str, Any]`` with the following keys:

        * ``python_version`` — e.g. ``"3.12.4"``
        * ``platform`` — result of :func:`platform.platform`
        * ``pytorch_version`` — :attr:`torch.__version__`
        * ``transformers_version`` — :attr:`transformers.__version__` or
          ``"not installed"``
        * ``cuda_available`` — ``True`` / ``False``
        * ``mps_available`` — ``True`` / ``False``
        * ``model_fingerprint`` — present only when *model_path* is not None
          and the path exists locally as a directory containing
          ``config.json``, OR when *model_path* is a remote/non-local string
          (e.g. a Hub ID).  ``None`` when *model_path* points to a local
          directory that does not contain ``config.json``.
        * ``cuda_version`` — present only when CUDA is available
        * ``gpu_count`` — present only when CUDA is available
        * ``gpu_names`` — present only when CUDA is available

    Raises:
        No exceptions are raised directly.  Missing optional dependencies
        (e.g. Transformers) are handled silently with a fallback string.

    Side effects:
        None.  This function is pure (reads system attributes and optionally
        the filesystem).
    """
    snapshot: dict[str, Any] = {"python_version": sys.version.split()[0], "platform": platform.platform(),
                "pytorch_version": torch.__version__, "transformers_version": TRANSFORMERS_VERSION,
                "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available()}
    if model_path is not None:
        snapshot["model_fingerprint"] = get_model_fingerprint(model_path)
    if torch.cuda.is_available():
        snapshot["cuda_version"] = torch.version.cuda
        snapshot["gpu_count"] = torch.cuda.device_count()
        snapshot["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return snapshot
