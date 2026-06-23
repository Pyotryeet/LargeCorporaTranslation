"""Environment snapshot — captures dependency versions for reproducibility."""

import hashlib
import os
import platform
import sys
from typing import Any

import torch

VERSION = "3.6"

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
