"""Inference package — model loading, batch assembly, translation (v3.0).

v3.0: Model-agnostic architecture with pluggable backends.
v2.0: speculative decoding, PagedAttention (experimental continuous batching behind env var).
"""

import os

from benchmark.inference.engine import InferenceEngine, TranslationResult, BatchResult
from benchmark.inference.batch_assembly import BatchAssembler
from benchmark.inference.batch_tuner import BatchSizeTuner
from benchmark.inference.sampling import DecodingParams, TEMPERATURE_EPSILON

# v3.0: Backend protocol and registry.
from benchmark.inference.backends.protocol import (
    InferenceBackend,
    ModelCapability,
    ModelType,
    GenerationOutput,
    BackendConfig,
)
from benchmark.inference.backends.registry import ModelRegistry
from benchmark.inference.backends.custom_plugin import (
    CustomModelPlugin,
    PluginRegistry,
    register_plugin,
)

# Experimental continuous batching — only import when explicitly opted in.
# This module has known KV-cache correctness issues and is NOT wired into
# the inference hot path.
if os.environ.get("TR_ENABLE_CONTINUOUS_BATCHING") == "1":
    from benchmark.inference.continuous_batcher import ContinuousBatcher  # noqa: F401

_DEFAULT_ALL = [
    # Engine facade
    "InferenceEngine", "TranslationResult", "BatchResult",
    "BatchAssembler", "BatchSizeTuner", "DecodingParams",
    # Backend protocol (for custom model authors)
    "InferenceBackend", "ModelCapability", "ModelType",
    "GenerationOutput", "BackendConfig",
    # Registry & plugins
    "ModelRegistry", "CustomModelPlugin", "PluginRegistry",
    "register_plugin",
]

# Extend __all__ with the experimental batcher only when opted in.
_EXPERIMENTAL_ALL = ["ContinuousBatcher"] if os.environ.get("TR_ENABLE_CONTINUOUS_BATCHING") == "1" else []
__all__ = _DEFAULT_ALL + _EXPERIMENTAL_ALL
