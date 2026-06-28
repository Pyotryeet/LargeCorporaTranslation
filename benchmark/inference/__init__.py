"""Inference package — model loading, batch assembly, translation (v3.0).

v3.0: Model-agnostic architecture with pluggable backends.
v3.5: speculative decoding, PagedAttention, continuous batching (gated behind CLI flags).
"""

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

from benchmark.inference.continuous_batcher import ContinuousBatcher

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
    "ContinuousBatcher",
]
