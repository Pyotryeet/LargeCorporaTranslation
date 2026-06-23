"""Model-agnostic inference backends (v3.0).

Pluggable architecture supporting:
- Autoregressive models (TranslateGemma, GPT, LLaMA, etc.)
- Diffusion LLM models (MDLM, DiffusionBERT, Dream, etc.)
- Custom user-defined models via the plugin registry

Dispatch is automatic based on model architecture detection.
"""

from benchmark.inference.backends.protocol import (
    InferenceBackend,
    ModelCapability,
    ModelType,
    GenerationOutput,
)
from benchmark.inference.backends.registry import ModelRegistry
from benchmark.inference.backends.custom_plugin import (
    CustomModelPlugin,
    PluginRegistry,
    register_plugin,
)
from benchmark.inference.backends.autoregressive import AutoregressiveBackend
from benchmark.inference.backends.diffusion import DiffusionBackend
from benchmark.inference.backends.tensorrt_backend import TensorRTBackend

__all__ = [
    "InferenceBackend",
    "ModelCapability",
    "ModelType",
    "GenerationOutput",
    "ModelRegistry",
    "CustomModelPlugin",
    "PluginRegistry",
    "register_plugin",
    "AutoregressiveBackend",
    "DiffusionBackend",
    "TensorRTBackend",
]
