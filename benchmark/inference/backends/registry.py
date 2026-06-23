"""Model registry — auto-detection, dispatch, and lifecycle management (v3.0).

Discovers the right ``InferenceBackend`` for a given model based on:
1. TensorRT engine — if use_tensorrt=true (CUDA-only, optional).
2. Explicit backend override in config (``model.backend_type: "diffusion"``).
3. Model architecture auto-detection (checks model config for diffusion markers).
4. Custom plugin registration (user-provided ``CustomModelPlugin``).
5. Fallback to autoregressive (safe default for any HF causal LM).

Auto-detection heuristics
--------------------------
=============== ============================================================
Model Type      Detection Signals
=============== ============================================================
DIFFUSION       - Model config has ``diffusion_steps`` or ``noise_schedule``.
                - Model name contains "diffusion", "ddpm", "mdlm".
                - Model class has a ``denoise`` or ``reverse_diffusion`` method.
                - Model config ``architectures`` includes "DiffusionLM".
AUTOREGRESSIVE  - Default fallback.  Any HF causal LM.
CUSTOM          - Registered via ``PluginRegistry``.
=============== ============================================================

Usage
-----
>>> registry = ModelRegistry()
>>> backend = registry.create_backend(config)
>>> backend.load()
>>> result = backend.translate_batch(batch)
"""

# WARNING: _detect_model_type() calls AutoConfig.from_pretrained() which downloads
# model configs from HuggingFace Hub. This is network I/O in the constructor path.
# For offline use, set backend_type explicitly in config.

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch

from benchmark.inference.backends.protocol import (
    BackendConfig,
    InferenceBackend,
    ModelType,
)
from benchmark.config.constants import DIFFUSION_KEYWORDS

logger = logging.getLogger(__name__)


class _LRUCache(dict):
    """A bounded dict that evicts the oldest entries when the size limit is exceeded."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._order: list[str] = []
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            self._order.remove(key)
        elif len(self._order) >= self.max_size:
            oldest = self._order.pop(0)
            super().__delitem__(oldest)
        self._order.append(key)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self._order.remove(key)
        super().__delitem__(key)


# Cache model configs to avoid re-downloading on auto-detect.
_CONFIG_CACHE: _LRUCache = _LRUCache(max_size=100)


def clear_config_cache() -> None:
    """Clear all cached HuggingFace model configs."""
    _CONFIG_CACHE.clear()
    _CONFIG_CACHE._order.clear()


class ModelRegistry:
    """Discovers and creates the appropriate inference backend.

    Spring-loaded pattern: each ``create_backend()`` call returns a
    fully-configured backend instance ready for ``.load()``.
    """

    def __init__(self):
        self._backends: dict[ModelType, type[InferenceBackend]] = {}

        # Register built-in backends lazily.
        self._register_builtin()

    def _register_builtin(self) -> None:
        """Register the built-in backend implementations."""
        try:
            from benchmark.inference.backends.autoregressive import AutoregressiveBackend
            self._backends[ModelType.AUTOREGRESSIVE] = AutoregressiveBackend
        except ImportError:
            pass

        try:
            from benchmark.inference.backends.diffusion import DiffusionBackend
            self._backends[ModelType.DIFFUSION] = DiffusionBackend
        except ImportError:
            pass

        try:
            from benchmark.inference.backends.nllb import NLLBBackend
            self._backends[ModelType.ENCODER_DECODER] = NLLBBackend
        except ImportError:
            pass

        # Custom backends are discovered via PluginRegistry, not here.

    def register(self, model_type: ModelType, backend_cls: type[InferenceBackend]) -> None:
        """Register a backend class for a model type.

        Plugin authors call this to make their custom backends discoverable.

        Parameters
        ----------
        model_type : ModelType
            The model architecture category.
        backend_cls : type[InferenceBackend]
            A concrete subclass of InferenceBackend.
        """
        if not issubclass(backend_cls, InferenceBackend):
            raise TypeError(
                f"backend_cls must be a subclass of InferenceBackend, "
                f"got {backend_cls.__name__}"
            )
        self._backends[model_type] = backend_cls
        logger.info(
            "Registered backend '%s' for %s models",
            backend_cls.display_name, model_type.value,
        )

    def create_backend(self, config: BackendConfig) -> InferenceBackend:
        """Create the appropriate backend for the given configuration.

        Dispatch order:
        0. TensorRT — if use_tensorrt=true in config (CUDA-only, experimental).
        1. Explicit ``backend_type`` in ``config.extra``.
        2. Auto-detect from model architecture.
        3. Custom plugin lookup.
        4. Fallback to autoregressive.

        TensorRT is tried as an upgrade to the AR path. If TensorRT is
        unavailable or engine build fails, falls back gracefully to the
        extreme-optimized AutoregressiveBackend.

        Returns
        -------
        InferenceBackend
            A concrete backend instance (not yet loaded).
        """
        # ── 0. TensorRT upgrade (v3.3) ──
        trt_cfg = config.extra.get("tensorrt", {})
        use_trt = config.extra.get("use_tensorrt", False) or bool(trt_cfg)

        if use_trt and torch.cuda.is_available():
            try:
                from benchmark.inference.backends.tensorrt_backend import TensorRTBackend
                trt_backend = TensorRTBackend.create(config)
                if trt_backend is not None:
                    logger.info(
                        "TensorRT engine ready — using TensorRTBackend "
                        "(precision=%s)", trt_cfg.get("precision", "fp16"),
                    )
                    return trt_backend
                else:
                    logger.info(
                        "TensorRT unavailable — falling back to AutoregressiveBackend"
                    )
            except Exception as e:
                logger.info(
                    "TensorRT backend creation failed (%s) — "
                    "falling back to AutoregressiveBackend", e,
                )

        # ── 1. Explicit override (skip "auto" — it means auto-detect) ──
        explicit = config.extra.get("backend_type")
        if explicit and explicit != "auto":
            try:
                model_type = ModelType(explicit)
            except ValueError:
                logger.warning("Unknown backend_type '%s' — falling back to auto-detect", explicit)
            else:
                if model_type in self._backends:
                    logger.info("Using explicit backend: %s", model_type.value)
                    return self._backends[model_type](config)

        # ── 2. Auto-detect ──
        detected = self._detect_model_type(config.model_path)
        if detected != ModelType.AUTOREGRESSIVE:
            logger.info("Auto-detected model type: %s", detected.value)
            if detected in self._backends:
                return self._backends[detected](config)

        # ── 3. Custom plugin lookup ──
        try:
            from benchmark.inference.backends.custom_plugin import PluginRegistry
            plugin = PluginRegistry.lookup(config.model_path)
            if plugin is not None:
                logger.info("Using custom plugin: %s", plugin.name)
                return plugin.create_backend(config)
        except ImportError:
            pass

        # ── 4. Fallback: autoregressive ──
        logger.info("Falling back to autoregressive backend")
        if ModelType.AUTOREGRESSIVE in self._backends:
            return self._backends[ModelType.AUTOREGRESSIVE](config)

        raise RuntimeError(
            "No inference backend available.  Install at least one backend "
            "or register a custom plugin."
        )

    def _detect_model_type(self, model_path: str) -> ModelType:
        """Auto-detect the model architecture from the model path or config.

        Heuristics:
        - Check model name for diffusion keywords.
        - If it's a local path, check model config.
        - For HuggingFace Hub IDs, try to fetch the config.
        - Default to AUTOREGRESSIVE.

        Returns
        -------
        ModelType
        """
        name_lower = model_path.lower()

        # ── Name-based heuristics: NLLB ──
        if "nllb" in name_lower:
            logger.info("NLLB encoder-decoder model detected via name: '%s'", model_path)
            return ModelType.ENCODER_DECODER

        # ── Name-based heuristics: diffusion ──
        for kw in DIFFUSION_KEYWORDS:
            if kw in name_lower:
                logger.info("Diffusion model detected via name keyword: '%s'", kw)
                return ModelType.DIFFUSION

        # ── Config-based detection (local paths) ──
        local_path = Path(model_path)
        if local_path.exists():
            config_file = local_path / "config.json"
            if config_file.exists():
                try:
                    import json
                    with open(config_file) as f:
                        cfg = json.load(f)

                    # Check model_type field.
                    if cfg.get("model_type") in ("diffusion", "diffusion_gemma"):
                        return ModelType.DIFFUSION

                    # Check architectures for encoder-decoder (NLLB, M2M100, BART, T5).
                    archs = cfg.get("architectures", [])
                    _ENCODER_DECODER_ARCHITECTURES = (
                        "M2M100ForConditionalGeneration",
                        "NllbMoeForConditionalGeneration",
                        "BartForConditionalGeneration",
                        "FSMTForConditionalGeneration",
                        "T5ForConditionalGeneration",
                    )
                    for arch in archs:
                        if arch in _ENCODER_DECODER_ARCHITECTURES:
                            logger.info(
                                "Encoder-decoder architecture detected: '%s'", arch,
                            )
                            return ModelType.ENCODER_DECODER

                    # Check for diffusion-specific config keys.
                    diffusion_config_keys = [
                        "diffusion_steps", "noise_schedule",
                        "num_diffusion_steps", "num_timesteps",
                        "diffusion_config",
                    ]
                    for key in diffusion_config_keys:
                        if key in cfg:
                            logger.info("Diffusion config key detected: '%s'", key)
                            return ModelType.DIFFUSION

                    # Check architectures.
                    archs = cfg.get("architectures", [])
                    for arch in archs:
                        arch_lower = arch.lower()
                        if any(kw in arch_lower for kw in DIFFUSION_KEYWORDS):
                            logger.info("Diffusion architecture detected: '%s'", arch)
                            return ModelType.DIFFUSION

                except Exception as e:
                    logger.debug("Config-based detection failed: %s", e)

        # ── HuggingFace Hub config check (cached) ──
        if "/" in model_path and not local_path.exists():
            try:
                cfg = self._get_hf_config(model_path)
                if cfg:
                    # Check model_type for diffusion indicators.
                    if cfg.get("model_type") in ("diffusion", "diffusion_gemma"):
                        logger.info("HF config model_type='%s'", cfg["model_type"])
                        return ModelType.DIFFUSION

                    # Check architectures for diffusion keywords.
                    archs = cfg.get("architectures", [])
                    for arch in archs:
                        arch_lower = arch.lower()
                        if any(kw in arch_lower for kw in DIFFUSION_KEYWORDS):
                            logger.info("HF config diffusion arch: '%s'", arch)
                            return ModelType.DIFFUSION

                    diffusion_keys = [
                        "diffusion_steps", "noise_schedule",
                        "num_diffusion_steps", "num_timesteps",
                        "diffusion_config",
                    ]
                    for key in diffusion_keys:
                        if key in cfg:
                            logger.info("HF config diffusion key: '%s'", key)
                            return ModelType.DIFFUSION
            except OSError:
                logger.error(
                    "network unavailable — cannot auto-detect model type. "
                    "Set backend_type in config."
                )
            except Exception:
                pass

        return ModelType.AUTOREGRESSIVE

    def _get_hf_config(self, model_id: str) -> Optional[dict]:
        """Fetch model config from HuggingFace Hub (cached).

        Retries up to 3 times with exponential backoff to handle transient
        network failures.
        """
        if model_id in _CONFIG_CACHE:
            return _CONFIG_CACHE[model_id]

        from transformers import AutoConfig

        max_attempts = 3
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                cfg = AutoConfig.from_pretrained(
                    model_id, trust_remote_code=False,
                    timeout=30,
                )  # Security: remote code execution disabled
                _CONFIG_CACHE[model_id] = cfg.to_dict()
                return _CONFIG_CACHE[model_id]
            except Exception as e:
                last_exc = e
                if attempt < max_attempts:
                    delay = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    logger.debug(
                        "HF config fetch attempt %d/%d failed for %s: %s; "
                        "retrying in %ds",
                        attempt, max_attempts, model_id, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.debug(
                        "Could not fetch HF config for %s after %d attempts: %s",
                        model_id, max_attempts, e,
                    )

        return None

    def list_available_backends(self) -> list[dict]:
        """Return metadata about all registered backends."""
        result = []
        for model_type, cls in self._backends.items():
            result.append({
                "model_type": model_type.value,
                "display_name": cls.display_name,
                "capabilities": [c.name for c in type(cls.capabilities) if c in cls.capabilities]
                if hasattr(cls.capabilities, '__iter__') else [],
                "class": cls.__name__,
            })

        # Add custom plugins.
        try:
            from benchmark.inference.backends.custom_plugin import PluginRegistry
            for plugin_name in PluginRegistry.list_plugins():
                result.append({
                    "model_type": "custom",
                    "display_name": plugin_name,
                    "capabilities": [],
                    "class": "CustomPlugin",
                })
        except ImportError:
            pass

        return result
