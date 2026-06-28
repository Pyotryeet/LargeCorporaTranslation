"""Model registry — auto-detection, dispatch, and lifecycle management (v3.0).

Discovers the right ``InferenceBackend`` for a given model based on:
1. Explicit backend override in config (``model.backend_type: "diffusion"``).
2. Model architecture auto-detection (checks model config for diffusion markers).
3. Custom plugin registration (user-provided ``CustomModelPlugin``).
4. Fallback to autoregressive (safe default for any HF causal LM).

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


class _FIFOCache(dict):
    """A bounded dict that evicts the oldest entries (insertion-order, FIFO).

    NOTE: Despite the common pattern, this is NOT an LRU cache — it uses
    insertion order, not access order, for eviction. Renamed from
    _LRUCache to reflect actual behaviour.

    Parameters
    ----------
    max_size : int, default=100
        Maximum number of entries before FIFO eviction begins.

    Returns
    -------
    None
        Initializes an empty dict with bounded capacity.

    Side Effects
    ------------
    - __setitem__ evicts the oldest key when capacity is exceeded.
    - __delitem__ removes the key from the internal insertion-order list.
    """

    def __init__(self, max_size: int = 100):
        """Initialize the bounded FIFO cache.

        Parameters
        ----------
        max_size : int, default=100
            Maximum number of entries before the oldest is evicted on insertion.

        Returns
        -------
        None
            Initializes the cache with empty storage and an empty order list.
        """
        self.max_size = max_size
        self._order: list[str] = []
        super().__init__()

    def __setitem__(self, key, value):
        """Set a key-value pair, evicting the oldest entry if at capacity.

        If the key already exists, its insertion position is updated to the end
        (refreshing its FIFO order). If the cache is at capacity, the oldest key
        (by insertion order) is evicted before the new entry is inserted.

        Parameters
        ----------
        key : hashable
            The dictionary key.
        value : any
            The value to associate with the key.

        Returns
        -------
        None

        Side Effects
        ------------
        - May evict the oldest key if the cache is full.
        - Reorders the key's position in the FIFO order list.
        """
        if key in self:
            self._order.remove(key)
        elif len(self._order) >= self.max_size:
            oldest = self._order.pop(0)
            super().__delitem__(oldest)
        self._order.append(key)
        super().__setitem__(key, value)

    def __delitem__(self, key):
        """Delete a key from the cache and remove it from the FIFO order.

        Parameters
        ----------
        key : hashable
            The key to delete.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the key does not exist in the cache.

        Side Effects
        ------------
        - Removes the key from the internal insertion-order list.
        """
        self._order.remove(key)
        super().__delitem__(key)


# Cache model configs to avoid re-downloading on auto-detect.
_CONFIG_CACHE: _FIFOCache = _FIFOCache(max_size=100)

# Known encoder-decoder architectures for auto-detection.
# Defined at module scope to avoid re-creating the tuple on every
# _detect_model_type() call.
_ENCODER_DECODER_ARCHITECTURES = (
    "M2M100ForConditionalGeneration",
    "NllbMoeForConditionalGeneration",
    "BartForConditionalGeneration",
    "FSMTForConditionalGeneration",
    "T5ForConditionalGeneration",
)

# Keys in a model config.json that indicate a diffusion model.
# Defined at module scope so the list is created once, not on every
# _detect_model_type() call (the method checks both local and HF config paths).
_DIFFUSION_CONFIG_KEYS = (
    "diffusion_steps",
    "noise_schedule",
    "num_diffusion_steps",
    "num_timesteps",
    "diffusion_config",
)


def clear_config_cache() -> None:
    """Clear all cached HuggingFace model configs.

    Removes all entries from the module-level _CONFIG_CACHE and resets the
    FIFO order list. Subsequent calls to _detect_model_type will re-fetch
    configs from disk or network.

    Returns
    -------
    None

    Side Effects
    ------------
    Mutates the module-level _FIFOCache instance by clearing both the dict
    contents and the internal order list.
    """
    _CONFIG_CACHE.clear()
    _CONFIG_CACHE._order.clear()


class ModelRegistry:
    """Discovers and creates the appropriate inference backend.

    Spring-loaded pattern: each ``create_backend()`` call returns a
    fully-configured backend instance ready for ``.load()``.

    On construction, lazily registers built-in backend implementations
    (autoregressive, diffusion, NLLB, vLLM) if their modules are importable.
    Custom plugins are discovered through PluginRegistry at dispatch time,
    not at construction.

    Parameters
    ----------
    None
        Takes no constructor arguments.

    Attributes
    ----------
    _backends : dict[ModelType, type[InferenceBackend]]
        Mapping from ModelType enum variant to backend class.
    """

    def __init__(self):
        self._backends: dict[ModelType, type[InferenceBackend]] = {}

        # Register built-in backends lazily.
        self._register_builtin()

    def _register_builtin(self) -> None:
        """Register the built-in backend implementations.

        Attempts to import and register AutoregressiveBackend, DiffusionBackend,
        NLLBBackend, and VLLMBackend. Import failures are silently ignored — each
        backend is optional and its absence does not prevent the registry from
        operating with the remaining backends.

        Returns
        -------
        None

        Side Effects
        ------------
        Populates self._backends with ModelType-to-backend-class mappings for
        each successfully imported module.
        """
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

        try:
            from benchmark.inference.backends.vllm import VLLMBackend
            self._backends[ModelType.VLLM] = VLLMBackend
        except ImportError:
            pass

        # Custom backends are discovered via PluginRegistry, not here.

    def register(self, model_type: ModelType, backend_cls: type[InferenceBackend]) -> None:
        """Register a backend class for a model type.

        Plugin authors call this to make their custom backends discoverable.

        Parameters
        ----------
        model_type : ModelType
            The model architecture category this backend handles.
        backend_cls : type[InferenceBackend]
            A concrete subclass of InferenceBackend.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If backend_cls is not a subclass of InferenceBackend.

        Side Effects
        ------------
        - Inserts or overwrites the backend class in self._backends.
        - Logs an info-level message with the backend display name and model type.
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
        1. Explicit ``backend_type`` in ``config.extra``.
        2. Auto-detect from model architecture.
        3. Custom plugin lookup.
        4. Fallback to autoregressive.

        Parameters
        ----------
        config : BackendConfig
            The benchmark configuration, including model_path and optional
            ``backend_type`` override in config.extra.

        Returns
        -------
        InferenceBackend
            A concrete backend instance (not yet loaded — caller must call .load()).

        Raises
        ------
        RuntimeError
            If no backend is available for the detected model type and the
            autoregressive fallback is also unavailable.
        """
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

        Heuristics (checked in order):
        1. Name-based: check for NLLB/MADLAD-400 keywords in the model path.
        2. Name-based: check for diffusion keywords (from DIFFUSION_KEYWORDS).
        3. Local path: read config.json, check model_type field, architectures
           list, and diffusion-specific config keys.
        4. HuggingFace Hub: fetch config via AutoConfig.from_pretrained() and
           apply the same checks as the local path.
        5. Default fallback: ModelType.AUTOREGRESSIVE.

        Results are cached in _CONFIG_CACHE keyed by
        ``f"__model_type__{model_path}"`` to avoid repeated network I/O and
        local JSON parsing on every call.

        Parameters
        ----------
        model_path : str
            A local filesystem path to a model directory, or a HuggingFace Hub
            model ID (e.g., "facebook/nllb-200-distilled-600M").

        Returns
        -------
        ModelType
            The detected model architecture category.

        Side Effects
        ------------
        - Populates _CONFIG_CACHE with the detection result and any intermediate
          config fetches (local JSON parsing, HF config).
        - Logs at debug level on detection failures; logs at error level if
          network is unavailable during HF config fetch.

        Exceptions Caught
        -----------------
        - OSError: network unavailable during HF config fetch (logged and skipped).
        - Any Exception during config-based detection (logged at debug level).
        """
        cache_key = f"__model_type__{model_path}"
        if cache_key in _CONFIG_CACHE:
            return _CONFIG_CACHE[cache_key]

        name_lower = model_path.lower()

        # ── Name-based heuristics: NLLB / MADLAD-400 ──
        if "nllb" in name_lower or "madlad" in name_lower:
            logger.info(
                "Encoder-decoder model detected via name: '%s'", model_path,
            )
            result = ModelType.ENCODER_DECODER
            _CONFIG_CACHE[cache_key] = result
            return result

        # ── Name-based heuristics: diffusion ──
        for kw in DIFFUSION_KEYWORDS:
            if kw in name_lower:
                logger.info("Diffusion model detected via name keyword: '%s'", kw)
                result = ModelType.DIFFUSION
                _CONFIG_CACHE[cache_key] = result
                return result

        # ── Config-based detection (local paths) ──
        local_path = Path(model_path)
        if local_path.exists():
            config_file = local_path / "config.json"
            if config_file.exists():
                try:
                    import json

                    # Cache local JSON config parsing to avoid re-reading on
                    # every call (keyed by model_path — different namespace
                    # from the model-type cache key).
                    local_cfg_key = f"__local_cfg__{model_path}"
                    if local_cfg_key in _CONFIG_CACHE:
                        cfg = _CONFIG_CACHE[local_cfg_key]
                    else:
                        with open(config_file) as f:
                            cfg = json.load(f)
                        _CONFIG_CACHE[local_cfg_key] = cfg

                    # Check model_type field.
                    if cfg.get("model_type") in ("diffusion", "diffusion_gemma"):
                        result = ModelType.DIFFUSION
                        _CONFIG_CACHE[cache_key] = result
                        return result

                    # Check architectures for encoder-decoder (NLLB, M2M100, BART, T5).
                    archs = cfg.get("architectures", [])
                    for arch in archs:
                        if arch in _ENCODER_DECODER_ARCHITECTURES:
                            logger.info(
                                "Encoder-decoder architecture detected: '%s'", arch,
                            )
                            result = ModelType.ENCODER_DECODER
                            _CONFIG_CACHE[cache_key] = result
                            return result

                    # Check for diffusion-specific config keys.
                    for key in _DIFFUSION_CONFIG_KEYS:
                        if key in cfg:
                            logger.info("Diffusion config key detected: '%s'", key)
                            result = ModelType.DIFFUSION
                            _CONFIG_CACHE[cache_key] = result
                            return result

                    # Check architectures.
                    archs = cfg.get("architectures", [])
                    for arch in archs:
                        arch_lower = arch.lower()
                        if any(kw in arch_lower for kw in DIFFUSION_KEYWORDS):
                            logger.info("Diffusion architecture detected: '%s'", arch)
                            result = ModelType.DIFFUSION
                            _CONFIG_CACHE[cache_key] = result
                            return result

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
                        result = ModelType.DIFFUSION
                        _CONFIG_CACHE[cache_key] = result
                        return result

                    # Check architectures for diffusion keywords.
                    archs = cfg.get("architectures", [])
                    for arch in archs:
                        arch_lower = arch.lower()
                        if any(kw in arch_lower for kw in DIFFUSION_KEYWORDS):
                            logger.info("HF config diffusion arch: '%s'", arch)
                            result = ModelType.DIFFUSION
                            _CONFIG_CACHE[cache_key] = result
                            return result

                    for key in _DIFFUSION_CONFIG_KEYS:
                        if key in cfg:
                            logger.info("HF config diffusion key: '%s'", key)
                            result = ModelType.DIFFUSION
                            _CONFIG_CACHE[cache_key] = result
                            return result
            except OSError:
                logger.error(
                    "network unavailable — cannot auto-detect model type. "
                    "Set backend_type in config."
                )
            except Exception:
                pass

        result = ModelType.AUTOREGRESSIVE
        _CONFIG_CACHE[cache_key] = result
        return result

    def _get_hf_config(self, model_id: str) -> Optional[dict]:
        """Fetch model config from HuggingFace Hub (cached).

        Retries up to 3 times with exponential backoff (1s, 2s, 4s) to handle
        transient network failures. Remote code execution is disabled
        (trust_remote_code=False).

        Parameters
        ----------
        model_id : str
            A HuggingFace Hub model ID (e.g., "facebook/nllb-200-distilled-600M").

        Returns
        -------
        Optional[dict]
            The model configuration as a dictionary if successfully fetched, or
            None if all attempts fail.

        Side Effects
        ------------
        - Populates _CONFIG_CACHE with the fetched config dict on success.
        - Logs debug messages on retries and failures.
        - Makes a network call to HuggingFace Hub (up to 3 attempts with
          exponential backoff).
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
        """Return metadata about all registered backends, including custom plugins.

        Enumerates all built-in backends (keyed by ModelType) plus any custom
        plugins discovered via PluginRegistry. Capabilities are reported as a
        list of single-bit flag names (compound convenience flags like
        FULL_TRANSLATION are excluded to avoid duplicates).

        Parameters
        ----------
        None

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing:
            - model_type (str): The model architecture category.
            - display_name (str): Human-readable backend name.
            - capabilities (list[str]): Names of supported ModelCapability flags.
            - class (str): The backend class name or "CustomPlugin".

        Side Effects
        ------------
        - May import PluginRegistry (benign ImportError if unavailable).
        - Does not mutate any instance or module state.
        """
        result = []
        # Only iterate over single-bit (power-of-two) capability flags,
        # skipping compound convenience flags like FULL_TRANSLATION and
        # FULL_DIFFUSION.  Iterating over the full IntFlag enum would
        # yield compound values that also match the bitwise AND check,
        # producing duplicate capability entries.
        _single_bit_caps = [
            cap for cap in ModelCapability
            if cap.value != 0 and (cap.value & (cap.value - 1)) == 0
        ]
        for model_type, cls in self._backends.items():
            result.append({
                "model_type": model_type.value,
                "display_name": cls.display_name,
                "capabilities": [
                    cap.name for cap in _single_bit_caps
                    if cap & cls.capabilities
                ],
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
