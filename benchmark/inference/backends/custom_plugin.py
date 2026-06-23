"""Custom model plugin system (v3.0).

Enables users to register their own inference backends for custom model
architectures without modifying the framework source code.

Plugins can be:
1. **Python modules** — placed in ``~/.tr_benchmark/plugins/``.
2. **Entry points** — registered via ``pyproject.toml`` under the
   ``tr_benchmark.plugins`` group.
3. **Inline registrations** — called at runtime via ``register_plugin()``.

Plugin contract
---------------
Each plugin is a Python module or class that provides:
- ``name`` : str — unique identifier.
- ``model_type`` : ModelType — AUTOREGRESSIVE, DIFFUSION, or CUSTOM.
- ``create_backend(config: BackendConfig) → InferenceBackend`` — factory.
- (optional) ``detect(model_path: str) → bool`` — custom auto-detection.
- (optional) ``get_config_schema() → dict`` — JSON Schema for validation.
- (optional) ``custom_kernels() → dict[str, Callable]`` — Triton/Metal kernels.

Example plugin (for a custom diffusion model)
----------------------------------------------
.. code-block:: python

    # my_custom_model_plugin.py
    from benchmark.inference.backends import (
        CustomModelPlugin, register_plugin,
        InferenceBackend, BackendConfig, ModelType,
    )

    class MyDiffusionBackend(InferenceBackend):
        model_type = ModelType.DIFFUSION
        display_name = "My Custom Diffusion Model"

        def load(self):
            # Custom loading logic...
            pass

        def translate_batch(self, batch):
            # Custom translation logic...
            pass

        def warmup(self, batches=20):
            pass

        def is_loaded(self):
            return self._loaded

    class MyPlugin(CustomModelPlugin):
        name = "my_diffusion_model"
        version = "1.0.0"
        description = "Custom diffusion model for EN→TR translation"

        def create_backend(self, config):
            return MyDiffusionBackend(config)

        def detect(self, model_path):
            return "my_model_signature" in model_path.lower()

    register_plugin(MyPlugin())

Discovery mechanism
-------------------
Plugins are discovered at import time in this order:
1. ``tr_benchmark.plugins`` entry points (setuptools) — gated by
   ``TR_ALLOW_UNTRUSTED_PLUGINS=1``.
2. ``~/.tr_benchmark/plugins/*.py`` files — gated by
   ``TR_ALLOW_UNTRUSTED_PLUGINS=1``.
3. ``TR_BENCHMARK_PLUGIN_PATH`` environment variable — gated by
   ``TR_ALLOW_UNTRUSTED_PLUGINS=1``.
4. Runtime ``register_plugin()`` calls — NOT gated (explicit opt-in).

Environment variable gate
-------------------------
**All** automatic plugin discovery paths are **disabled by default** —
including entry points (setuptools), directory scanning, and the
project-local ``plugins/`` directory.  To enable any of them you must
set the environment variable::

    export TR_ALLOW_UNTRUSTED_PLUGINS=1

Without this gate the framework will skip all automatic discovery and
log an info message.  Only runtime ``register_plugin()`` calls bypass
this gate — the caller has already explicitly chosen to load the plugin
by invoking the function directly.

Security considerations
-----------------------
Plugins execute **arbitrary Python code** with the full privileges of the
calling process.  Loading a plugin from an untrusted source is equivalent
to running an untrusted script.

- **Set ``TR_ALLOW_UNTRUSTED_PLUGINS=1`` only in environments you fully
  control** (CI containers, dedicated development machines, air-gapped
  benchmark rigs).  Never set it on multi-tenant or shared hosts.
- **Audit third-party plugins before use.**  Review the plugin source,
  check ``pip freeze`` dependencies, and prefer plugins that ship with a
  cryptographic signature or provenance attestation.
- **Prefer entry points or explicit ``register_plugin()`` calls** rather
  than directory scanning when distributing plugins to others — this
  gives end users more visibility into what they are loading.
- **The framework does not sandbox or isolate plugins.**  A malicious
  plugin can read/write files, access the network, and exfiltrate
  environment variables (including API keys).  Treat every plugin as
  fully trusted.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin protocol
# ---------------------------------------------------------------------------


class CustomModelPlugin(ABC):
    """Protocol for custom model plugins.

    Subclass this, implement the required methods, and call
    ``register_plugin()`` or place the module in the plugin directory.

    Attributes
    ----------
    name : str
        Unique plugin identifier.  Used in config overrides and logging.
    version : str
        Semantic version string.
    description : str
        Human-readable description for discovery.
    author : str
        Plugin author.
    requires : list[str]
        List of pip package requirements.
    """

    name: str = "unnamed_plugin"
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    requires: list[str] = []

    @abstractmethod
    def create_backend(self, config: Any) -> Any:
        """Create an ``InferenceBackend`` instance for this model.

        Parameters
        ----------
        config : BackendConfig
            Full backend configuration.

        Returns
        -------
        InferenceBackend
            A concrete, not-yet-loaded backend instance.
        """
        ...

    def detect(self, model_path: str) -> bool:
        """Return True if this plugin can handle the given model.

        Override for custom detection heuristics.  Default returns False.

        Parameters
        ----------
        model_path : str
            HuggingFace model ID or local path.

        Returns
        -------
        bool
        """
        return False

    def get_config_schema(self) -> Optional[dict]:
        """Return a JSON Schema for plugin-specific configuration.

        Override to enable config validation for plugin-specific extra fields.

        Returns
        -------
        dict or None
            JSON Schema object or None if no validation needed.
        """
        return None

    def get_custom_kernels(self) -> dict[str, Callable]:
        """Return custom Triton/Metal kernels for this model.

        Override to inject custom fused operations.  Each kernel must be
        a callable compatible with the model's forward pass.

        Returns
        -------
        dict[str, Callable]
            Kernel name → callable (Triton kernel or Metal shader function).
        """
        return {}

    def get_optimized_attention_fn(self) -> Optional[Callable]:
        """Return a custom attention function for this model.

        Override to replace the default attention with a model-specific
        optimized implementation (e.g., custom FlashAttention variant).

        Returns
        -------
        Callable or None
        """
        return None

    def get_decoding_params(self) -> dict:
        """Return recommended decoding hyperparameters.

        Returns
        -------
        dict
            Keys like ``temperature``, ``top_p``, ``num_beams``,
            ``diffusion_steps``, ``guidance_scale``, etc.
        """
        return {}

    def __repr__(self) -> str:
        return f"<CustomModelPlugin '{self.name}' v{self.version}>"


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------


class _PluginRegistry:
    """Singleton registry for custom model plugins."""

    _plugins: dict[str, CustomModelPlugin] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, plugin: CustomModelPlugin) -> None:
        """Register a plugin instance.

        Parameters
        ----------
        plugin : CustomModelPlugin
        """
        if not isinstance(plugin, CustomModelPlugin):
            raise TypeError(
                f"Expected CustomModelPlugin, got {type(plugin).__name__}"
            )
        if plugin.name in cls._plugins:
            logger.warning(
                "Plugin '%s' already registered — overwriting", plugin.name,
            )
        cls._plugins[plugin.name] = plugin
        logger.info(
            "Registered plugin '%s' v%s (%s)", plugin.name, plugin.version, plugin.description,
        )

    @classmethod
    def lookup(cls, model_path: str) -> Optional[CustomModelPlugin]:
        """Find a plugin that can handle the given model.

        Parameters
        ----------
        model_path : str
            Model path or HuggingFace ID.

        Returns
        -------
        CustomModelPlugin or None
        """
        cls._ensure_discovered()

        for plugin in cls._plugins.values():
            try:
                if plugin.detect(model_path):
                    return plugin
            except Exception as e:
                logger.debug("Plugin '%s' detection error: %s", plugin.name, e)

        return None

    @classmethod
    def get(cls, name: str) -> Optional[CustomModelPlugin]:
        """Get a plugin by name.

        Parameters
        ----------
        name : str
            Unique plugin identifier (the ``name`` attribute of the plugin).

        Returns
        -------
        CustomModelPlugin or None
            The registered plugin instance, or None if no plugin with that
            name has been registered.
        """
        cls._ensure_discovered()
        return cls._plugins.get(name)

    @classmethod
    def list_plugins(cls) -> list[str]:
        """Return list of registered plugin names."""
        cls._ensure_discovered()
        return list(cls._plugins.keys())

    @classmethod
    def list_plugins_detailed(cls) -> list[dict]:
        """Return detailed metadata for all registered plugins."""
        cls._ensure_discovered()
        return [
            {
                "name": p.name,
                "version": p.version,
                "description": p.description,
                "author": p.author,
                "requires": p.requires,
            }
            for p in cls._plugins.values()
        ]

    @classmethod
    def _ensure_discovered(cls) -> None:
        """Lazy discovery — scan plugin directories on first access."""
        if cls._discovered:
            return
        cls._discovered = True

        # ── 1. Entry points ──
        cls._discover_entry_points()

        # ── 2. User plugin directory ──
        cls._discover_directory(Path.home() / ".tr_benchmark" / "plugins")

        # ── 3. Environment variable path ──
        env_path = os.environ.get("TR_BENCHMARK_PLUGIN_PATH")
        if env_path:
            for p in env_path.split(":"):
                cls._discover_directory(Path(p))

        # ── 4. Project-local plugin directory ──
        cls._discover_directory(Path.cwd() / "plugins")

    @classmethod
    def _discover_entry_points(cls) -> None:
        """Discover plugins registered via setuptools entry points.

        Gated behind TR_ALLOW_UNTRUSTED_PLUGINS=1 — entry points can
        execute arbitrary code when installed from PyPI packages, and
        should only be activated in environments the user fully controls.
        """
        if os.environ.get("TR_ALLOW_UNTRUSTED_PLUGINS") != "1":
            logger.info(
                "Skipping plugin entry point discovery "
                "(TR_ALLOW_UNTRUSTED_PLUGINS not set)"
            )
            return
        try:
            eps = importlib.metadata.entry_points(group="tr_benchmark.plugins")
        except Exception:
            return

        for ep in eps:
            try:
                plugin_factory = ep.load()
                if callable(plugin_factory):
                    plugin = plugin_factory()
                else:
                    plugin = plugin_factory
                if isinstance(plugin, CustomModelPlugin):
                    cls.register(plugin)
                else:
                    logger.warning(
                        "Entry point '%s' did not return a CustomModelPlugin", ep.name,
                    )
            except Exception as e:
                logger.warning("Failed to load plugin entry point '%s': %s", ep.name, e)

    @classmethod
    def _discover_directory(cls, directory: Path) -> None:
        """Scan a directory for Python plugin modules.

        WARNING: Plugins execute arbitrary code. Only enable with
        TR_ALLOW_UNTRUSTED_PLUGINS=1 in environments you fully control.
        """
        if not directory.exists() or not directory.is_dir():
            return

        if os.environ.get("TR_ALLOW_UNTRUSTED_PLUGINS") != "1":
            logger.info("Skipping plugin dir %s (TR_ALLOW_UNTRUSTED_PLUGINS not set)", directory)
            return

        for py_file in sorted(directory.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                mod_name = py_file.stem
                # SECURITY: importlib.import_module executes arbitrary code.
                # Guarded by the TR_ALLOW_UNTRUSTED_PLUGINS env-var check above.
                # Load the plugin module in isolation via spec to avoid
                # polluting sys.path for the entire process.
                spec = importlib.util.spec_from_file_location(
                    mod_name, str(py_file),
                )
                if spec is None or spec.loader is None:
                    logger.warning("Could not create spec for plugin %s", py_file)
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                # Find CustomModelPlugin subclasses in the module.
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, CustomModelPlugin)
                        and attr is not CustomModelPlugin
                    ):
                        try:
                            instance = attr()
                            cls.register(instance)
                        except Exception as e:
                            logger.warning(
                                "Failed to instantiate plugin %s.%s: %s",
                                mod_name, attr_name, e,
                            )
            except Exception as e:
                logger.debug("Skipping plugin file %s: %s", py_file, e)


# ── Public API ────────────────────────────────────────────────────────────


# Singleton instance.
PluginRegistry = _PluginRegistry


def register_plugin(plugin: CustomModelPlugin) -> None:
    """Register a custom model plugin at runtime.

    Usage
    -----
    >>> class MyPlugin(CustomModelPlugin):
    ...     name = "my_diffusion_model"
    ...     def create_backend(self, config):
    ...         return MyDiffusionBackend(config)
    ...
    >>> register_plugin(MyPlugin())
    """
    PluginRegistry.register(plugin)
