"""Plugin architecture for custom backends (M31).

Provides an abstract base class for backend plugins and a registry for
discovering / loading them.
"""

from __future__ import annotations

import abc
import importlib
import importlib.metadata
from argparse import Namespace
from typing import Any

from xpyd_bench.bench.models import RequestResult

__all__ = [
    "BackendPlugin",
    "PluginRegistry",
    "registry",
]

ENTRY_POINT_GROUP = "xpyd.backends"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BackendPlugin(abc.ABC):
    """Interface that every backend plugin must implement."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique backend name (e.g. ``"openai"``, ``"vllm-native"``)."""

    @abc.abstractmethod
    def build_payload(
        self,
        args: Namespace,
        prompt: str,
        *,
        is_chat: bool = False,
        is_embeddings: bool = False,
    ) -> dict[str, Any]:
        """Build the JSON body for a single request."""

    @abc.abstractmethod
    async def send_request(
        self,
        client: Any,
        url: str,
        payload: dict[str, Any],
        *,
        is_streaming: bool = False,
        request_timeout: float = 300.0,
        retries: int = 0,
        retry_delay: float = 1.0,
    ) -> RequestResult:
        """Send one request and return collected metrics."""

    def build_url(self, base_url: str, args: Namespace) -> str:
        """Build the full request URL.  Default: ``base_url + args.endpoint``."""
        return f"{base_url.rstrip('/')}{args.endpoint}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Central registry for backend plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, BackendPlugin] = {}
        self._ep_loaded = False

    # -- public API ---------------------------------------------------------

    def register(self, plugin: BackendPlugin) -> None:
        """Register a plugin instance.  Overwrites any existing with same name."""
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> BackendPlugin:
        """Return the plugin for *name*, raising ``KeyError`` if not found."""
        self._load_entry_points()
        if name not in self._plugins:
            raise KeyError(
                f"Unknown backend '{name}'. "
                f"Available: {', '.join(self.list_backends())}"
            )
        return self._plugins[name]

    def list_backends(self) -> list[str]:
        """Return sorted list of registered backend names."""
        self._load_entry_points()
        return sorted(self._plugins)

    # -- entry-point discovery ----------------------------------------------

    def _load_entry_points(self) -> None:
        """Discover plugins registered via ``pyproject.toml`` entry points."""
        if self._ep_loaded:
            return
        self._ep_loaded = True
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns SelectableGroups; earlier returns dict
            if hasattr(eps, "select"):
                group = eps.select(group=ENTRY_POINT_GROUP)
            else:
                group = eps.get(ENTRY_POINT_GROUP, [])
            for ep in group:
                try:
                    plugin_cls_or_inst = ep.load()
                    if isinstance(plugin_cls_or_inst, type):
                        inst = plugin_cls_or_inst()
                    else:
                        inst = plugin_cls_or_inst
                    if isinstance(inst, BackendPlugin) and inst.name not in self._plugins:
                        self._plugins[inst.name] = inst
                except Exception:  # noqa: BLE001
                    pass  # silently skip broken entry points
        except Exception:  # noqa: BLE001
            pass

    def load_module_plugin(self, module_path: str) -> BackendPlugin:
        """Load a plugin from a dotted module path.

        The module must define a ``plugin`` attribute (instance) **or** a
        ``Plugin`` class that is a ``BackendPlugin`` subclass.
        """
        mod = importlib.import_module(module_path)
        if hasattr(mod, "plugin") and isinstance(mod.plugin, BackendPlugin):
            plugin = mod.plugin
        elif hasattr(mod, "Plugin") and issubclass(mod.Plugin, BackendPlugin):
            plugin = mod.Plugin()
        else:
            raise ImportError(
                f"Module '{module_path}' does not expose a 'plugin' instance "
                "or a 'Plugin' class that subclasses BackendPlugin."
            )
        self.register(plugin)
        return plugin


# Singleton registry
registry = PluginRegistry()


def _register_builtins() -> None:
    """Register the built-in OpenAI backend plugin."""
    from xpyd_bench.plugins.openai_backend import OpenAIBackendPlugin

    registry.register(OpenAIBackendPlugin())


_register_builtins()
