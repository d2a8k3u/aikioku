"""Plugin manager."""

from __future__ import annotations


class PluginManager:
    """Manages plugin registration, activation, and deactivation."""

    def __init__(self) -> None:
        self._plugins: dict[str, dict] = {}

    def register_plugin(self, name: str, manifest: dict) -> None:
        """Register a plugin with its manifest metadata."""
        self._plugins[name] = {"name": name, "manifest": manifest, "active": False}

    def list_plugins(self) -> list[str]:
        """Return a list of registered plugin names."""
        return list(self._plugins.keys())

    def activate_plugin(self, name: str) -> None:
        """Activate a registered plugin."""
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' is not registered.")
        self._plugins[name]["active"] = True

    def deactivate_plugin(self, name: str) -> None:
        """Deactivate a registered plugin."""
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' is not registered.")
        self._plugins[name]["active"] = False
