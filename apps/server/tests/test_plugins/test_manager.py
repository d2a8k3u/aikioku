"""Tests for plugin manager."""

from __future__ import annotations

import pytest


class TestPluginManager:
    def test_import(self):
        pass

    def test_register_plugin(self):
        from src.plugins.manager import PluginManager

        mgr = PluginManager()
        manifest = {"version": "1.0.0", "entrypoint": "main.py"}
        mgr.register_plugin("test_plugin", manifest)
        assert "test_plugin" in mgr.list_plugins()

    def test_list_plugins_empty(self):
        from src.plugins.manager import PluginManager

        mgr = PluginManager()
        assert mgr.list_plugins() == []

    def test_list_plugins_multiple(self):
        from src.plugins.manager import PluginManager

        mgr = PluginManager()
        mgr.register_plugin("p1", {"version": "1.0.0"})
        mgr.register_plugin("p2", {"version": "2.0.0"})
        plugins = mgr.list_plugins()
        assert len(plugins) == 2
        assert "p1" in plugins
        assert "p2" in plugins

    def test_activate_plugin(self):
        from src.plugins.manager import PluginManager

        mgr = PluginManager()
        mgr.register_plugin("p1", {"version": "1.0.0"})
        mgr.activate_plugin("p1")
        mgr.list_plugins()
        assert mgr._plugins["p1"]["active"] is True

    def test_deactivate_plugin(self):
        from src.plugins.manager import PluginManager

        mgr = PluginManager()
        mgr.register_plugin("p1", {"version": "1.0.0"})
        mgr.activate_plugin("p1")
        mgr.deactivate_plugin("p1")
        assert mgr._plugins["p1"]["active"] is False

    def test_activate_unknown_raises(self):
        from src.plugins.manager import PluginManager

        mgr = PluginManager()
        with pytest.raises(ValueError):
            mgr.activate_plugin("nonexistent")

    def test_deactivate_unknown_raises(self):
        from src.plugins.manager import PluginManager

        mgr = PluginManager()
        with pytest.raises(ValueError):
            mgr.deactivate_plugin("nonexistent")


class TestPluginAPI:
    def test_import(self):
        pass

    def test_api_surface(self):
        from src.plugins.api import PluginAPI

        api = PluginAPI()
        assert hasattr(api, "register")
        assert hasattr(api, "call")
