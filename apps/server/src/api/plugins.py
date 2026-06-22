"""Plugin API endpoints with manifest validation and hook wiring."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator

from src.plugins.manager import PluginManager
from src.plugins.api import PluginAPI

router = APIRouter(prefix="/api/plugins", tags=["plugins"])


class PluginManifest(BaseModel):
    name: str
    version: str
    description: str = ""
    hooks: list[str] = []

    @field_validator("hooks")
    @classmethod
    def validate_hooks(cls, v: list[str]) -> list[str]:
        allowed = {"onNoteSave", "onQuery", "onReview"}
        for h in v:
            if h not in allowed:
                raise ValueError(f"invalid hook: {h}. Allowed: {allowed}")
        return v


def _get_manager(request: Request) -> PluginManager:
    manager = getattr(request.app.state, "plugin_manager", None)
    if manager is None:
        manager = PluginManager()
        request.app.state.plugin_manager = manager
    return manager


def _get_plugin_api(request: Request) -> PluginAPI:
    api = getattr(request.app.state, "plugin_api", None)
    if api is None:
        api = PluginAPI()
        request.app.state.plugin_api = api
    return api


@router.get("/")
async def list_plugins(request: Request) -> list[dict]:
    """List all registered plugins."""
    manager = _get_manager(request)
    return [
        {"name": name, **info}
        for name, info in manager._plugins.items()
    ]


@router.post("/")
async def register_plugin(request: Request, manifest: PluginManifest) -> dict:
    """Register a new plugin from a manifest."""
    manager = _get_manager(request)
    manager.register_plugin(manifest.name, manifest.model_dump())
    return {"status": "registered", "name": manifest.name}


@router.post("/{name}/activate")
async def activate_plugin(request: Request, name: str) -> dict:
    """Activate a registered plugin."""
    manager = _get_manager(request)
    try:
        manager.activate_plugin(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "activated", "name": name}


@router.post("/{name}/deactivate")
async def deactivate_plugin(request: Request, name: str) -> dict:
    """Deactivate a registered plugin."""
    manager = _get_manager(request)
    try:
        manager.deactivate_plugin(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "deactivated", "name": name}


@router.delete("/{name}")
async def unregister_plugin(request: Request, name: str) -> dict:
    """Remove a registered plugin."""
    manager = _get_manager(request)
    if name not in manager._plugins:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {name}")
    del manager._plugins[name]
    return {"status": "unregistered", "name": name}


@router.get("/{name}/hooks")
async def list_plugin_hooks(request: Request, name: str) -> list[str]:
    """List hooks registered for a plugin."""
    manager = _get_manager(request)
    info = manager._plugins.get(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {name}")
    manifest = info.get("manifest", {})
    return manifest.get("hooks", [])
