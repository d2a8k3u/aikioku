"""Plugin API surface with EventBus integration."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from src.events import EventBus

logger = logging.getLogger(__name__)


class PluginAPI:
    """API surface for plugin interactions with EventBus hook wiring."""

    _VALID_HOOKS: set[str] = {"onNoteSave", "onQuery", "onReview"}

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._hooks: dict[str, list[Callable]] = {}
        self._event_bus = event_bus

    def register(self, hook_name: str, handler: Callable) -> None:
        """Register a handler for a named hook.

        Valid hooks: onNoteSave, onQuery, onReview.
        If an EventBus is wired, the handler is also subscribed to the
        corresponding event type so it runs automatically.
        """
        if hook_name not in self._VALID_HOOKS:
            raise ValueError(f"invalid hook: {hook_name}")
        self._hooks.setdefault(hook_name, []).append(handler)
        if self._event_bus is not None:
            event_type = self._hook_to_event_type(hook_name)
            if event_type:
                self._event_bus.subscribe(event_type, handler)
                logger.debug("plugin_hook_subscribed: %s -> %s", hook_name, event_type)

    def call(self, hook_name: str, *args, **kwargs) -> list[Any]:
        """Invoke all handlers registered for a hook and return results."""
        results = []
        for handler in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    # For sync call, we can't await — log warning
                    logger.warning(
                        "async_handler_called_in_sync_context: %s", hook_name
                    )
                    results.append(None)
                else:
                    results.append(handler(*args, **kwargs))
            except Exception as exc:
                logger.error("plugin_hook_error: %s - %s", hook_name, str(exc))
                results.append(None)
        return results

    async def call_async(self, hook_name: str, *args, **kwargs) -> list[Any]:
        """Invoke all handlers asynchronously and return results."""
        results = []
        for handler in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    results.append(await handler(*args, **kwargs))
                else:
                    results.append(handler(*args, **kwargs))
            except Exception as exc:
                logger.error("plugin_hook_error_async: %s - %s", hook_name, str(exc))
                results.append(None)
        return results

    def _hook_to_event_type(self, hook_name: str) -> str | None:
        mapping = {
            "onNoteSave": "note.created",
            "onQuery": "query.submitted",
            "onReview": "card.reviewed",
        }
        return mapping.get(hook_name)

    def get_registered_hooks(self) -> dict[str, list[str]]:
        """Return a dict of hook names to list of handler names."""
        return {
            hook: [getattr(h, "__name__", repr(h)) for h in handlers]
            for hook, handlers in self._hooks.items()
        }
