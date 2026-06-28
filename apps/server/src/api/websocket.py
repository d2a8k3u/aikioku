"""WebSocket endpoint for real-time EventBus events."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.events import EventBus, Event

router = APIRouter(tags=["websocket"])


class _EventBusBroadcaster:
    """Manages WebSocket connections and broadcasts EventBus events."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._handler_registered = False

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)
        if not self._handler_registered:
            self._event_bus.subscribe("*", self._on_event)
            self._handler_registered = True

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)

    async def _on_event(self, event: Event) -> None:
        message = json.dumps(
            {
                "id": event.id,
                "type": event.type,
                "data": event.data,
                "created": event.created.isoformat(),
            }
        )
        async with self._lock:
            to_remove: list[WebSocket] = []
            for ws in self._connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    to_remove.append(ws)
            for ws in to_remove:
                self._connections.remove(ws)

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        event = Event(event_type, data)
        await self._on_event(event)


# Global broadcaster instance (set on app startup)
_broadcaster: _EventBusBroadcaster | None = None


def set_broadcaster(event_bus: EventBus) -> None:
    global _broadcaster
    _broadcaster = _EventBusBroadcaster(event_bus)


def get_broadcaster() -> "_EventBusBroadcaster | None":
    """Return the global broadcaster (used by background jobs to push events).

    NOTE: ``EventBus.publish`` dispatches by exact ``event.type`` while the
    broadcaster subscribes to ``"*"``, so EventBus → WebSocket never fires.
    Callers that want events on the socket must call ``broadcaster.broadcast``.
    """
    return _broadcaster


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    """WebSocket endpoint that streams all EventBus events in real-time."""
    if _broadcaster is None:
        await websocket.close(code=1011, reason="EventBus not initialized")
        return
    await _broadcaster.connect(websocket)
    try:
        while True:
            # Keep connection alive; clients can send ping/keepalive messages
            data = await websocket.receive_text()
            # Echo back as simple heartbeat acknowledgment
            await websocket.send_text(json.dumps({"type": "heartbeat_ack", "received": data}))
    except WebSocketDisconnect:
        await _broadcaster.disconnect(websocket)
    except Exception:
        await _broadcaster.disconnect(websocket)
