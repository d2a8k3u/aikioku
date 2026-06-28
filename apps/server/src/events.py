"""Event bus for inter-module communication."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class Event:
    """Represents a system event."""

    def __init__(self, type: str, data: dict[str, Any]) -> None:
        self.id = str(uuid4())
        self.type = type
        self.data = data
        self.created = datetime.utcnow()


class EventBus:
    """SQLite-backed async event bus for inter-module communication."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._subscribers: dict[str, list[Callable[[Event], Any]]] = {}
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the events table."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created TEXT NOT NULL,
                    processed INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed)")

    def subscribe(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug("event_bus.subscribe", event_type=event_type)

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers and persist to DB."""
        # Persist
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO events (id, type, data, created) VALUES (?, ?, ?, ?)",
                (event.id, event.type, str(event.data), event.created.isoformat()),
            )

        # Notify subscribers
        handlers = self._subscribers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error("event_bus.handler_error", event_type=event.type, error=str(e))

        logger.debug("event_bus.publish", event_type=event.type, id=event.id)

    def get_unprocessed(self, limit: int = 100) -> list[Event]:
        """Get unprocessed events."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, type, data, created FROM events WHERE processed = 0 ORDER BY created LIMIT ?",
                (limit,),
            ).fetchall()
        events = []
        for row in rows:
            events.append(Event(type=row[1], data={"raw": row[2]}))
            events[-1].id = row[0]
        return events

    async def publish_async(self, event: Event) -> None:
        """Persist event and dispatch async/sync handlers."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO events (id, type, data, created) VALUES (?, ?, ?, ?)",
                (event.id, event.type, str(event.data), event.created.isoformat()),
            )
        handlers = self._subscribers.get(event.type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error("event_bus.handler_error", event_type=event.type, error=str(e))

    async def process_unprocessed(self, limit: int = 100) -> int:
        """Poll and dispatch unprocessed events asynchronously."""
        events = self.get_unprocessed(limit)
        for event in events:
            await self.publish_async(event)
            self._mark_processed(event.id)
        return len(events)

    def _mark_processed(self, event_id: str) -> None:
        """Mark an event as processed."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("UPDATE events SET processed = 1 WHERE id = ?", (event_id,))
