"""Tests for async EventBus methods."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.events import Event, EventBus


@pytest.fixture
def event_bus():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "events.db")
        bus = EventBus(db_path)
        yield bus


class TestEventBusAsync:
    """Test publish_async and process_unprocessed."""

    @pytest.mark.asyncio
    async def test_publish_async_calls_sync_handler(self, event_bus):
        called = []

        def handler(ev):
            called.append(ev.type)

        event_bus.subscribe("note.created", handler)
        event = Event("note.created", {"note_id": "n1"})
        await event_bus.publish_async(event)
        assert called == ["note.created"]

    @pytest.mark.asyncio
    async def test_publish_async_calls_async_handler(self, event_bus):
        called = []

        async def handler(ev):
            called.append(ev.type)

        event_bus.subscribe("note.created", handler)
        event = Event("note.created", {"note_id": "n1"})
        await event_bus.publish_async(event)
        assert called == ["note.created"]

    @pytest.mark.asyncio
    async def test_process_unprocessed_dispatches_events(self, event_bus):
        called = []

        def handler(ev):
            called.append(ev.type)

        event_bus.subscribe("note.created", handler)
        # publish_async persists but doesn't mark processed
        event = Event("note.created", {"note_id": "n1"})
        await event_bus.publish_async(event)
        count = await event_bus.process_unprocessed(limit=100)
        assert count == 1
        # handler called once by publish_async, once by process_unprocessed
        assert called == ["note.created", "note.created"]

    @pytest.mark.asyncio
    async def test_process_unprocessed_returns_zero_when_empty(self, event_bus):
        count = await event_bus.process_unprocessed(limit=100)
        assert count == 0
