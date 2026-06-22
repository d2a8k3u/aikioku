"""Shared test fixtures for Aikioku backend tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest


@pytest.fixture
def fixed_uuid() -> str:
    """Return a deterministic UUID string for assertions."""
    return "12345678-1234-5678-1234-567812345678"


@pytest.fixture
def fixed_uuid2() -> str:
    """Return a second deterministic UUID string."""
    return "87654321-4321-8765-4321-876543210987"


@pytest.fixture
def fixed_datetime() -> datetime:
    """Return a fixed UTC datetime for deterministic assertions."""
    return datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def fixed_datetime_later() -> datetime:
    """Return a later fixed UTC datetime."""
    return datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def reset_app_state(tmp_path):
    """Reset app state and use temp DB paths to avoid container lock contention.

    Patches settings BEFORE importing src.main so module-level code
    (runtime_config.cors_origins()) sees the temp paths.
    """
    from src.config import settings

    tmp_db = str(tmp_path / "test.db")

    with patch.object(settings, "sqlite_db_path", tmp_db), \
         patch.object(settings, "notes_dir", str(tmp_path)):
        # Now safe to import main — module-level code uses patched settings
        from src.main import app

        for attr in ("knowledge_graph", "embedding_store", "hybrid_fusion", "note_store",
                     "serendipity_engine", "summarizer", "connection_discovery",
                     "cognitive_tracker", "git_sync", "llm_provider", "event_bus",
                     "spaced_repetition", "auth", "card_auto_generator"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)

        yield
