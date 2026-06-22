"""Stats API endpoint: system-wide counts."""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Request

from src.config import settings
from src.storage.note_store import NoteStore

router = APIRouter(prefix="/api/stats", tags=["stats"])

_VERSION = "0.1.0"


# ------------------------------------------------------------------ helpers (patchable in tests)


async def _get_note_count(request: Request) -> int:
    """Count notes via the O(1) metadata index (no corpus scan)."""
    store = getattr(request.app.state, "note_store", None)
    if store is None:
        store = NoteStore(settings.notes_dir)
        request.app.state.note_store = store
    return store.count()


async def _get_entity_count(request: Request) -> int:
    """Count entities in the KnowledgeGraph."""
    graph = request.app.state.knowledge_graph
    return graph.count_entities()


async def _get_relation_count(request: Request) -> int:
    """Count relations in the KnowledgeGraph."""
    graph = request.app.state.knowledge_graph
    return graph.count_relations()


async def _get_memory_count(request: Request) -> int:
    """Count memories from the SQLite memories table."""
    try:
        conn = sqlite3.connect(settings.sqlite_db_path)
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
    except Exception:
        total = 0
    return total


async def _get_card_count(request: Request) -> int:
    """Count cards in the SQLite cards table."""
    try:
        conn = sqlite3.connect(settings.sqlite_db_path)
        total = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        conn.close()
    except Exception:
        total = 0
    return total


# ------------------------------------------------------------------ endpoint


@router.get("/")
async def get_stats(request: Request) -> dict:
    """Return system-wide counts and version."""
    return {
        "notes": await _get_note_count(request),
        "entities": await _get_entity_count(request),
        "relations": await _get_relation_count(request),
        "memories": await _get_memory_count(request),
        "cards": await _get_card_count(request),
        "version": _VERSION,
    }
