"""Search API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query

from src.models.note import Note
from src.storage.note_store import NoteStore

router = APIRouter(prefix="/api/search", tags=["search"])


def get_note_store() -> NoteStore:
    global _store
    if _store is None:
        from src.config import settings
        _store = NoteStore(settings.notes_dir)
    return _store


_store: NoteStore | None = None


@router.get("/")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
) -> list[Note]:
    """Full-text search across all notes."""
    store = get_note_store()
    return store.search(q)[:limit]
