"""Summarization API endpoint for notes."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from src.augmentation.summarization import ProgressiveSummarizer

router = APIRouter(prefix="/api/notes", tags=["notes"])


def _get_summarizer(request: Request) -> ProgressiveSummarizer:
    """Get or create a ProgressiveSummarizer from app state."""
    summarizer = getattr(request.app.state, "summarizer", None)
    if summarizer is None:
        llm = request.app.state.llm_provider
        summarizer = ProgressiveSummarizer(llm)
        request.app.state.summarizer = summarizer
    return summarizer


@router.post("/{note_id}/summarize")
async def summarize_note(request: Request, note_id: UUID) -> dict:
    """Generate a multi-level summary for a note."""
    from src.storage.note_store import NoteStore
    from src.config import settings

    store = getattr(request.app.state, "note_store", None)
    if store is None:
        store = NoteStore(settings.notes_dir)
        request.app.state.note_store = store

    note = store.get(str(note_id))
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")

    summarizer = _get_summarizer(request)
    try:
        summary = await summarizer.summarize(note)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"note_id": str(note_id), "summary": summary}
