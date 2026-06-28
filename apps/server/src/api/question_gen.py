"""Question generation API endpoint for notes."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request

from src.reasoning.question_gen import QuestionGenerator

router = APIRouter(prefix="/api/notes", tags=["notes"])


def _get_generator() -> QuestionGenerator:
    """Get a QuestionGenerator instance."""
    return QuestionGenerator()


@router.get("/{note_id}/questions")
async def generate_questions(
    request: Request,
    note_id: UUID,
    count: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Generate review questions from a note."""
    from src.storage.note_store import NoteStore
    from src.config import settings

    store = getattr(request.app.state, "note_store", None)
    if store is None:
        store = NoteStore(settings.notes_dir)
        request.app.state.note_store = store

    note = store.get(str(note_id))
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")

    generator = _get_generator()
    try:
        questions = generator.generate_from_note(note, count=count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "note_id": str(note_id),
        "questions": [
            {"type": q.type, "question": q.question, "answer": q.answer} for q in questions
        ],
    }
