"""Auto-tagging API endpoints."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from src.augmentation.auto_tag import AutoTagger
from src.auth import CurrentUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tags", tags=["tags"])


def _get_tagger(request: Request) -> AutoTagger:
    tagger = getattr(request.app.state, "auto_tagger", None)
    if tagger is None:
        llm = getattr(request.app.state, "llm_provider", None)
        tagger = AutoTagger(llm_provider=llm)
        request.app.state.auto_tagger = tagger
    return tagger


@router.post("/auto/{note_id}")
async def auto_tag_note(
    request: Request,
    note_id: UUID,
    user: CurrentUser,
) -> dict[str, str | list[str]]:
    """Generate tags for a note using the auto-tagging engine and persist them."""
    from src.config import settings
    from src.storage.note_store import NoteStore

    store = getattr(request.app.state, "note_store", None)
    if store is None:
        store = NoteStore(settings.notes_dir)
        request.app.state.note_store = store

    note = store.get(str(note_id))
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")

    tagger = _get_tagger(request)
    try:
        tags = await tagger.generate_tags(note)
    except Exception as exc:
        logger.exception("auto-tag generation failed for note %s", note_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    note.frontmatter["tags"] = tags
    store.update(note)
    # Invalidate semantic cache — updated note may change correct answers
    import asyncio

    from src.cache.semantic_cache import cache_invalidate

    asyncio.ensure_future(cache_invalidate())

    return {"note_id": str(note_id), "tags": tags}
