"""Admin / maintenance endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

if TYPE_CHECKING:
    from src.models.note import Note

router = APIRouter(prefix="/api/admin", tags=["admin"])
logger = logging.getLogger(__name__)

# Bound concurrent LLM extraction calls so the remote model stays responsive.
_REEXTRACT_CONCURRENCY = 4

# Module-level progress so the status endpoint can report on an in-flight run.
_reextract_state: dict[str, int | bool] = {
    "running": False,
    "processed": 0,
    "total": 0,
    "entities": 0,
    "errors": 0,
}


@router.post("/reextract")
async def reextract_entities(request: Request) -> dict[str, Any]:
    """Re-run entity extraction over all notes into the shared knowledge graph.

    Runs in the background with bounded concurrency and returns immediately.
    Useful after a bulk import (whose on-create extraction was rate-limited) or
    a change to the extraction pipeline. Poll GET /api/admin/reextract/status.
    """
    from src.knowledge.pipeline import extract_entities_from_note

    if _reextract_state["running"]:
        return {"status": "already_running", **_reextract_state}

    store = request.app.state.note_store
    llm = request.app.state.llm_provider
    graph = request.app.state.knowledge_graph
    notes = store.list_all()

    _reextract_state.update(running=True, processed=0, total=len(notes), entities=0, errors=0)

    async def _run() -> None:
        sem = asyncio.Semaphore(_REEXTRACT_CONCURRENCY)

        async def _one(note: Note) -> None:
            async with sem:
                try:
                    # The LLM call is the only await; the Kuzu writes that follow
                    # are synchronous, so they never overlap on the single shared
                    # connection (the event loop serialises them).
                    ents = await extract_entities_from_note(note, llm, graph)
                    _reextract_state["entities"] += len(ents)
                except Exception as exc:  # noqa: BLE001
                    _reextract_state["errors"] += 1
                    logger.warning(
                        "reextract failed for note %s: %s", getattr(note, "id", "?"), exc
                    )
                finally:
                    _reextract_state["processed"] += 1

        await asyncio.gather(*[_one(n) for n in notes])
        _reextract_state["running"] = False
        logger.info("reextract complete: %s", _reextract_state)

    asyncio.ensure_future(_run())
    return {"status": "started", "total": len(notes)}


@router.get("/reextract/status")
async def reextract_status() -> dict[str, int | bool]:
    """Return progress of the most recent re-extraction run."""
    return dict(_reextract_state)
