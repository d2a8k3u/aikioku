"""Budget gate for LLM-backed ingestion work + the drain that resumes it.

``gated`` runs a unit of work immediately when the daily budget allows, or
enqueues it (:mod:`pending_work`) and pauses when the budget is exhausted.
``drain`` re-runs queued work once the budget is available again. The same
``RUNNERS`` reconstruct work from a JSON payload in both paths, so live and
deferred processing are identical.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from fastapi import FastAPI

from src.llm.router import BudgetExceededError
from src.processing import pending_work
from src.processing.budget_status import broadcast_budget_status

logger = logging.getLogger(__name__)

WORK_NOTE_PROCESSING = "note_processing"
WORK_MEMORY_EXTRACTION = "memory_extraction"


async def _run_note_processing(app: FastAPI, payload: dict[str, Any]) -> None:
    """Entity extraction + embedding for a note (source of truth: its markdown)."""
    note_id = payload["note_id"]
    store = getattr(app.state, "note_store", None)
    if store is None:
        from src.api.notes import get_note_store

        store = get_note_store()
    note = store.get(note_id)
    if note is None:
        return
    from src.knowledge.pipeline import (
        extract_entities_from_note,
        store_note_embeddings,
    )

    await extract_entities_from_note(note, app.state.llm_provider, app.state.knowledge_graph)
    await store_note_embeddings(note, app.state.embedding_provider, app.state.embedding_store)
    fusion = getattr(app.state, "hybrid_fusion", None)
    sparse = getattr(fusion, "sparse", None)
    if sparse is not None and hasattr(sparse, "mark_dirty"):
        sparse.mark_dirty()


async def _run_memory_extraction(app: FastAPI, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """(Re)extract memory triples from a note or free text and persist them."""
    from src.api.memory import (
        _create_memory_from_text,
        _extract_memories,
        _get_stats,
    )
    from src.api.websocket import get_broadcaster

    llm = getattr(app.state, "llm_provider", None)
    provider = getattr(app.state, "embedding_provider", None)
    store = getattr(app.state, "memory_embedding_store", None)
    if payload.get("mode") == "note":
        memories = await _extract_memories(
            payload["note_id"], llm=llm, provider=provider, store=store
        )
        event = "memory.extracted"
    else:
        memories = await _create_memory_from_text(
            payload["text"],
            payload.get("source", "user"),
            llm=llm,
            provider=provider,
            store=store,
        )
        event = "memory.created"

    # Project the new memories into the knowledge graph (entities + relations).
    # Fire-and-forget so the response never blocks on per-memory LLM typing,
    # mirroring note entity extraction in src/api/notes.py.
    graph = getattr(app.state, "knowledge_graph", None)
    if graph is not None and llm is not None and memories:
        from src.api.memory import _memory_from_dict
        from src.memory.graph_sync import sync_memories_to_graph

        mem_objs = [_memory_from_dict(m) for m in memories]
        asyncio.ensure_future(sync_memories_to_graph(mem_objs, graph, llm))

    broadcaster = get_broadcaster()
    if broadcaster is not None:
        stats = await _get_stats()
        await broadcaster.broadcast(event, {"count": len(memories), "tier_counts": stats})
    return memories


RUNNERS: dict[str, Callable[[FastAPI, dict[str, Any]], Awaitable[Any]]] = {
    WORK_NOTE_PROCESSING: _run_note_processing,
    WORK_MEMORY_EXTRACTION: _run_memory_extraction,
}


async def gated(app: FastAPI, work_type: str, entity_id: str, payload: dict[str, Any]) -> Any:
    """Run ``work_type`` now if budget allows, else enqueue it and pause.

    Returns the runner's result on the live path, or ``None`` when the work was
    deferred. Sync callers treat ``None`` as "queued"; fire-and-forget callers
    ignore the return value.
    """
    tracker = getattr(app.state, "cost_tracker", None)
    if tracker is not None and tracker.is_exhausted():
        pending_work.enqueue(work_type, entity_id, payload)
        await broadcast_budget_status(app)
        return None
    try:
        result = await RUNNERS[work_type](app, payload)
        await broadcast_budget_status(app)
        return result
    except BudgetExceededError:
        pending_work.enqueue(work_type, entity_id, payload)
        await broadcast_budget_status(app)
        return None


async def drain(app: FastAPI) -> int:
    """Re-run queued work while the budget allows. Returns items processed."""
    tracker = getattr(app.state, "cost_tracker", None)
    if tracker is None:
        return 0
    processed = 0
    for item in pending_work.list_pending(limit=100):
        if tracker.is_exhausted():
            break
        runner = RUNNERS.get(item["work_type"])
        if runner is None:
            pending_work.delete(item["id"])
            continue
        try:
            await runner(app, item["payload"])
            pending_work.delete(item["id"])
            processed += 1
        except BudgetExceededError:
            break
        except Exception as exc:  # isolate one poison item from the rest
            logger.warning("drain.item_failed id=%s: %s", item["id"], exc, exc_info=True)
            pending_work.bump_attempt(item["id"], str(exc))
    if processed:
        logger.info("drain.done processed=%d remaining=%d", processed, pending_work.count())
        await broadcast_budget_status(app, force=True)
    return processed
