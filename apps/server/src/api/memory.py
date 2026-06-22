"""Memory API endpoints for extraction, consolidation, listing, and stats."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.config import settings
from src.events import EventBus
from src.knowledge.graph import KnowledgeGraph
from src.memory.consolidation import MemoryConsolidator
from src.memory.graph_sync import remove_memory_from_graph
from src.memory.extraction import MemoryExtractor
from src.models.memory import Memory, MemoryTier
from src.storage.note_store import NoteStore

# Amount by which an entity-scoped load reinforces a memory's vitality, capped
# at 1.0. Models a light spaced-repetition signal: recall keeps a memory alive.
_REINFORCE_DELTA = 0.05

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])

# Module-level instances — set by tests or startup
_extractor: MemoryExtractor | None = None
_consolidator: MemoryConsolidator | None = None

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    source TEXT,
    created TEXT NOT NULL,
    modified TEXT NOT NULL,
    vitality_score REAL NOT NULL DEFAULT 0.5,
    tier TEXT NOT NULL DEFAULT 'warm'
)
"""


def _ensure_table():
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(_DB_SCHEMA)


def _db_memory_to_dict(row: tuple) -> dict:
    return {
        "id": row[0],
        "subject": row[1],
        "predicate": row[2],
        "object": row[3],
        "confidence": row[4],
        "source": row[5],
        "created": row[6],
        "modified": row[7],
        "vitality_score": row[8],
        "tier": row[9],
    }


def _store_memories(memories: list[Memory]) -> None:
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        for m in memories:
            conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, subject, predicate, object, confidence, source, created, modified, vitality_score, tier)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    m.id,
                    m.subject,
                    m.predicate,
                    m.object,
                    m.confidence,
                    m.source,
                    m.created.isoformat(),
                    m.modified.isoformat(),
                    m.vitality_score,
                    m.tier.value,
                ),
            )


def _load_memories(entity: str | None = None) -> list[dict]:
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        if entity:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE subject = ? OR object = ? ORDER BY modified DESC",
                (entity, entity),
            )
        else:
            cursor = conn.execute("SELECT * FROM memories ORDER BY modified DESC")
        rows = [_db_memory_to_dict(row) for row in cursor.fetchall()]

    # An entity-scoped recall reinforces the matched memories' vitality (a light
    # spaced-repetition signal). The bump is persisted but the returned rows
    # already reflect the post-bump score so the caller sees the live value.
    if entity and rows:
        for row in rows:
            _reinforce_memory(row["id"])
            row["vitality_score"] = min(1.0, row["vitality_score"] + _REINFORCE_DELTA)
    return rows


def _reinforce_memory(memory_id: str) -> None:
    """Bump a memory's vitality_score (capped at 1.0) and persist it."""
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(
            "UPDATE memories SET vitality_score = MIN(1.0, vitality_score + ?) WHERE id = ?",
            (_REINFORCE_DELTA, memory_id),
        )


def _persist_consolidation(input_memories: list[Memory], processed: list[Memory]) -> None:
    """Persist the results of a consolidation run back to SQLite.

    Deletes any input memory that did NOT survive (removed by dedup/merge), then
    upserts every processed memory (including new ``consolidation_summary``
    memories) with its updated tier/confidence/vitality.

    Args:
        input_memories: The memories fed into the consolidator.
        processed: The memories returned by ``MemoryConsolidator.run``.
    """
    _ensure_table()
    processed_ids = {m.id for m in processed}
    removed_ids = [m.id for m in input_memories if m.id not in processed_ids]

    with sqlite3.connect(settings.sqlite_db_path) as conn:
        for mem_id in removed_ids:
            conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
    # Upsert survivors and any new summary memories with their updated fields.
    _store_memories(processed)


def _get_extractor(llm=None) -> MemoryExtractor:
    """Build a MemoryExtractor.

    A module-level override (``_extractor``, set by tests) takes precedence.
    Otherwise the extractor is built with the supplied ``llm`` — in production
    the app's shared ``app.state.llm_provider``. When no ``llm`` is supplied it
    falls back to the configured provider via ``build_llm_provider`` (honours
    OpenRouter / Ollama Cloud / local settings), never a bare localhost default.
    """
    if _extractor is not None:
        return _extractor
    event_bus = EventBus(settings.sqlite_db_path)
    if llm is None:
        from src.llm.factory import build_llm_provider

        llm = build_llm_provider()
    return MemoryExtractor(llm, event_bus)


def _get_consolidator(graph=None, llm=None) -> MemoryConsolidator:
    """Build a MemoryConsolidator.

    A module-level override (``_consolidator``, set by tests) takes precedence.
    Otherwise it is built around the SHARED ``graph`` (the app's single Kuzu
    handle — Kuzu is single-writer, so we must not open a second handle) and the
    app's ``llm`` provider.
    """
    if _consolidator is not None:
        return _consolidator
    event_bus = EventBus(settings.sqlite_db_path)
    if graph is None:
        kg_path = settings.sqlite_db_path.replace(".db", "_kg.db")
        graph = KnowledgeGraph(kg_path)
    return MemoryConsolidator(graph, event_bus, llm_provider=llm)


class ExtractRequest(BaseModel):
    note_id: str


class CreateMemoryRequest(BaseModel):
    text: str
    source: str = "user"


class UpdateMemoryRequest(BaseModel):
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    tier: str | None = None


# ------------------------------------------------------------------ public helpers (for test patching)


async def _extract_memories(note_id: str, llm=None, *, provider=None, store=None) -> list[dict]:
    """Extract memories from a note. Returns list of memory dicts.

    Args:
        note_id: The note to extract from.
        llm: The LLM provider (the app's shared provider in production).
        provider: The embedding provider, for the memory vector store.
        store: The memory vector store (chroma_memories).
    """
    note_store = NoteStore(settings.notes_dir)
    note = note_store.get(note_id)
    if note is None:
        raise HTTPException(status_code=404, detail=f"Note not found: {note_id}")
    extractor = _get_extractor(llm=llm)
    memories = await extractor.extract_from_note(note)
    _store_memories(memories)
    await _embed_memories(memories, provider, store)
    return [_memory_to_dict(m) for m in memories]


def _memory_from_dict(data: dict[str, Any]) -> Memory:
    """Reconstruct a Memory from the dict produced by :func:`_memory_to_dict`."""
    return Memory(
        id=data["id"],
        subject=data["subject"],
        predicate=data["predicate"],
        object=data["object"],
        confidence=data["confidence"],
        source=data["source"],
        created=datetime.fromisoformat(data["created"]),
        modified=datetime.fromisoformat(data["modified"]),
        vitality_score=data["vitality_score"],
        tier=MemoryTier(data["tier"]),
    )


def _memory_to_dict(memory: Memory) -> dict:
    return {
        "id": memory.id,
        "subject": memory.subject,
        "predicate": memory.predicate,
        "object": memory.object,
        "confidence": memory.confidence,
        "source": memory.source,
        "created": memory.created.isoformat(),
        "modified": memory.modified.isoformat(),
        "vitality_score": memory.vitality_score,
        "tier": memory.tier.value,
    }


def _memories_from_rows(rows: list[dict]) -> list[Memory]:
    """Reconstruct Memory objects from SQLite rows."""
    from datetime import datetime, timezone

    return [
        Memory(
            id=m["id"],
            subject=m["subject"],
            predicate=m["predicate"],
            object=m["object"],
            confidence=m["confidence"],
            source=m["source"],
            created=datetime.fromisoformat(m["created"]).replace(tzinfo=timezone.utc),
            modified=datetime.fromisoformat(m["modified"]).replace(tzinfo=timezone.utc),
            vitality_score=m["vitality_score"],
            tier=MemoryTier(m["tier"]),
        )
        for m in rows
    ]


async def _run_consolidation(graph=None, llm=None, *, provider=None, store=None) -> dict:
    """Run the consolidation pipeline on all stored memories and persist results.

    Loads persisted memories, runs the consolidator (using the SHARED graph and
    app LLM provider in production), persists the processed results back, mirrors
    the changes into the memory vector store, and returns a JSON-safe summary
    (the raw Memory objects are stripped).

    Args:
        graph: The shared KnowledgeGraph handle (app.state.knowledge_graph).
        llm: The shared LLM provider (app.state.llm_provider).
        provider: The shared embedding provider (app.state.embedding_provider).
        store: The memory vector store (app.state.memory_embedding_store).
    """
    consolidator = _get_consolidator(graph=graph, llm=llm)
    loaded = _memories_from_rows(_load_memories())
    summary = await consolidator.run(loaded)
    processed = summary.get("memories", [])
    _persist_consolidation(loaded, processed)
    processed_ids = {m.id for m in processed}
    removed_ids = [m.id for m in loaded if m.id not in processed_ids]
    await _persist_consolidation_vectors(removed_ids, processed, provider, store)
    return {k: v for k, v in summary.items() if k != "memories"}


async def _list_memories(entity: str | None = None) -> list[dict]:
    """List stored memories, optionally filtered by entity."""
    return _load_memories(entity)


async def _get_stats() -> dict:
    """Return memory counts by tier."""
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        hot = conn.execute("SELECT COUNT(*) FROM memories WHERE tier = 'hot'").fetchone()[0]
        warm = conn.execute("SELECT COUNT(*) FROM memories WHERE tier = 'warm'").fetchone()[0]
        cold = conn.execute("SELECT COUNT(*) FROM memories WHERE tier = 'cold'").fetchone()[0]
    return {"total": total, "hot": hot, "warm": warm, "cold": cold}


# ------------------------------------------------------------------ embedding + CRUD helpers

# Fields a PUT may change. Whitelisted so the dynamic UPDATE never interpolates
# caller-supplied column names.
_UPDATABLE_FIELDS = ("subject", "predicate", "object", "confidence", "tier")

# Backfill progress so the status endpoint can report on an in-flight run.
_backfill_state: dict = {"running": False, "processed": 0, "total": 0, "errors": 0}


async def _embed_one(
    memory_id: str,
    subject: str,
    predicate: str,
    object_: str,
    tier: str,
    source: str | None,
    provider,
    store,
) -> None:
    """Write-through a single memory triple into the vector store (best-effort).

    Embeds ``"subject predicate object"`` as one atomic document keyed by the
    memory id (stored under the ``note_id`` metadata key so ``EmbeddingStore``'s
    note-grouped ``search``/``delete`` work unchanged). A transient embed failure
    is logged, never raised — the SQLite write is the source of truth.
    """
    if provider is None or store is None:
        return
    try:
        text = f"{subject} {predicate} {object_}"
        embedding = await provider.embed(text)
        store.delete(memory_id)  # idempotent: drop any prior vector for this id
        store.add_document(
            memory_id,
            text,
            embedding,
            metadata={
                "note_id": memory_id,
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "tier": tier,
                "source": source or "",
            },
        )
    except Exception:
        logger.warning("memory embed failed for %s", memory_id, exc_info=True)


async def _embed_memories(memories: list[Memory], provider, store) -> None:
    """Write-through a batch of Memory objects into the vector store."""
    for m in memories:
        await _embed_one(
            m.id, m.subject, m.predicate, m.object, m.tier.value, m.source, provider, store
        )


def _get_memory(memory_id: str) -> dict | None:
    """Fetch a single memory by id, or None if absent."""
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return _db_memory_to_dict(row) if row else None


async def _update_memory(memory_id: str, fields: dict, *, provider=None, store=None) -> dict | None:
    """Update whitelisted fields of a memory, bump ``modified``, re-embed on triple change.

    Returns the updated row, or None if the memory does not exist.
    """
    if _get_memory(memory_id) is None:
        return None
    updates = {k: v for k, v in fields.items() if k in _UPDATABLE_FIELDS and v is not None}
    now = datetime.utcnow().isoformat()
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            conn.execute(
                f"UPDATE memories SET {set_clause}, modified = ? WHERE id = ?",
                [*updates.values(), now, memory_id],
            )
        else:
            conn.execute("UPDATE memories SET modified = ? WHERE id = ?", (now, memory_id))

    updated = _get_memory(memory_id)
    if updated is not None and any(k in updates for k in ("subject", "predicate", "object")):
        await _embed_one(
            updated["id"],
            updated["subject"],
            updated["predicate"],
            updated["object"],
            updated["tier"],
            updated["source"],
            provider,
            store,
        )
    return updated


def _delete_memory(memory_id: str, *, store=None, graph=None) -> bool:
    """Delete a memory and its vector + graph contribution.

    Returns False if the id was not found. The memory is fetched before the row is
    removed so its triple is available for graph cleanup. Vector and graph cleanup
    are each isolated: a downstream failure is logged, not raised, since the
    authoritative SQLite row is already gone.
    """
    existing = _get_memory(memory_id)
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        deleted = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,)).rowcount > 0
    if not deleted:
        return False
    if store is not None:
        try:
            store.delete(memory_id)
        except Exception:
            logger.warning("memory vector delete failed for %s", memory_id, exc_info=True)
    if graph is not None and existing is not None:
        try:
            remove_memory_from_graph(_memory_from_dict(existing), graph)
        except Exception:
            logger.warning("memory graph cleanup failed for %s", memory_id, exc_info=True)
    return True


async def _search_memories_semantic(query: str, limit: int, provider, store) -> list[dict]:
    """Semantic search over memory triples. Returns memory dicts with a ``score`` field.

    Hits are hydrated from SQLite (the source of truth); vectors orphaned by an
    external delete are silently dropped.
    """
    if provider is None or store is None:
        return []
    embedding = await provider.embed(query)
    results: list[dict] = []
    for hit in store.search(embedding, limit=limit):
        mem = _get_memory(hit["note_id"])
        if mem is not None:
            mem["score"] = hit["score"]
            results.append(mem)
    return results


async def _create_memory_from_text(
    text: str, source: str = "user", *, llm=None, provider=None, store=None
) -> list[dict]:
    """Create memory triples from free text via the LLM, persist + embed them."""
    extractor = _get_extractor(llm=llm)
    memories = await extractor.extract_from_text(text, source=source)
    _store_memories(memories)
    await _embed_memories(memories, provider, store)
    return [_memory_to_dict(m) for m in memories]


async def _persist_consolidation_vectors(
    removed_ids: list[str], processed: list[Memory], provider, store
) -> None:
    """Mirror a consolidation run into the vector store: drop removed, re-embed survivors.

    Kept separate from the synchronous ``_persist_consolidation`` so the SQLite
    persistence stays sync (and its test untouched) while the vector side is async.
    """
    if provider is None or store is None:
        return
    for mem_id in removed_ids:
        try:
            store.delete(mem_id)
        except Exception:
            logger.warning("memory vector delete failed for %s", mem_id, exc_info=True)
    await _embed_memories(processed, provider, store)


async def _backfill_memory_embeddings(provider, store) -> None:
    """Embed every stored memory (one-shot maintenance after first deploy / a reembed)."""
    mems = _memories_from_rows(_load_memories())
    _backfill_state.update(running=True, processed=0, total=len(mems), errors=0)
    for m in mems:
        try:
            await _embed_one(
                m.id, m.subject, m.predicate, m.object, m.tier.value, m.source, provider, store
            )
        except Exception:
            _backfill_state["errors"] += 1
        finally:
            _backfill_state["processed"] += 1
    _backfill_state["running"] = False
    logger.info("memory embedding backfill complete: %s", _backfill_state)


# ------------------------------------------------------------------ endpoints


@router.post("/extract")
async def extract_memories(request: Request, body: ExtractRequest) -> list[dict]:
    """Extract memories from a note and return the list.

    Uses the app's shared LLM provider (``app.state.llm_provider``). When the
    daily LLM budget is exhausted the extraction is queued — the note is the
    source of truth — and runs after the budget resets; an empty list is
    returned in that case.
    """
    from src.processing.budget_gate import WORK_MEMORY_EXTRACTION, gated

    result = await gated(
        request.app,
        WORK_MEMORY_EXTRACTION,
        body.note_id,
        {"mode": "note", "note_id": body.note_id},
    )
    return result if result is not None else []


@router.post("/consolidate")
async def consolidate_memories(request: Request) -> dict:
    """Trigger the memory consolidation pipeline and return a summary.

    Reuses the app's shared Kuzu handle (``app.state.knowledge_graph``) and LLM
    provider so it never opens a second single-writer Kuzu connection.
    """
    graph = getattr(request.app.state, "knowledge_graph", None)
    llm = getattr(request.app.state, "llm_provider", None)
    provider = getattr(request.app.state, "embedding_provider", None)
    store = getattr(request.app.state, "memory_embedding_store", None)
    return await _run_consolidation(graph=graph, llm=llm, provider=provider, store=store)


@router.get("/")
async def list_memories(
    entity: str | None = None,
) -> list[dict]:
    """List memories with optional entity filter."""
    return await _list_memories(entity)


@router.get("/stats")
async def memory_stats() -> dict:
    """Return memory statistics: total and tier counts."""
    return await _get_stats()


@router.post("/")
async def create_memory(request: Request, body: CreateMemoryRequest) -> list[dict]:
    """Create memory triples from free text. The LLM parses the text into
    subject-predicate-object triples, which are persisted and embedded. When the
    daily LLM budget is exhausted the text is queued and processed after the
    budget resets; an empty list is returned in that case."""
    from uuid import uuid4

    from src.processing.budget_gate import WORK_MEMORY_EXTRACTION, gated

    if getattr(request.app.state, "llm_provider", None) is None:
        raise HTTPException(status_code=503, detail="LLM provider not configured")
    result = await gated(
        request.app,
        WORK_MEMORY_EXTRACTION,
        f"text:{uuid4()}",
        {"mode": "text", "text": body.text, "source": body.source},
    )
    return result if result is not None else []


@router.get("/search")
async def search_memories(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
) -> list[dict]:
    """Semantic search over stored memories. Returns memory dicts with a ``score``."""
    provider = getattr(request.app.state, "embedding_provider", None)
    store = getattr(request.app.state, "memory_embedding_store", None)
    return await _search_memories_semantic(q, limit, provider, store)


@router.post("/backfill-embeddings")
async def backfill_memory_embeddings(request: Request) -> dict:
    """Embed every stored memory in the background (run after first deploy or a reembed).

    Returns immediately; poll GET /api/memory/backfill-embeddings/status.
    """
    if _backfill_state["running"]:
        return {"status": "already_running", **_backfill_state}
    provider = getattr(request.app.state, "embedding_provider", None)
    store = getattr(request.app.state, "memory_embedding_store", None)
    if provider is None or store is None:
        raise HTTPException(status_code=503, detail="Embedding provider not configured")
    total = len(_load_memories())
    asyncio.ensure_future(_backfill_memory_embeddings(provider, store))
    return {"status": "started", "total": total}


@router.get("/backfill-embeddings/status")
async def backfill_memory_embeddings_status() -> dict:
    """Return progress of the most recent memory embedding backfill run."""
    return dict(_backfill_state)


@router.get("/{memory_id}")
async def get_memory(memory_id: str) -> dict:
    """Fetch a single memory by id."""
    mem = _get_memory(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
    return mem


@router.put("/{memory_id}")
async def update_memory(request: Request, memory_id: str, body: UpdateMemoryRequest) -> dict:
    """Update a memory's subject/predicate/object/confidence/tier."""
    fields = body.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    provider = getattr(request.app.state, "embedding_provider", None)
    store = getattr(request.app.state, "memory_embedding_store", None)
    updated = await _update_memory(memory_id, fields, provider=provider, store=store)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
    return updated


@router.delete("/{memory_id}")
async def delete_memory(request: Request, memory_id: str) -> dict:
    """Delete a memory by id."""
    store = getattr(request.app.state, "memory_embedding_store", None)
    graph = getattr(request.app.state, "knowledge_graph", None)
    if not _delete_memory(memory_id, store=store, graph=graph):
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
    return {"deleted": True}
