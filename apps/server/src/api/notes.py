"""Notes API endpoints."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.auth import UserInDB, require_auth
from src.cache.semantic_cache import cache_invalidate
from src.llm.long_text import condense_for_prompt
from src.models.note import Note, NoteUpdate
from src.storage.note_store import NoteStore
from src.api.websocket import get_broadcaster

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notes", tags=["notes"])

# Module-level store for non-request contexts (tests, etc.)
_note_store: NoteStore | None = None


def get_note_store() -> NoteStore:
    global _note_store
    if _note_store is None:
        from src.config import settings

        _note_store = NoteStore(settings.notes_dir)
    return _note_store


def _get_note_store_from_request(request: Request) -> NoteStore:
    """Get the shared NoteStore from app state, creating it on first use.

    The lifespan installs the singleton in ``app.state.note_store`` (index
    warmed at startup). When absent (e.g. tests that only set the module-level
    store, or non-lifespan contexts) fall back to the module singleton, then to
    a freshly constructed store derived from settings.
    """
    store = getattr(request.app.state, "note_store", None)
    if store is None:
        store = get_note_store()
        request.app.state.note_store = store
    return store


def _extract_and_store_entities(request: Request, note: Note) -> None:
    """Schedule budget-gated processing (entity extraction + embedding) for a note.

    Fire-and-forget: runs in the background so it never blocks the HTTP response,
    and failures are logged, not propagated. When the daily LLM budget is
    exhausted the work is queued instead and drained after the budget resets or
    is raised; the note itself is already persisted by the caller.
    """
    import asyncio

    from src.processing.budget_gate import WORK_NOTE_PROCESSING, gated

    async def _do_processing() -> None:
        try:
            await gated(
                request.app,
                WORK_NOTE_PROCESSING,
                note.id,
                {"note_id": note.id},
            )
        except Exception as exc:
            logger.error("Note processing failed for note %s: %s", note.id, exc, exc_info=True)

    asyncio.ensure_future(_do_processing())


def _invalidate_sparse_index(request: Request) -> None:
    """Mark the shared sparse retriever dirty so the next search reindexes.

    The shared retriever lives at ``app.state.hybrid_fusion.sparse`` (built in
    the lifespan). Guarded with getattr so this is a no-op when fusion is unset
    (e.g. lightweight test contexts).
    """
    fusion = getattr(request.app.state, "hybrid_fusion", None)
    sparse = getattr(fusion, "sparse", None)
    if sparse is not None and hasattr(sparse, "mark_dirty"):
        sparse.mark_dirty()


def _store_note_embeddings(request: Request, note: Note) -> None:
    """No-op: a note's embeddings are produced inside the budget-gated processing
    task scheduled by :func:`_extract_and_store_entities`."""
    return None


def _purge_note_artifacts(request: Request, note_id: str) -> None:
    """Cascade-delete a deleted note's vectors + graph entities/relations.

    Resolves the shared stores from app state (unset in lightweight test contexts →
    no-op). Failures are logged, never raised: the note file/index is already gone,
    so the delete must still succeed even if a downstream store hiccups.
    """
    from src.knowledge.note_cascade import purge_note_artifacts

    embedding_store = getattr(request.app.state, "embedding_store", None)
    graph = getattr(request.app.state, "knowledge_graph", None)
    if embedding_store is None and graph is None:
        return
    try:
        result = purge_note_artifacts(note_id, embedding_store=embedding_store, graph=graph)
        logger.info("note %s delete cascade: %s", note_id, result)
    except Exception as exc:
        logger.error("note %s delete cascade failed: %s", note_id, exc, exc_info=True)


@router.get("/")
async def list_notes(
    request: Request,
    tag: str | None = None,
    search: str | None = None,
    source_type: str | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> list[Note]:
    """List notes with optional filtering.

    The unfiltered list path uses the metadata index (``store.list``) so a
    request reads at most ``limit`` files, never the whole corpus. Tag/search
    paths read only matching files / fall back to a scan respectively.
    """
    store = _get_note_store_from_request(request)
    if tag:
        notes: list[Note] = store.get_by_tag(tag)
        return notes[skip : skip + limit]
    if search:
        notes = store.search(search)
        return notes[skip : skip + limit]
    return store.list(skip=skip, limit=limit, source_type=source_type)


@router.get("/{note_id}")
async def get_note(request: Request, note_id: UUID) -> Note:
    """Get a single note by ID."""
    store = _get_note_store_from_request(request)
    note = store.get(str(note_id))
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.post("/")
async def create_note(
    request: Request,
    note: Note,
    user: UserInDB = Depends(require_auth),
) -> Note:
    """Create a new note and trigger entity extraction + embedding."""
    import asyncio

    store = _get_note_store_from_request(request)
    created = store.create(note)
    _extract_and_store_entities(request, created)
    _store_note_embeddings(request, created)
    _invalidate_sparse_index(request)
    # Invalidate semantic cache — new notes may change correct answers
    asyncio.ensure_future(cache_invalidate())
    # Publish event for downstream async processing
    event_bus = getattr(request.app.state, "event_bus", None)
    if event_bus is not None:
        from src.events import Event

        asyncio.ensure_future(
            event_bus.publish_async(Event("note.created", {"note_id": created.id}))
        )

    # Broadcast WebSocket events
    broadcaster = get_broadcaster()
    if broadcaster:
        asyncio.ensure_future(
            broadcaster.broadcast(
                "note.updated",
                {
                    "note_id": created.id,
                    "title": created.title,
                    "action": "created",
                },
            )
        )
        asyncio.ensure_future(
            broadcaster.broadcast(
                "stats.updated",
                {
                    "notes": store.count(),
                },
            )
        )

    # Auto-title generation: if title is "New Note" or empty, generate one via LLM
    if created.content.strip() and (
        created.title.strip() == "New Note" or not created.title.strip()
    ):
        from src import runtime_config

        if runtime_config.auto_title():
            asyncio.ensure_future(_auto_generate_title(request, created, store))

    return created


async def _auto_generate_title(request: Request, note: Note, store: NoteStore) -> None:
    """Generate a title for a note using the LLM and update it."""
    import asyncio

    try:
        llm_provider = request.app.state.llm_provider
        # Never truncate: fold oversized notes (chunk + summarize) so the title
        # reflects the whole note. Normal-size notes pass through verbatim.
        source = await condense_for_prompt(llm_provider, note.content)
        prompt = (
            "Generate a concise, descriptive title (max 10 words) for the following note content. "
            "Return ONLY the title, no quotes or extra text:\n\n" + source
        )
        generated_title = await llm_provider.complete(prompt)
        generated_title = generated_title.strip().strip('"').strip("'")
        if generated_title and len(generated_title) <= 120:
            note.title = generated_title
            note.modified = note.modified  # keep existing modified
            store.update(note)
            # Invalidate semantic cache — updated note may change correct answers
            asyncio.ensure_future(cache_invalidate())
            logger.info("Auto-generated title for note %s: %s", note.id, generated_title)
    except Exception as exc:
        logger.warning("Auto-title generation failed for note %s: %s", note.id, exc)


@router.put("/{note_id}")
async def update_note(
    request: Request,
    note_id: UUID,
    note: NoteUpdate,
    user: UserInDB = Depends(require_auth),
) -> Note:
    """Update an existing note and trigger entity extraction + embedding."""
    store = _get_note_store_from_request(request)
    existing = store.get(str(note_id))
    if existing is None:
        raise HTTPException(status_code=404, detail="Note not found")
    update_data = note.model_dump(exclude_unset=True)
    # Ensure path is preserved if not provided in the update payload
    if "path" not in update_data:
        update_data["path"] = existing.path
    merged = existing.model_copy(update=update_data)
    merged.id = str(note_id)
    updated = store.update(merged)
    _extract_and_store_entities(request, updated)
    _store_note_embeddings(request, updated)
    _invalidate_sparse_index(request)
    # Invalidate semantic cache — updated notes may change correct answers
    import asyncio

    asyncio.ensure_future(cache_invalidate())
    # Broadcast WebSocket event
    broadcaster = get_broadcaster()
    if broadcaster:
        asyncio.ensure_future(
            broadcaster.broadcast(
                "note.updated",
                {
                    "note_id": str(note_id),
                    "title": updated.title,
                    "action": "updated",
                },
            )
        )
    return updated


@router.delete("/{note_id}")
async def delete_note(
    request: Request,
    note_id: UUID,
    user: UserInDB = Depends(require_auth),
) -> dict[str, str]:
    """Delete a note."""
    store = _get_note_store_from_request(request)
    deleted = store.delete(str(note_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Note not found")
    _invalidate_sparse_index(request)
    _purge_note_artifacts(request, str(note_id))
    # Invalidate semantic cache — deleted notes may change correct answers
    import asyncio

    asyncio.ensure_future(cache_invalidate())
    # Broadcast WebSocket events
    broadcaster = get_broadcaster()
    if broadcaster:
        asyncio.ensure_future(broadcaster.broadcast("note.deleted", {"note_id": str(note_id)}))
        asyncio.ensure_future(broadcaster.broadcast("stats.updated", {"notes": store.count()}))
    return {"status": "deleted", "id": str(note_id)}


# ---------------------------------------------------------------------------
# Backlinks
# ---------------------------------------------------------------------------


@router.get("/{note_id}/backlinks")
async def get_backlinks(request: Request, note_id: UUID) -> list[dict[str, str]]:
    """Find notes that link to this note.

    Searches all notes for:
    - Wikilink references: [[note-id]] in content
    - Explicit links in the ``links`` frontmatter field
    """
    store = _get_note_store_from_request(request)
    target_id = str(note_id)

    # Verify the target note exists
    if store.get(target_id) is None:
        raise HTTPException(status_code=404, detail="Note not found")

    backlinks: list[dict[str, str]] = []
    wikilink_pattern = f"[[{target_id}]]"

    all_notes: list[Note] = store.list_all()
    for note in all_notes:
        if note.id == target_id:
            continue
        is_backlink = False
        reason = ""

        # Check wikilink references in content
        if wikilink_pattern in note.content:
            is_backlink = True
            reason = "wikilink"

        # Check explicit links in frontmatter
        if not is_backlink and target_id in note.links:
            is_backlink = True
            reason = "link"

        if is_backlink:
            backlinks.append(
                {
                    "id": note.id,
                    "title": note.title,
                    "path": note.path,
                    "reason": reason,
                }
            )

    return backlinks


# ---------------------------------------------------------------------------
# Related notes
# ---------------------------------------------------------------------------


@router.get("/{note_id}/related")
async def get_related_notes(request: Request, note_id: UUID) -> dict[str, Any]:
    """Get entities, related notes, and surprise connections for a note.

    Returns:
        entities: Entities linked to this note (source_note_ids contains note_id)
        related_notes: Other notes that share entities with this note
        surprise: Serendipity random walk from one of the note's entities
    """
    store = _get_note_store_from_request(request)
    target_id = str(note_id)

    # Verify the target note exists
    note = store.get(target_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")

    graph = request.app.state.knowledge_graph

    # Find entities linked to this note
    # We need to scan all entities since Kuzu stores source_note_ids as JSON string
    all_entities = graph.find_entities(limit=10000)
    note_entities: list[dict[str, Any]] = []
    for entity in all_entities:
        if target_id in entity.source_note_ids:
            note_entities.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type.value,
                    "confidence": entity.confidence,
                }
            )

    # Find related notes via shared entities
    related_note_ids: set[str] = set()
    for entity in all_entities:
        if target_id in entity.source_note_ids:
            # Other notes that share this entity
            for other_note_id in entity.source_note_ids:
                if other_note_id != target_id:
                    related_note_ids.add(other_note_id)

    related_notes: list[dict[str, str]] = []
    for rid in related_note_ids:
        related_note = store.get(rid)
        if related_note is not None:
            related_notes.append(
                {
                    "id": related_note.id,
                    "title": related_note.title,
                    "path": related_note.path,
                }
            )

    # Surprise connections: random walk from one of the note's entities
    surprise: list[dict[str, str]] = []
    if note_entities:
        from src.augmentation.serendipity import SerendipityEngine

        engine = SerendipityEngine(graph)
        start_entity_id = note_entities[0]["id"]
        try:
            walk_path = engine.random_walk(start_entity_id, steps=5)
            for entity_id in walk_path:
                entity = graph.get_entity(entity_id)
                if entity is not None:
                    surprise.append(
                        {
                            "id": entity.id,
                            "name": entity.name,
                            "type": entity.type.value,
                        }
                    )
        except Exception as exc:
            logger.warning("Serendipity walk failed for note %s: %s", target_id, exc)

    # AI insight: one-sentence summary of what makes this note interesting
    insight = ""
    if note_entities or related_notes:
        try:
            llm = getattr(request.app.state, "llm_provider", None)
            if llm is not None:
                entity_names = [e["name"] for e in note_entities[:3]]
                related_titles = [n["title"] for n in related_notes[:3]]
                prompt = (
                    f"Note: '{note.title}'. "
                    f"Connected entities: {', '.join(entity_names)}. "
                    f"Related notes: {', '.join(related_titles)}. "
                    f"In one sentence, written in the same language as the note, "
                    f"say what makes this note interesting "
                    f"in the user's knowledge graph."
                )
                insight = await llm.complete(prompt)
                insight = insight.strip().strip('"')
        except Exception:
            logger.warning("Insight generation failed for note %s", target_id)

    return {
        "entities": note_entities,
        "related_notes": related_notes,
        "surprise": surprise,
        "insight": insight,
    }


# ---------------------------------------------------------------------------
# Note versioning
# ---------------------------------------------------------------------------


@router.get("/{note_id}/history")
async def note_history(
    request: Request,
    note_id: UUID,
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, str]]:
    """Get Git commit history for a specific note."""
    from src.storage.git_sync import GitSync
    from src.config import settings

    gs = getattr(request.app.state, "git_sync", None)
    if gs is None:
        gs = GitSync(settings.notes_dir)
        request.app.state.git_sync = gs
    history = gs.get_note_history(str(note_id), limit=limit)
    return history


@router.get("/{note_id}/diff")
async def note_diff(
    request: Request,
    note_id: UUID,
    commit_a: str | None = None,
    commit_b: str | None = None,
) -> dict[str, str]:
    """Get diff for a specific note.

    If commit_a and commit_b are provided, returns the diff between them.
    Otherwise returns the latest diff (HEAD~1 vs HEAD).
    """
    from src.storage.git_sync import GitSync
    from src.config import settings

    gs = getattr(request.app.state, "git_sync", None)
    if gs is None:
        gs = GitSync(settings.notes_dir)
        request.app.state.git_sync = gs

    if commit_a and commit_b:
        file_path = f"{note_id}.md"
        try:
            result = gs._run_git("diff", commit_a, commit_b, "--", file_path)
            diff = result.stdout
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    else:
        diff = gs.get_diff(str(note_id))
    return {"note_id": str(note_id), "diff": diff}
