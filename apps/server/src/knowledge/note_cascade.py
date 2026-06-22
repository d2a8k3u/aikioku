"""Cascade-delete a deleted note's derived artifacts across stores.

A note writes to several stores beyond its markdown file + SQLite index:
ChromaDB vectors (sub-window embeddings) and the Kuzu graph (entities extracted
from it and their co-occurrence relations). The delete endpoint removes the file
and index but, without this module, leaves those derived artifacts orphaned —
deleted notes keep surfacing in retrieval, citations 404, and the graph fills
with phantom entities/edges.

The note-graph cleanup mirrors :func:`src.memory.graph_sync.remove_memory_from_graph`:
provenance is trimmed and an entity is deleted only when nothing sources it any
more.
"""

from __future__ import annotations

import logging
from typing import Any

from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


def remove_note_vectors(note_id: str, embedding_store: EmbeddingStore) -> None:
    """Delete all of a note's sub-window vectors from the embedding store."""
    embedding_store.delete(note_id)


def remove_note_from_graph(note_id: str, graph: KnowledgeGraph) -> dict[str, Any]:
    """Remove a note's graph contribution.

    Trims ``note_id`` from each sourced entity's ``source_note_ids``. An entity is
    deleted (which also drops all its edges) only when it has no remaining source
    at all — empty ``source_note_ids`` AND empty ``properties['source_memory_ids']``
    AND empty ``properties['source_conversation_turns']``. Otherwise its provenance
    is trimmed and it survives.

    Returns ``{"deleted": int, "trimmed_ids": list[str]}`` where ``trimmed_ids`` are
    the surviving entities that referenced the note (input to co-occurrence pruning).
    """
    deleted = 0
    trimmed_ids: list[str] = []
    for entity in graph.find_entities_by_note_id(note_id):
        if note_id in entity.source_note_ids:
            entity.source_note_ids = [n for n in entity.source_note_ids if n != note_id]
        memory_ids = entity.properties.get("source_memory_ids") or []
        turn_ids = entity.properties.get("source_conversation_turns") or []
        if not entity.source_note_ids and not memory_ids and not turn_ids:
            graph.delete_entity(entity.id)
            deleted += 1
        else:
            graph.update_entity(entity)
            trimmed_ids.append(entity.id)
    return {"deleted": deleted, "trimmed_ids": trimmed_ids}


def prune_cooccurrence_for_entities(entity_ids: list[str], graph: KnowledgeGraph) -> int:
    """Drop co-occurrence edges among ``entity_ids`` that are no longer justified.

    A co-occurrence edge A-B is justified iff A and B still share a source note
    (``set(A.source_note_ids) & set(B.source_note_ids)`` is non-empty). Any edge that
    became unjustified by a delete sits between two entities that BOTH referenced the
    deleted note, so checking pairs among the surviving affected entities is complete.
    Memory-derived ``related_to`` edges (predicate-bearing) are left untouched.
    Returns the number of co-occurrence edges removed.
    """
    entities = [e for e in (graph.get_entity(eid) for eid in entity_ids) if e is not None]
    removed = 0
    for i, a in enumerate(entities):
        a_notes = set(a.source_note_ids)
        for b in entities[i + 1 :]:
            if a_notes & set(b.source_note_ids):
                continue
            removed += graph.delete_cooccurrence_relation(a.id, b.id)
            removed += graph.delete_cooccurrence_relation(b.id, a.id)
    return removed


def purge_note_artifacts(
    note_id: str,
    *,
    embedding_store: EmbeddingStore | None,
    graph: KnowledgeGraph | None,
) -> dict[str, Any]:
    """Remove a deleted note's derived artifacts from Chroma + Kuzu.

    Each step self-isolates so one store's failure cannot skip the others. Returns
    a counts dict for logging.
    """
    result: dict[str, Any] = {
        "vectors": False,
        "entities_deleted": 0,
        "entities_trimmed": 0,
        "cooccurrence_pruned": 0,
    }

    if embedding_store is not None:
        try:
            remove_note_vectors(note_id, embedding_store)
            result["vectors"] = True
        except Exception as exc:
            logger.error("note %s: vector cleanup failed: %s", note_id, exc, exc_info=True)

    if graph is not None:
        try:
            graph_result = remove_note_from_graph(note_id, graph)
            result["entities_deleted"] = graph_result["deleted"]
            trimmed_ids = graph_result["trimmed_ids"]
            result["entities_trimmed"] = len(trimmed_ids)
            try:
                result["cooccurrence_pruned"] = prune_cooccurrence_for_entities(trimmed_ids, graph)
            except Exception as exc:
                logger.error("note %s: co-occurrence prune failed: %s", note_id, exc, exc_info=True)
        except Exception as exc:
            logger.error("note %s: graph cleanup failed: %s", note_id, exc, exc_info=True)

    return result
