"""Sync memory (SPO) triples into the knowledge graph.

A memory's subject/object become resolved graph entities and its predicate a
directed relation, so memory knowledge is first-class in the graph (subgraph,
paths, discovery) alongside note-derived entities. Provenance lives in
``properties['source_memory_ids']`` on both entities and relations, so
consolidation can remove a memory's graph contribution without deleting entities
that notes or other memories still source.

Reuses the note pipeline: :class:`EntityResolver` for resolve-or-create (dedups a
memory's subject against the existing note entity of the same name) and
``_coerce_entity_type`` for enum coercion of LLM-assigned types.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from src.knowledge.entity_resolution import EntityResolver
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.pipeline import _coerce_entity_type
from src.llm.base import LLMProvider
from src.llm.json_parse import LLMOutputParseError, parse_llm_json
from src.models.entity import Entity, EntityType
from src.models.memory import Memory
from src.models.relation import Relation, RelationType

logger = logging.getLogger(__name__)

# Timeout in seconds for the batched entity-typing LLM call.
LLM_MEMORY_TYPING_TIMEOUT = 60

# Max names per typing LLM call.
LLM_MEMORY_TYPING_CHUNK_SIZE = 64

# Map free-form memory predicates onto the fixed RelationType enum. The original
# predicate is always preserved verbatim in ``relation.properties['predicate']``,
# so the related_to fallback loses no information.
_PREDICATE_SYNONYMS: dict[str, RelationType] = {
    "worksat": RelationType.works_at,
    "employedby": RelationType.works_at,
    "worksfor": RelationType.works_at,
    "joined": RelationType.works_at,
    "created": RelationType.created,
    "authored": RelationType.created,
    "built": RelationType.created,
    "wrote": RelationType.created,
    "developed": RelationType.created,
    "made": RelationType.created,
    "founded": RelationType.created,
    "invented": RelationType.created,
    "dependson": RelationType.depends_on,
    "requires": RelationType.depends_on,
    "uses": RelationType.depends_on,
    "needs": RelationType.depends_on,
    "basedon": RelationType.depends_on,
    "partof": RelationType.part_of,
    "memberof": RelationType.part_of,
    "belongsto": RelationType.part_of,
    "contains": RelationType.part_of,
    "includes": RelationType.part_of,
    "locatedin": RelationType.located_in,
    "livesin": RelationType.located_in,
    "bornin": RelationType.located_in,
    "basedin": RelationType.located_in,
    "mentions": RelationType.mentions,
    "references": RelationType.mentions,
    "describes": RelationType.mentions,
    "follows": RelationType.follows,
    "precededby": RelationType.follows,
    "succeeds": RelationType.follows,
}


def map_predicate(predicate: str) -> RelationType:
    """Map a free-form memory predicate onto the fixed RelationType enum.

    Exact (case-insensitive) enum match wins, then a keyword synonym table,
    otherwise ``related_to``. Purely keyword-based — no LLM.
    """
    if not predicate or not predicate.strip():
        return RelationType.related_to
    s = predicate.strip()
    for t in RelationType:
        if s.lower() == t.value.lower():
            return t
    key = re.sub(r"[^a-z]", "", s.lower())
    return _PREDICATE_SYNONYMS.get(key, RelationType.related_to)


def _build_typing_prompt(names: list[str]) -> str:
    joined = "\n".join(f"- {n}" for n in names)
    return (
        "Classify each of the following names into an entity type.\n"
        "Return a JSON array of objects with keys: name, type.\n"
        "Type must be one of: Person, Place, Concept, Project, Event, "
        "Organization, Document, Task.\n"
        "Return only the JSON array, nothing else.\n\n"
        f"Names:\n{joined}"
    )


async def type_names(names: list[str], llm: LLMProvider) -> dict[str, EntityType]:
    """Assign an EntityType to each name in ONE batched LLM call.

    Returns ``{name: EntityType}`` keyed by the input spelling. Names missing from
    the response — or any LLM/parse failure — are simply absent, and the caller
    defaults them to ``Concept``. Never raises.
    """
    unique = list({n.strip(): None for n in names if n and n.strip()})
    if not unique:
        return {}
    try:
        response = await asyncio.wait_for(
            llm.complete(prompt=_build_typing_prompt(unique)),
            timeout=LLM_MEMORY_TYPING_TIMEOUT,
        )
    except Exception as exc:  # timeout or provider error → degrade to Concept
        logger.warning("memory typing: LLM call failed: %s", exc)
        return {}
    if not response or not isinstance(response, str):
        return {}
    try:
        data = parse_llm_json(response, expect="list")
    except LLMOutputParseError as exc:
        logger.warning("memory typing: failed to parse LLM response: %s", exc)
        return {}
    by_lower = {n.lower(): n for n in unique}
    result: dict[str, EntityType] = {}
    for item in data:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"]).strip()
        original = by_lower.get(name.lower(), name)
        result[original] = _coerce_entity_type(item.get("type"))
    return result


async def type_names_chunked(
    names: list[str],
    llm: LLMProvider,
    chunk_size: int = LLM_MEMORY_TYPING_CHUNK_SIZE,
) -> dict[str, EntityType]:
    """Type many names across several batched LLM calls (one per ``chunk_size``)."""
    unique = list({n.strip(): None for n in names if n and n.strip()})
    result: dict[str, EntityType] = {}
    for start in range(0, len(unique), chunk_size):
        result.update(await type_names(unique[start : start + chunk_size], llm))
    return result


def _add_source_memory_id(properties: dict[str, Any], memory_id: str) -> bool:
    """Append ``memory_id`` to ``properties['source_memory_ids']`` if absent.

    Returns True when the list changed.
    """
    ids = properties.setdefault("source_memory_ids", [])
    if memory_id in ids:
        return False
    ids.append(memory_id)
    return True


def _find_entity_by_name(graph: KnowledgeGraph, name: str) -> Entity | None:
    """Exact-name lookup (``find_entities`` does substring CONTAINS, so filter)."""
    name = name.strip()
    for entity in graph.find_entities(name=name, limit=50):
        if entity.name == name:
            return entity
    return None


def _known_entity_types(graph: KnowledgeGraph, names: list[str]) -> dict[str, EntityType]:
    """Stored types for names already present in the graph (read-only)."""
    known: dict[str, EntityType] = {}
    for name in names:
        entity = _find_entity_by_name(graph, name)
        if entity is not None:
            known[name] = entity.type
    return known


async def sync_memory_to_graph(
    memory: Memory,
    graph: KnowledgeGraph,
    llm: LLMProvider,
    resolver: EntityResolver | None = None,
    types: dict[str, EntityType] | None = None,
) -> tuple[Entity, Entity, Relation] | None:
    """Resolve a memory's subject & object to graph entities and upsert a relation.

    Idempotent: re-syncing the same memory merges provenance instead of creating a
    duplicate edge. Returns ``(subject_entity, object_entity, relation)`` or None
    when the triple is unusable (blank subject/object).
    """
    subject = memory.subject.strip()
    obj = memory.object.strip()
    if not subject or not obj:
        return None

    resolver = resolver or EntityResolver(graph)
    if types is None:
        types = await type_names([subject, obj], llm)

    def _resolve(name: str) -> Entity:
        raw = Entity(
            name=name,
            type=types.get(name, EntityType.Concept),
            confidence=memory.confidence,
            properties={"source_memory_ids": [memory.id]},
        )
        resolved = resolver.resolve(raw)
        changed = _add_source_memory_id(resolved.properties, memory.id)
        # resolve() returns the pre-existing entity (different id) on a match;
        # blend its confidence toward the memory's.
        if resolved.id != raw.id:
            resolved.confidence = round((resolved.confidence + memory.confidence) / 2, 4)
            changed = True
        if changed:
            graph.update_entity(resolved)
        return resolved

    subj_entity = _resolve(subject)
    obj_entity = _resolve(obj)

    rel_type = map_predicate(memory.predicate)
    existing = graph.find_relation(
        subj_entity.id, obj_entity.id, rel_type.value, predicate=memory.predicate
    )
    if existing is not None:
        if _add_source_memory_id(existing.properties, memory.id):
            existing.confidence = round((existing.confidence + memory.confidence) / 2, 4)
            graph.delete_relation(
                subj_entity.id, obj_entity.id, rel_type.value, predicate=memory.predicate
            )
            graph.create_relation(existing)
        return subj_entity, obj_entity, existing

    relation = Relation(
        source_entity_id=subj_entity.id,
        target_entity_id=obj_entity.id,
        type=rel_type,
        confidence=memory.confidence,
        properties={
            "predicate": memory.predicate,
            "source_memory_ids": [memory.id],
            "derived_from": "memory",
        },
    )
    graph.create_relation(relation)
    return subj_entity, obj_entity, relation


async def sync_memories_to_graph(
    memories: list[Memory],
    graph: KnowledgeGraph,
    llm: LLMProvider,
    resolver: EntityResolver | None = None,
) -> None:
    """Sync a batch of memories into the graph, isolating per-item failures."""
    resolver = resolver or EntityResolver(graph)

    all_names = list(
        {
            name: None
            for memory in memories
            for name in (memory.subject.strip(), memory.object.strip())
            if name
        }
    )
    known_types = _known_entity_types(graph, all_names)
    new_names = [name for name in all_names if name not in known_types]
    types_map = {**known_types, **await type_names_chunked(new_names, llm)}

    for memory in memories:
        try:
            await sync_memory_to_graph(memory, graph, llm, resolver, types=types_map)
        except Exception as exc:
            logger.warning("memory graph sync failed for %s: %s", memory.id, exc, exc_info=True)


def remove_memory_from_graph(memory: Memory, graph: KnowledgeGraph) -> None:
    """Remove a memory's graph contribution.

    Drops ``memory.id`` from the derived relation's provenance and deletes the
    relation when no other memory sources it. Each endpoint entity is deleted only
    when it has no remaining source at all (empty ``source_note_ids`` AND empty
    ``properties['source_memory_ids']``); otherwise its provenance is just trimmed.
    """
    subj_entity = _find_entity_by_name(graph, memory.subject)
    obj_entity = _find_entity_by_name(graph, memory.object)
    rel_type = map_predicate(memory.predicate)

    if subj_entity is not None and obj_entity is not None:
        rel = graph.find_relation(
            subj_entity.id, obj_entity.id, rel_type.value, predicate=memory.predicate
        )
        if rel is not None:
            ids = rel.properties.get("source_memory_ids", [])
            if memory.id in ids:
                ids.remove(memory.id)
            graph.delete_relation(
                subj_entity.id, obj_entity.id, rel_type.value, predicate=memory.predicate
            )
            if ids:
                rel.properties["source_memory_ids"] = ids
                graph.create_relation(rel)

    for entity in (subj_entity, obj_entity):
        if entity is None:
            continue
        ids = entity.properties.get("source_memory_ids", [])
        if memory.id in ids:
            ids.remove(memory.id)
            entity.properties["source_memory_ids"] = ids
        if not entity.source_note_ids and not entity.properties.get("source_memory_ids"):
            graph.delete_entity(entity.id)
        else:
            graph.update_entity(entity)
