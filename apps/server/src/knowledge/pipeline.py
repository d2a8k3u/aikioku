"""Note -> Knowledge Graph entity extraction pipeline.

Wires together LLM-based entity extraction, entity resolution,
KG storage, relation creation, and embedding storage.

Follows the MemoryExtractor pattern: build prompt -> call LLM -> parse response.
"""
from __future__ import annotations

import asyncio
import logging
import re

from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.entity_resolution import EntityResolver
from src.knowledge.graph import KnowledgeGraph
from src.llm.base import LLMProvider
from src.llm.json_parse import LLMOutputParseError, parse_llm_json
from src.models.entity import Entity, EntityType
from src.models.note import Note
from src.models.relation import Relation, RelationType

logger = logging.getLogger(__name__)

# Timeout in seconds for LLM entity extraction calls
LLM_ENTITY_EXTRACTION_TIMEOUT = 60


# --------------------------------------------------------------------------- #
# Entity extraction
# --------------------------------------------------------------------------- #


def _build_entity_extraction_prompt(content: str) -> str:
    """Build a prompt that instructs the LLM to extract named entities.

    Returns a JSON array of objects with keys: name, type, aliases, confidence.
    """
    return (
        "Extract named entities from the following text as a JSON array of objects.\n"
        "Each object must have keys: name, type, aliases, confidence.\n"
        "Type must be one of: Person, Place, Concept, Project, Event, Organization, Document, Task.\n"
        "Aliases is a list of alternative names (strings).\n"
        "Confidence is a float between 0.0 and 1.0.\n"
        "Return only the JSON array, nothing else.\n\n"
        f"Text:\n{content}"
    )


# Map common free-form types the LLM emits onto the fixed EntityType enum, so we
# don't drop entities just because the model used "Library"/"Framework"/etc.
_TYPE_SYNONYMS: dict[str, EntityType] = {
    "programminglanguage": EntityType.Concept,
    "language": EntityType.Concept,
    "framework": EntityType.Concept,
    "library": EntityType.Concept,
    "tool": EntityType.Concept,
    "technology": EntityType.Concept,
    "software": EntityType.Concept,
    "method": EntityType.Concept,
    "methodology": EntityType.Concept,
    "technique": EntityType.Concept,
    "algorithm": EntityType.Concept,
    "skill": EntityType.Concept,
    "topic": EntityType.Concept,
    "field": EntityType.Concept,
    "protocol": EntityType.Concept,
    "format": EntityType.Concept,
    "company": EntityType.Organization,
    "organisation": EntityType.Organization,
    "org": EntityType.Organization,
    "team": EntityType.Organization,
    "product": EntityType.Project,
    "system": EntityType.Project,
    "app": EntityType.Project,
    "application": EntityType.Project,
    "service": EntityType.Project,
    "platform": EntityType.Project,
    "people": EntityType.Person,
    "author": EntityType.Person,
    "user": EntityType.Person,
    "location": EntityType.Place,
    "city": EntityType.Place,
    "country": EntityType.Place,
    "paper": EntityType.Document,
    "file": EntityType.Document,
    "doc": EntityType.Document,
    "book": EntityType.Document,
    "action": EntityType.Task,
}


def _coerce_entity_type(raw: object) -> EntityType:
    """Map an LLM-provided type string onto the EntityType enum.

    Exact (case-insensitive) match wins; then a synonym table; otherwise default
    to Concept so a technical entity is never dropped for an out-of-enum type.
    """
    if not raw:
        return EntityType.Concept
    s = str(raw).strip()
    for t in EntityType:
        if s.lower() == t.value.lower():
            return t
    key = re.sub(r"[^a-z]", "", s.lower())
    return _TYPE_SYNONYMS.get(key, EntityType.Concept)


def _parse_entities_from_llm(response: str, note_id: str) -> list[Entity]:
    """Parse the LLM response into a list of Entity objects.

    Each entity gets the note_id added to its source_note_ids.
    """
    try:
        data = parse_llm_json(response, expect="list")
    except LLMOutputParseError as exc:
        logger.warning("Entity extraction: failed to parse LLM response as JSON: %s", exc)
        return []

    entities: list[Entity] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        try:
            entity = Entity(
                name=str(item["name"]).strip(),
                type=_coerce_entity_type(item.get("type")),
                aliases=item.get("aliases", []) if isinstance(item.get("aliases"), list) else [],
                confidence=float(item.get("confidence", 0.5)),
                source_note_ids=[note_id] if note_id else [],
            )
            entities.append(entity)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Entity extraction: skipping malformed entity item: %s", exc)
            continue

    return entities


async def extract_entities_from_note(
    note: Note,
    llm_provider: LLMProvider,
    graph: KnowledgeGraph,
) -> list[Entity]:
    """Extract entities from a note into the KG.

    Thin wrapper over :func:`extract_entities_from_text` using the note's content
    and id (id recorded as provenance in each entity's ``source_note_ids``).
    """
    return await extract_entities_from_text(
        text=note.content,
        source_id=note.id,
        llm_provider=llm_provider,
        graph=graph,
    )


async def extract_entities_from_text(
    text: str,
    source_id: str,
    llm_provider: LLMProvider,
    graph: KnowledgeGraph,
    source_is_note: bool = True,
) -> list[Entity]:
    """Extract entities from arbitrary text, resolve against KG, store, and relate.

    The graph is the project's source of truth, so chat turns enrich it just like
    notes do. Provenance is recorded WITHOUT conflating id namespaces:

    - ``source_is_note=True`` (notes): ``source_id`` is a note id and is appended
      to each entity's ``source_note_ids`` (drives note citations/retrieval).
    - ``source_is_note=False`` (chat turns): ``source_id`` is a conversation turn
      id and is recorded in ``properties["source_conversation_turns"]`` instead —
      keeping ``source_note_ids`` strictly note ids so chat turns never surface as
      broken ``/notes/<turn_id>`` citations.

    Pipeline:
        1. Build extraction prompt from the text
        2. Call LLM to get entity JSON (with timeout)
        3. Parse response into Entity objects (with format validation)
        4. Resolve each entity against existing KG (via EntityResolver)
        5. Store resolved entities in KG
        6. Create co-occurrence relations between extracted entities

    Returns the list of resolved entities stored in the KG.
    On any failure, logs the error and returns an empty list.
    """
    if not text or not text.strip():
        return []

    # Step 1: Build prompt
    prompt = _build_entity_extraction_prompt(text)

    # Step 2: Call LLM with timeout and error handling
    try:
        response = await asyncio.wait_for(
            llm_provider.complete(prompt=prompt),
            timeout=LLM_ENTITY_EXTRACTION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            "Entity extraction: LLM call timed out after %ds for source %s",
            LLM_ENTITY_EXTRACTION_TIMEOUT,
            source_id,
        )
        return []
    except Exception as exc:
        logger.error(
            "Entity extraction: LLM call failed for source %s: %s",
            source_id,
            exc,
            exc_info=True,
        )
        return []

    # Validate LLM response is a non-empty string
    if not response or not isinstance(response, str):
        logger.warning(
            "Entity extraction: empty or invalid LLM response for source %s (type=%s)",
            source_id,
            type(response).__name__,
        )
        return []

    # Step 3: Parse entities with format validation. Only seed source_note_ids
    # when the source is a real note — a chat turn id must not enter that field.
    raw_entities = _parse_entities_from_llm(
        response, note_id=source_id if source_is_note else ""
    )

    if not raw_entities:
        logger.info("No entities extracted from source %s", source_id)
        return []

    # Step 4 & 5: Resolve and store each entity
    resolver = EntityResolver(graph=graph)
    resolved_entities: list[Entity] = []
    for entity in raw_entities:
        try:
            resolved = resolver.resolve(entity)
            # Record provenance without conflating note ids and conversation ids.
            if source_is_note:
                if source_id not in resolved.source_note_ids:
                    resolved.source_note_ids.append(source_id)
                    graph.update_entity(resolved)
            else:
                turns = resolved.properties.setdefault("source_conversation_turns", [])
                if source_id not in turns:
                    turns.append(source_id)
                    graph.update_entity(resolved)
            resolved_entities.append(resolved)
        except Exception as exc:
            logger.warning(
                "Entity extraction: failed to resolve/store entity '%s' from source %s: %s",
                entity.name,
                source_id,
                exc,
            )
            continue

    # Step 6: Create co-occurrence relations between all pairs
    try:
        _create_cooccurrence_relations(resolved_entities, graph)
    except Exception as exc:
        logger.warning(
            "Entity extraction: failed to create co-occurrence relations for source %s: %s",
            source_id,
            exc,
        )

    logger.info(
        "Extracted %d entities from source %s, KG now has %d entities",
        len(resolved_entities),
        source_id,
        graph.count_entities(),
    )
    return resolved_entities


def _create_cooccurrence_relations(
    entities: list[Entity],
    graph: KnowledgeGraph,
) -> list[Relation]:
    """Create bidirectional 'related_to' relations between all co-occurring entity pairs.

    Only creates a relation if both entities exist in the graph and
    no relation already exists between them.
    """
    relations: list[Relation] = []
    for i, source in enumerate(entities):
        for j, target in enumerate(entities):
            if i >= j:
                continue  # skip self and duplicates
            # Verify both entities are in the graph
            if graph.get_entity(source.id) is None or graph.get_entity(target.id) is None:
                continue
            rel = Relation(
                source_entity_id=source.id,
                target_entity_id=target.id,
                type=RelationType.related_to,
                confidence=0.5,
                properties={"co_occurrence": True},
            )
            graph.create_relation(rel)
            relations.append(rel)
    return relations


# --------------------------------------------------------------------------- #
# Sub-window chunking
# --------------------------------------------------------------------------- #


def sub_window_chunk(
    text: str, window_size: int = 256, stride: int = 128
) -> list[dict]:
    """Split text into overlapping sub-windows using word-based tokenization.

    Each window is ``window_size`` words; windows slide by ``stride`` words
    (50% overlap when stride = window_size / 2).  Short texts (< window_size
    words) return a single window.

    Returns a list of dicts with keys: ``text``, ``start_word``, ``end_word``.
    """
    words = text.split()
    if not words:
        return []

    if len(words) <= window_size:
        return [{"text": text, "start_word": 0, "end_word": len(words)}]

    windows: list[dict] = []
    start = 0
    while start < len(words):
        end = min(start + window_size, len(words))
        window_text = " ".join(words[start:end])
        windows.append({
            "text": window_text,
            "start_word": start,
            "end_word": end,
        })
        if end >= len(words):
            break
        start += stride
    return windows


# --------------------------------------------------------------------------- #
# Embedding storage
# --------------------------------------------------------------------------- #


async def store_note_embeddings(
    note: Note,
    embedder: LLMProvider,
    embedding_store: EmbeddingStore,
) -> None:
    """Generate sub-window embeddings for a note and store them.

    Pipeline:
        1. Delete any existing embeddings for this note (idempotent re-embed).
        2. Split note content into semantic chunks (sentence-boundary-aware).
        3. For each chunk, generate overlapping sub-windows (256 words,
           128 stride → 50% overlap).
        4. Embed each sub-window separately via the LLM provider.
        5. Store each sub-window as a separate document with metadata:
           note_id, chunk_index, sub_window_index, window_start, window_end.

    At retrieval time the store groups by note_id and takes the max
    similarity across all sub-windows, so a note matches if *any* of its
    sub-windows is relevant to the query.
    """
    if not note.content:
        logger.info("Note %s has no content, skipping embedding", note.id)
        return

    # Step 1: Clear old embeddings for this note so re-embed is idempotent.
    embedding_store.delete(note.id)

    # Step 2: Semantic chunking (sentence-boundary-aware, ~2000 chars).
    from src.knowledge.embeddings import _semantic_chunks

    chunks = _semantic_chunks(note.content)
    if not chunks:
        chunks = [note.content]

    total_sub_windows = 0

    # Step 3–5: For each chunk → sub-windows → embed → store.
    for chunk_idx, chunk_text in enumerate(chunks):
        sub_windows = sub_window_chunk(chunk_text)
        for sw_idx, sw in enumerate(sub_windows):
            embedding = await embedder.embed(sw["text"])
            doc_id = f"{note.id}#c{chunk_idx}#sw{sw_idx}"
            embedding_store.add_document(
                doc_id=doc_id,
                text=sw["text"],
                embedding=embedding,
                metadata={
                    "note_id": note.id,
                    "chunk_index": chunk_idx,
                    "sub_window_index": sw_idx,
                    "window_start": sw["start_word"],
                    "window_end": sw["end_word"],
                },
            )
            total_sub_windows += 1

    logger.info(
        "Stored %d sub-window embeddings for note %s (%d semantic chunks)",
        total_sub_windows,
        note.id,
        len(chunks),
    )
