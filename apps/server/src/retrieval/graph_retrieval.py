"""GraphRetriever: entity-based retrieval using a KnowledgeGraph."""

from __future__ import annotations

import string

from src.knowledge.graph import KnowledgeGraph
from src.models.entity import Entity
from src.retrieval.search_result import SearchResult

# Properties that are internal provenance metadata, not user-facing.
_INTERNAL_PROPERTY_KEYS: frozenset[str] = frozenset(
    {"source_conversation_turns", "source_memory_ids"}
)

# Snippet truncation limits.
_MAX_RELATIONS = 10
_MAX_PROPERTIES_CHARS = 500
_MAX_SNIPPET_CHARS = 2000

# Common English stop words filtered out during query tokenization so that
# natural-language questions like "What is FEP?" reduce to meaningful tokens
# (["fep"]) that can be matched against entity names and aliases.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "what",
        "is",
        "a",
        "an",
        "of",
        "and",
        "how",
        "why",
        "when",
        "who",
        "tell",
        "me",
        "about",
        "explain",
        "describe",
        "are",
        "was",
        "were",
        "can",
        "could",
        "do",
        "does",
        "did",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
        "this",
        "that",
        "these",
        "those",
    }
)


def _tokenize_query(query: str) -> list[str]:
    """Split a natural-language query into meaningful search tokens.

    Strips punctuation, removes stop words (case-insensitive), and drops
    tokens shorter than 2 characters.  Original casing is preserved because
    Kuzu's ``CONTAINS`` is case-sensitive — ``"FEP"`` matches the alias
    ``"FEP"`` but ``"fep"`` does not.

    For example ``"What is FEP?"`` becomes ``["FEP"]`` and
    ``"Tell me about Active interface"`` becomes ``["Active", "interface"]``.

    Args:
        query: The raw query string (may be a full question).

    Returns:
        A list of meaningful tokens (original casing), preserving first-seen
        order.  Duplicate tokens (case-insensitive) are collapsed to the first
        occurrence.
    """
    translator = str.maketrans("", "", string.punctuation)
    raw_tokens = query.translate(translator).split()
    seen: set[str] = set()
    tokens: list[str] = []
    for tok in raw_tokens:
        if len(tok) < 2:
            continue
        if tok.lower() in _STOP_WORDS:
            continue
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(tok)
    return tokens


def build_entity_context_snippet(entity: Entity, graph: KnowledgeGraph) -> str:
    """Build a rich context snippet for an entity from the knowledge graph.

    Includes the entity's name, type, aliases, relations, and user-facing
    properties. Internal provenance fields are excluded. The snippet is
    truncated to stay within reasonable display bounds.

    Args:
        entity: The entity to build a snippet for.
        graph: The KnowledgeGraph to query for relations and related entities.

    Returns:
        A multi-line string describing the entity and its graph context.
    """
    lines: list[str] = [f"{entity.name} ({entity.type.value})"]

    # Aliases
    if entity.aliases:
        lines.append(f"Aliases: {', '.join(entity.aliases)}")

    # Relations
    relations = graph.get_relations(entity.id)
    if relations:
        rel_lines: list[str] = ["Relations:"]
        shown = 0
        for rel in relations:
            if shown >= _MAX_RELATIONS:
                break
            # Determine the "other" entity in the relation and its name.
            if rel.source_entity_id == entity.id:
                other_id = rel.target_entity_id
                # outgoing: entity -> other
                other = graph.get_entity(other_id)
                other_name = other.name if other else other_id
                rel_lines.append(f"  - {entity.name} {rel.type.value} {other_name}")
            else:
                other_id = rel.source_entity_id
                # incoming: other -> entity
                other = graph.get_entity(other_id)
                other_name = other.name if other else other_id
                rel_lines.append(f"  - {other_name} {rel.type.value} {entity.name}")
            shown += 1
        lines.extend(rel_lines)

    # Properties (exclude internal provenance fields)
    public_props = {k: v for k, v in entity.properties.items() if k not in _INTERNAL_PROPERTY_KEYS}
    if public_props:
        prop_lines: list[str] = ["Properties:"]
        prop_section_len = len("Properties:\n")
        for key, value in public_props.items():
            entry = f"  {key}: {value}"
            if prop_section_len + len(entry) + 1 > _MAX_PROPERTIES_CHARS:
                break
            prop_lines.append(entry)
            prop_section_len += len(entry) + 1
        lines.extend(prop_lines)

    lines.append("Source: knowledge graph (derived from conversation history)")

    snippet = "\n".join(lines)
    if len(snippet) > _MAX_SNIPPET_CHARS:
        snippet = snippet[:_MAX_SNIPPET_CHARS]
    return snippet


class GraphRetriever:
    """Retrieves notes by searching for entities in a KnowledgeGraph.

    Uses graph.find_entities() and graph.find_entities_by_alias() to find
    matching entities, then maps their source_note_ids to SearchResult objects.
    Orphaned entities (those with no source_note_ids) produce synthetic
    SearchResults with source_type="entity".
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        """Create a GraphRetriever.

        Args:
            graph: The KnowledgeGraph to search for entities.
        """
        self._graph = graph

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search for notes by finding entities matching the query.

        Looks up entities whose name contains the query string AND entities
        whose aliases contain the query string, merges the results (deduped
        by entity id), then returns a SearchResult for each unique source_note_id
        plus synthetic results for orphaned entities.

        Natural-language queries (e.g. ``"What is FEP?"``) are tokenized:
        the full query is tried first, then each meaningful token is searched
        individually against entity names and aliases so that short entity
        names / aliases buried inside a longer question still match.

        Args:
            query: The search query text (matched against entity names and aliases).
            limit: Maximum number of results to return.

        Returns:
            A list of SearchResult objects with source="graph".
        """
        seen_ids: set[str] = set()
        merged: list[Entity] = []

        # 1. Full query — handles exact name/alias substring matches.
        for entity in self._graph.find_entities(
            name=query, limit=limit
        ) + self._graph.find_entities_by_alias(alias=query, limit=limit):
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                merged.append(entity)

        # 2. Per-token search — handles natural-language questions.
        #    Kuzu's CONTAINS is case-sensitive, so we search multiple case
        #    variants of each token to catch entities stored in a different
        #    case (e.g. query "fep" matching alias "FEP").
        for token in _tokenize_query(query):
            if len(merged) >= limit:
                break
            seen_variants: set[str] = set()
            variants = [token, token.lower(), token.upper()]
            for variant in variants:
                if variant in seen_variants or len(merged) >= limit:
                    continue
                seen_variants.add(variant)
                for entity in self._graph.find_entities(
                    name=variant, limit=limit
                ) + self._graph.find_entities_by_alias(alias=variant, limit=limit):
                    if entity.id not in seen_ids:
                        seen_ids.add(entity.id)
                        merged.append(entity)

        return self._entity_to_results(merged)[:limit]

    def _entity_to_results(self, entities: list[Entity]) -> list[SearchResult]:
        """Convert entities to SearchResults.

        For entities WITH non-empty ``source_note_ids``: each unique
        source_note_id becomes one SearchResult with ``source_type="note"``.

        For entities with EMPTY ``source_note_ids`` (orphaned): a single
        synthetic SearchResult is produced with ``source_type="entity"``,
        ``note_id="entity:<id>"``, and a rich snippet built from the graph.

        Args:
            entities: List of Entity objects to convert.

        Returns:
            A list of SearchResult objects with source="graph".
        """
        seen_notes: set[str] = set()
        seen_entities: set[str] = set()
        results: list[SearchResult] = []

        for entity in entities:
            if entity.source_note_ids:
                for note_id in entity.source_note_ids:
                    if note_id not in seen_notes:
                        seen_notes.add(note_id)
                        results.append(
                            SearchResult(
                                note_id=note_id,
                                score=entity.confidence,
                                source="graph",
                                snippet=entity.name,
                                source_type="note",
                            )
                        )
            else:
                # Orphaned entity — produce a synthetic result.
                if entity.id not in seen_entities:
                    seen_entities.add(entity.id)
                    results.append(
                        SearchResult(
                            note_id=f"entity:{entity.id}",
                            score=entity.confidence,
                            source="graph",
                            snippet=build_entity_context_snippet(entity, self._graph),
                            metadata={
                                "entity_id": entity.id,
                                "entity_type": entity.type.value,
                                "entity_name": entity.name,
                            },
                            source_type="entity",
                        )
                    )
        return results
