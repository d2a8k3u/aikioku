"""GraphRetriever: entity-based retrieval using a KnowledgeGraph."""

from __future__ import annotations

from src.knowledge.graph import KnowledgeGraph
from src.models.entity import Entity
from src.retrieval.search_result import SearchResult


class GraphRetriever:
    """Retrieves notes by searching for entities in a KnowledgeGraph.

    Uses graph.find_entities() to find matching entities, then maps
    their source_note_ids to SearchResult objects.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        """Create a GraphRetriever.

        Args:
            graph: The KnowledgeGraph to search for entities.
        """
        self._graph = graph

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search for notes by finding entities matching the query.

        Looks up entities whose name contains the query string, then
        returns a SearchResult for each unique source_note_id.

        Args:
            query: The search query text (matched against entity names).
            limit: Maximum number of results to return.

        Returns:
            A list of SearchResult objects with source="graph".
        """
        entities = self._graph.find_entities(name=query, limit=limit)
        return self._entity_to_results(entities)[:limit]

    def _entity_to_results(self, entities: list[Entity]) -> list[SearchResult]:
        """Convert entities to SearchResults using their source_note_ids.

        Each unique source_note_id across all entities becomes one
        SearchResult. If multiple entities reference the same note,
        it appears only once in the output.

        Args:
            entities: List of Entity objects to convert.

        Returns:
            A list of SearchResult objects with source="graph".
        """
        seen: set[str] = set()
        results: list[SearchResult] = []
        for entity in entities:
            for note_id in entity.source_note_ids:
                if note_id not in seen:
                    seen.add(note_id)
                    results.append(
                        SearchResult(
                            note_id=note_id,
                            score=entity.confidence,
                            source="graph",
                            snippet=entity.name,
                        )
                    )
        return results
