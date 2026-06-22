"""ConnectionDiscovery: discover indirect connections between entities via graph traversal + embedding similarity."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.knowledge.embeddings import EmbeddingStore
    from src.knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """Represents a discovered connection between two entities.

    Attributes:
        path: Ordered list of entity IDs from source to target.
        strength: Combined graph + embedding similarity score [0.0, 1.0].
        explanation: Human-readable description of the connection.
    """
    path: list[str]
    strength: float
    explanation: str


class ConnectionDiscovery:
    """Discovers connections between entities using graph traversal and embedding similarity."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        embedding_store: EmbeddingStore,
    ) -> None:
        """Create a ConnectionDiscovery.

        Args:
            graph: The KnowledgeGraph for entity/relation lookup.
            embedding_store: The EmbeddingStore for semantic similarity.
        """
        self._graph = graph
        self._emb = embedding_store

    def discover_connections(
        self,
        entity_id: str,
        max_distance: int = 3,
    ) -> list[Connection]:
        """Discover connections from the given entity to others.

        Performs BFS up to max_distance hops, then scores each discovered
        path by combining graph confidence and embedding cosine similarity.

        Args:
            entity_id: The starting entity ID.
            max_distance: Maximum number of relation hops.

        Returns:
            A list of Connection objects sorted by descending strength.
        """
        discovered: dict[str, Connection] = {}
        visited: set[str] = set()

        def _dfs(current_id: str, depth: int, path: list[str], min_confidence: float) -> None:
            if depth > max_distance:
                return
            visited.add(current_id)
            relations = self._graph.get_relations(current_id)
            for rel in relations:
                # Follow outgoing edges from current_id
                if rel.source_entity_id == current_id:
                    next_id = rel.target_entity_id
                elif rel.target_entity_id == current_id:
                    next_id = rel.source_entity_id
                else:
                    continue
                if next_id in visited or next_id == entity_id:
                    continue
                new_path = path + [next_id]
                conf = min(min_confidence, rel.confidence)
                if next_id not in discovered or conf > discovered[next_id].strength:
                    discovered[next_id] = Connection(
                        path=new_path,
                        strength=conf,
                        explanation="",
                    )
                _dfs(next_id, depth + 1, new_path, conf)
            visited.discard(current_id)

        _dfs(entity_id, 1, [entity_id], 1.0)

        # Build final connections with explanations and refined scores
        results: list[Connection] = []
        for target_id, conn in discovered.items():
            target_entity = self._graph.get_entity(target_id)
            source_entity = self._graph.get_entity(entity_id)
            if target_entity is None or source_entity is None:
                continue

            # Refine strength with embedding similarity if both have source notes
            sim = self._embedding_similarity(source_entity.source_note_ids, target_entity.source_note_ids)
            strength = conn.strength * 0.6 + sim * 0.4

            # Build path of names for explanation
            path_names = []
            for pid in conn.path:
                ent = self._graph.get_entity(pid)
                path_names.append(ent.name if ent else pid)

            if len(path_names) <= 2:
                explanation = f"{path_names[0]} is directly connected to {path_names[-1]}."
            else:
                steps = " → ".join(path_names)
                explanation = f"Connection path: {steps}."

            results.append(Connection(
                path=conn.path,
                strength=round(strength, 4),
                explanation=explanation,
            ))

        # Sort by strength descending and deduplicate by target (last element)
        results.sort(key=lambda c: c.strength, reverse=True)
        seen: set[str] = set()
        deduped: list[Connection] = []
        for c in results:
            target = c.path[-1]
            if target not in seen:
                seen.add(target)
                deduped.append(c)

        return deduped

    def _embedding_similarity(
        self,
        source_note_ids: list[str],
        target_note_ids: list[str],
    ) -> float:
        """Compute max pairwise cosine similarity between source and target note embeddings.

        Args:
            source_note_ids: Note IDs linked to the source entity.
            target_note_ids: Note IDs linked to the target entity.

        Returns:
            Cosine similarity in [0.0, 1.0]. Returns 0.0 if no embeddings found.
        """
        if not source_note_ids or not target_note_ids:
            return 0.0

        best = 0.0
        emb_map = self._emb.get_embeddings(source_note_ids + target_note_ids)
        import numpy as np
        for s_id in source_note_ids:
            s_vec = emb_map.get(s_id)
            if s_vec is None:
                continue
            for t_id in target_note_ids:
                t_vec = emb_map.get(t_id)
                if t_vec is None:
                    continue
                s = np.array(s_vec, dtype=np.float32)
                t = np.array(t_vec, dtype=np.float32)
                norm = np.linalg.norm(s) * np.linalg.norm(t)
                if norm > 0:
                    sim = float(np.dot(s, t) / norm)
                    if sim > best:
                        best = sim
        return max(0.0, min(1.0, best))
