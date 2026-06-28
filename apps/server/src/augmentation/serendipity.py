"""SerendipityEngine: random walks and surprise scoring for serendipitous discovery."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class SerendipityEngine:
    """Provides random-walk exploration and surprise scoring over the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        """Create a SerendipityEngine.

        Args:
            graph: The KnowledgeGraph to explore.
        """
        self._graph = graph

    def random_walk(
        self,
        start_entity_id: str,
        steps: int = 5,
    ) -> list[str]:
        """Perform a random walk from the starting entity.

        At each step, randomly chooses one of the current entity's neighbors.
        If the current entity has no neighbors, the walk stops early.

        Args:
            start_entity_id: The entity ID to start from.
            steps: Maximum number of steps to take.

        Returns:
            A list of entity IDs visited, starting with `start_entity_id`.
        """
        visited = [start_entity_id]
        current = start_entity_id
        for _ in range(steps):
            relations = self._graph.get_relations(current)
            neighbors = []
            for rel in relations:
                if rel.source_entity_id == current:
                    neighbors.append(rel.target_entity_id)
                elif rel.target_entity_id == current:
                    neighbors.append(rel.source_entity_id)
            if not neighbors:
                break
            current = random.choice(neighbors)
            visited.append(current)
        return visited

    def surprise_score(self, entity_id: str) -> float:
        """Compute a surprise score for an entity.

        Higher score means the entity is more surprising / unexpected.
        Based on inverse relation count and average confidence.
        Isolated entities get the highest surprise.

        Args:
            entity_id: The entity ID to score.

        Returns:
            A float in [0.0, 1.0].
        """
        relations = self._graph.get_relations(entity_id)
        degree = len(relations)
        if degree == 0:
            return 1.0

        avg_confidence = sum(r.confidence for r in relations) / degree
        # More relations and higher confidence = less surprising
        # Normalize using a sigmoid-like factor
        max_degree = max(1, self._graph.count_relations())
        degree_factor = degree / max_degree
        score = 1.0 - (avg_confidence * (1.0 - degree_factor))
        return round(max(0.0, min(1.0, score)), 4)

    def random_surprise(self) -> tuple[str, float]:
        """Pick a random entity and return its surprise score.

        Returns:
            A tuple of (entity_id, score).

        Raises:
            ValueError: If the graph contains no entities.
        """
        entity = self._graph.random_entity()
        if entity is None:
            raise ValueError("No entities in the graph")
        score = self.surprise_score(entity.id)
        return entity.id, score
