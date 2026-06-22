"""EntityResolver: multi-signal entity resolution for KnowledgeGraph."""

from __future__ import annotations

from src.knowledge.graph import KnowledgeGraph
from src.models.entity import Entity


class EntityResolver:
    """Resolves entities against a KnowledgeGraph using multi-signal scoring.

    Scoring signals:
        - Exact name match: 1.0
        - Alias match: 0.9
        - Levenshtein distance < 3: 0.7
        - Type compatibility: 0.3

    Thresholds:
        - > 0.85: merge (high confidence match)
        - > 0.6: suggest (possible match)
    """

    MERGE_THRESHOLD = 0.85
    SUGGEST_THRESHOLD = 0.6

    def __init__(self, graph: KnowledgeGraph) -> None:
        """Initialize resolver with a KnowledgeGraph reference."""
        self.graph = graph

    def find_candidates(self, entity: Entity) -> list[Entity]:
        """Query KG for potential matches by name, alias, and type.

        Searches the graph for entities that could match the given entity
        using name containment, alias overlap, and type filtering.
        """

        candidates: dict[str, Entity] = {}

        # Search by name (partial match)
        if entity.name:
            for e in self.graph.find_entities(name=entity.name, limit=50):
                candidates[e.id] = e

        # Search by type
        type_results = self.graph.find_entities(type=entity.type.value, limit=50)
        for e in type_results:
            if e.id not in candidates:
                candidates[e.id] = e

        # Search by aliases: find entities whose aliases contain the entity name
        # or whose names match any of the entity's aliases
        if entity.name:
            # Look for entities that have this entity's name as an alias
            # We need to scan all entities and check aliases
            all_of_type = self.graph.find_entities(limit=200)
            for e in all_of_type:
                aliases = entity.aliases if entity.aliases else []
                # Check if candidate's name is in query entity's aliases
                if e.name in aliases and e.id not in candidates:
                    candidates[e.id] = e
                # Check if query name is in candidate's aliases
                try:
                    cand_aliases = e.aliases if e.aliases else []
                except Exception:
                    cand_aliases = []
                if entity.name in cand_aliases and e.id not in candidates:
                    candidates[e.id] = e

        return list(candidates.values())

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute the Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return EntityResolver._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    def compute_score(self, entity: Entity, candidate: Entity) -> float:
        """Compute a weighted multi-signal score between two entities.

        Signals:
            - Exact name match: 1.0 (terminal, no type bonus)
            - Alias match: 0.9 (terminal, no type bonus)
            - Levenshtein distance < 3: 0.7
            - Type compatibility: 0.3 (additive with levenshtein/only)
        """
        # Exact name match — terminal score
        if entity.name == candidate.name:
            return 1.0

        # Alias match — terminal score
        if (entity.name in candidate.aliases) or (candidate.name in entity.aliases):
            return 0.9

        score = 0.0

        # Levenshtein distance
        dist = self._levenshtein_distance(entity.name, candidate.name)
        if dist < 3:
            score = 0.7

        # Type compatibility (additive)
        if entity.type == candidate.type:
            score += 0.3

        return score

    def resolve(self, entity: Entity) -> Entity:
        """Find existing entity or create new one using multi-signal scoring.

        Finds candidates via find_candidates, scores each with compute_score,
        and if the best score exceeds MERGE_THRESHOLD (>0.85), returns the
        existing entity (merge). Otherwise, creates a new entity in the graph.
        """
        candidates = self.find_candidates(entity)

        if not candidates:
            # No candidates — create new entity
            return self.graph.create_entity(entity)

        # Score all candidates
        best_score = 0.0
        best_candidate: Entity | None = None
        for candidate in candidates:
            score = self.compute_score(entity, candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_score > self.MERGE_THRESHOLD and best_candidate is not None:
            # Merge: return the existing entity
            return best_candidate

        # Below threshold — create new entity
        return self.graph.create_entity(entity)

    def merge_entities(self, source_id: str, target_id: str) -> Entity:
        """Merge source entity into target entity.

        Combines aliases, properties, and source_note_ids from both entities.
        Keeps the higher confidence score. Removes the source entity from the graph.
        Returns the updated target entity.
        """
        source = self.graph.get_entity(source_id)
        target = self.graph.get_entity(target_id)

        if source is None:
            raise ValueError(f"Source entity {source_id} not found")
        if target is None:
            raise ValueError(f"Target entity {target_id} not found")

        # Merge aliases: union of both alias lists
        merged_aliases = list(target.aliases)
        for alias in source.aliases:
            if alias not in merged_aliases:
                merged_aliases.append(alias)

        # Merge properties: source properties fill in missing keys
        merged_properties = dict(target.properties)
        for key, value in source.properties.items():
            if key not in merged_properties:
                merged_properties[key] = value

        # Merge source_note_ids: union without duplicates
        merged_note_ids = list(target.source_note_ids)
        for note_id in source.source_note_ids:
            if note_id not in merged_note_ids:
                merged_note_ids.append(note_id)

        # Keep higher confidence
        merged_confidence = max(source.confidence, target.confidence)

        # Build updated target entity
        updated = Entity(
            id=target.id,
            name=target.name,
            type=target.type,
            aliases=merged_aliases,
            properties=merged_properties,
            confidence=merged_confidence,
            source_note_ids=merged_note_ids,
        )

        # Update target in graph, then delete source
        self.graph.update_entity(updated)
        self.graph.delete_entity(source_id)

        return updated
