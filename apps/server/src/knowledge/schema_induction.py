"""Schema induction: LLM-driven discovery of new entity/relation types.

Uses the LLM to analyze existing graph data and suggest new types that aren't
yet in the SchemaRegistry. Suggestions are returned for human review — they are
NOT auto-applied.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from src.knowledge.graph import KnowledgeGraph
from src.knowledge.schema import SchemaRegistry
from src.llm.base import LLMProvider
from src.llm.json_parse import LLMOutputParseError, parse_llm_json
from src.models.entity import Entity, EntityType
from src.models.relation import Relation, RelationType

logger = logging.getLogger(__name__)

# Timeout for LLM induction calls (seconds).
LLM_INDUCTION_TIMEOUT = 60

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Known entity type values from the enum — used to identify "unknown" types.
_KNOWN_ENTITY_TYPES: set[str] = {et.value for et in EntityType}

# Known relation type values from the enum.
_KNOWN_RELATION_TYPES: set[str] = {rt.value for rt in RelationType}


def _entity_to_flat_dict(entity: Entity) -> dict[str, Any]:
    """Serialize an entity to a flat dict suitable for LLM prompts."""
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type.value,
        "aliases": entity.aliases,
        "properties": entity.properties,
        "confidence": entity.confidence,
    }


def _relation_to_flat_dict(relation: Relation) -> dict[str, Any]:
    """Serialize a relation to a flat dict suitable for LLM prompts."""
    return {
        "source_entity_id": relation.source_entity_id,
        "target_entity_id": relation.target_entity_id,
        "type": relation.type.value,
        "confidence": relation.confidence,
        "properties": relation.properties,
    }


# --------------------------------------------------------------------------- #
# SchemaInducer
# --------------------------------------------------------------------------- #


class SchemaInducer:
    """Uses an LLM to discover new entity/relation types from graph data.

    Suggestions are returned for human review — they are never auto-applied
    to the SchemaRegistry. A separate approval step is required.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        schema_registry: SchemaRegistry,
        knowledge_graph: KnowledgeGraph,
    ) -> None:
        self._llm = llm_provider
        self._registry = schema_registry
        self._graph = knowledge_graph

    # ------------------------------------------------------------------ #
    # Entity analysis
    # ------------------------------------------------------------------ #

    async def analyze_entities(self, entities: list[Entity] | None = None) -> list[dict[str, Any]]:
        """Analyze unclassified/low-confidence entities and suggest new types.

        Args:
            entities: Optional pre-filtered list. If None, fetches all entities
                      from the graph and filters to those with unknown types
                      or confidence < 0.5.

        Returns:
            List of suggestion dicts, each with keys: name, description, parent_type.
        """
        if entities is None:
            entities = self._graph.find_entities(limit=500)

        # Filter to entities worth analyzing: unknown type or low confidence.
        candidates = [
            e for e in entities if e.type.value not in _KNOWN_ENTITY_TYPES or e.confidence < 0.5
        ]

        if not candidates:
            logger.info("Schema induction: no entity candidates to analyze")
            return []

        # Build a compact summary for the LLM.
        existing_types = sorted(self._registry._entity_types.keys())
        candidate_summary = [
            {
                "name": e.name,
                "current_type": e.type.value,
                "confidence": e.confidence,
                "aliases": e.aliases[:5],
                "properties": e.properties,
            }
            for e in candidates[:100]  # cap to avoid huge prompts
        ]

        prompt = (
            "You are analyzing a knowledge graph to discover new entity types.\n\n"
            f"Existing entity types: {json.dumps(existing_types)}\n\n"
            "Below are entities that are either unclassified (type not in the existing list) "
            "or have low confidence. Based on their names, aliases, and properties, "
            "suggest new entity types that would better categorize them.\n\n"
            f"Candidate entities:\n{json.dumps(candidate_summary, indent=2)}\n\n"
            "Return a JSON array of suggested new entity types. Each object must have:\n"
            '  - "name": a short CamelCase name for the type (e.g. "Software", "Theory")\n'
            '  - "description": a one-sentence description of what this type represents\n'
            '  - "parent_type": an existing type this could be a subtype of, or null\n'
            "Return ONLY the JSON array, nothing else."
        )

        try:
            response = await asyncio.wait_for(
                self._llm.complete(prompt=prompt),
                timeout=LLM_INDUCTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("Schema induction: entity analysis timed out")
            return []
        except Exception as exc:
            logger.error("Schema induction: entity analysis LLM call failed: %s", exc)
            return []

        if not response or not isinstance(response, str):
            logger.warning("Schema induction: empty entity analysis response")
            return []

        try:
            suggestions = parse_llm_json(response, expect="list")
        except LLMOutputParseError as exc:
            logger.warning("Schema induction: failed to parse entity suggestions: %s", exc)
            return []

        # Validate and normalize each suggestion.
        result: list[dict[str, Any]] = []
        for item in suggestions:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            result.append(
                {
                    "name": str(item["name"]).strip(),
                    "description": str(item.get("description", "")).strip(),
                    "parent_type": item.get("parent_type"),
                }
            )

        logger.info(
            "Schema induction: entity analysis produced %d suggestions from %d candidates",
            len(result),
            len(candidates),
        )
        return result

    # ------------------------------------------------------------------ #
    # Relation analysis
    # ------------------------------------------------------------------ #

    async def analyze_relations(
        self, relations: list[Relation] | None = None
    ) -> list[dict[str, Any]]:
        """Analyze relations and suggest new relation types.

        Args:
            relations: Optional pre-filtered list. If None, fetches all
                       relations from the graph.

        Returns:
            List of suggestion dicts, each with keys: name, description, inverse_name.
        """
        if relations is None:
            relations = self._graph.get_all_relations(limit=500)

        if not relations:
            logger.info("Schema induction: no relations to analyze")
            return []

        # Build a summary: group by type and show counts + examples.
        existing_types = sorted(self._registry._relation_types.keys())
        type_counts: dict[str, int] = {}
        type_examples: dict[str, list[dict[str, Any]]] = {}
        for r in relations:
            t = r.type.value
            type_counts[t] = type_counts.get(t, 0) + 1
            if len(type_examples.get(t, [])) < 3:
                type_examples.setdefault(t, []).append(
                    {
                        "source": r.source_entity_id,
                        "target": r.target_entity_id,
                        "confidence": r.confidence,
                    }
                )

        relation_summary = {
            "type_counts": type_counts,
            "type_examples": type_examples,
        }

        prompt = (
            "You are analyzing a knowledge graph to discover new relation types.\n\n"
            f"Existing relation types: {json.dumps(existing_types)}\n\n"
            "Below is a summary of relations currently in the graph, grouped by type. "
            "Based on the patterns you see, suggest new relation types that would "
            "better capture the semantics of these connections.\n\n"
            f"Relation summary:\n{json.dumps(relation_summary, indent=2)}\n\n"
            "Return a JSON array of suggested new relation types. Each object must have:\n"
            '  - "name": a short snake_case name (e.g. "authored_by", "supersedes")\n'
            '  - "description": a one-sentence description of the relationship\n'
            '  - "inverse_name": the inverse relation name (e.g. "authored" for "authored_by"), or null\n'
            "Return ONLY the JSON array, nothing else."
        )

        try:
            response = await asyncio.wait_for(
                self._llm.complete(prompt=prompt),
                timeout=LLM_INDUCTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("Schema induction: relation analysis timed out")
            return []
        except Exception as exc:
            logger.error("Schema induction: relation analysis LLM call failed: %s", exc)
            return []

        if not response or not isinstance(response, str):
            logger.warning("Schema induction: empty relation analysis response")
            return []

        try:
            suggestions = parse_llm_json(response, expect="list")
        except LLMOutputParseError as exc:
            logger.warning("Schema induction: failed to parse relation suggestions: %s", exc)
            return []

        result: list[dict[str, Any]] = []
        for item in suggestions:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            result.append(
                {
                    "name": str(item["name"]).strip(),
                    "description": str(item.get("description", "")).strip(),
                    "inverse_name": item.get("inverse_name"),
                }
            )

        logger.info(
            "Schema induction: relation analysis produced %d suggestions",
            len(result),
        )
        return result

    # ------------------------------------------------------------------ #
    # Full induction cycle
    # ------------------------------------------------------------------ #

    async def run_induction(self) -> dict[str, Any]:
        """Run a full induction cycle: analyze entities + relations.

        Returns:
            Dict with keys:
              - entity_suggestions: list of entity type suggestion dicts
              - relation_suggestions: list of relation type suggestion dicts
        """
        entity_suggestions = await self.analyze_entities()
        relation_suggestions = await self.analyze_relations()

        return {
            "entity_suggestions": entity_suggestions,
            "relation_suggestions": relation_suggestions,
        }
