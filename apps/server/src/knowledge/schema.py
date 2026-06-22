"""Schema registry for knowledge graph entity and relation types."""

from __future__ import annotations

from src.models.entity import EntityType
from src.models.relation import RelationType


class SchemaRegistry:
    """Manages entity and relation type definitions, seeded from existing enums."""

    def __init__(self) -> None:
        self._entity_types: dict[str, dict] = {}
        self._relation_types: dict[str, dict] = {}
        # Seed with existing enums
        for et in EntityType:
            self._entity_types[et.value] = {"name": et.value}
        for rt in RelationType:
            self._relation_types[rt.value] = {"name": rt.value}

    def register_entity_type(self, name: str, description: str) -> None:
        """Register a new entity type."""
        self._entity_types[name] = {"name": name, "description": description}

    def register_relation_type(
        self, name: str, description: str, domain: str, range: str
    ) -> None:
        """Register a new relation type."""
        self._relation_types[name] = {
            "name": name,
            "description": description,
            "domain": domain,
            "range": range,
        }

    def list_types(self) -> dict:
        """Return a dict with keys entity_types and relation_types."""
        return {
            "entity_types": dict(self._entity_types),
            "relation_types": dict(self._relation_types),
        }

    def get_type(self, name: str) -> dict | None:
        """Return type info for name if it exists in either entity or relation types."""
        if name in self._entity_types:
            return dict(self._entity_types[name])
        if name in self._relation_types:
            return dict(self._relation_types[name])
        return None
