"""Relation data model for Aikioku knowledge graph."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """Types of directed relationships between entities."""

    works_at = "works_at"
    created = "created"
    depends_on = "depends_on"
    related_to = "related_to"
    part_of = "part_of"
    located_in = "located_in"
    mentions = "mentions"
    follows = "follows"


class Relation(BaseModel):
    """Represents a directed relationship between two entities.

    Attributes:
        id: Auto-generated UUID4 primary key.
        source_entity_id: UUID of the source/origin entity.
        target_entity_id: UUID of the target/destination entity.
        type: Relationship type from RelationType enum.
        confidence: Relationship confidence score [0.0, 1.0].
        properties: Arbitrary key-value metadata about the relationship.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str
    target_entity_id: str
    type: RelationType
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    properties: dict[str, Any] = Field(default_factory=dict)
