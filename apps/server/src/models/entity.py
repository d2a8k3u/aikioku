"""Entity data model for Aikioku knowledge graph."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""

    Person = "Person"
    Place = "Place"
    Concept = "Concept"
    Project = "Project"
    Event = "Event"
    Organization = "Organization"
    Document = "Document"
    Task = "Task"


class Entity(BaseModel):
    """Represents a named entity extracted from notes.

    Attributes:
        id: Auto-generated UUID4 primary key.
        name: Human-readable entity name.
        type: Entity classification from EntityType enum.
        aliases: Alternative names for the entity.
        properties: Arbitrary key-value metadata.
        confidence: Extraction confidence score [0.0, 1.0].
        source_note_ids: UUIDs of notes this entity was derived from.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1)
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source_note_ids: list[str] = Field(default_factory=list)
