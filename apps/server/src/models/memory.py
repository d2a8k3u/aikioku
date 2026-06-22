"""Memory data model for Aikioku knowledge store."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MemoryTier(str, Enum):
    """Vitality tiers for memory decay tracking."""

    hot = "hot"
    warm = "warm"
    cold = "cold"


class Memory(BaseModel):
    """Represents a subject-predicate-object memory triple.

    Attributes:
        id: Auto-generated UUID4 primary key.
        subject: The subject of the memory statement.
        predicate: The relationship verb.
        object: The object of the memory statement.
        confidence: Confidence score [0.0, 1.0].
        source: Identifier of the source note or extraction run.
        created: Timestamp of creation (auto-defaults to now).
        modified: Timestamp of last modification (auto-defaults to now).
        vitality_score: Decay tracking score for spaced repetition.
        tier: Hot/warm/cold classification for review scheduling.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source: str = Field(min_length=1)
    created: datetime = Field(default_factory=datetime.utcnow)
    modified: datetime = Field(default_factory=datetime.utcnow)
    vitality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    tier: MemoryTier = MemoryTier.hot
    properties: dict = Field(default_factory=dict)
