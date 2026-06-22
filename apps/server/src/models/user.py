"""User data model for Aikioku."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class User(BaseModel):
    """Represents an application user with preferences.

    Attributes:
        id: Auto-generated UUID4 primary key.
        name: Human-readable display name.
        email: User email address.
        preferences: Arbitrary key-value user preferences.
        created_at: Timestamp of creation (auto-defaults to now).
        updated_at: Timestamp of last update (auto-defaults to now).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1)
    email: str = Field(min_length=1)
    preferences: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
