"""Note data model for Aikioku."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Note(BaseModel):
    """Represents a knowledge note extracted from source material.

    Attributes:
        id: Auto-generated UUID4 primary key.
        title: Human-readable note title.
        content: Raw text content of the note.
        frontmatter: Parsed YAML frontmatter as a dictionary.
        links: List of linked note identifiers.
        path: File system or vault path to the note source.
        source_type: Origin of the note: "note" (user-created) or "memory" (MCP-generated).
        created: Timestamp of creation (auto-defaults to now).
        modified: Timestamp of last modification (auto-defaults to now).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(min_length=1)
    content: str = ""
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    links: list[str] = Field(default_factory=list)
    path: str = Field(min_length=1)
    source_type: str = "note"
    created: datetime = Field(default_factory=datetime.utcnow)
    modified: datetime = Field(default_factory=datetime.utcnow)


class NoteUpdate(BaseModel):
    """Partial model for updating an existing Note.

    All fields are optional; omitted fields preserve the existing values.
    """

    id: str | None = None
    title: str | None = None
    content: str | None = None
    frontmatter: dict[str, Any] | None = None
    links: list[str] | None = None
    path: str | None = None
    source_type: str | None = None
    created: datetime | None = None
    modified: datetime | None = None
