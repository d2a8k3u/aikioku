"""Conversation message model for persisted chat history.

A single continuous chat thread per user (keyed by ``user_id``). Messages are
appended by the chat endpoints and read back for the UI. Matches the modelling
convention of :mod:`src.models.memory` (uuid4 id, ``datetime`` timestamps).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """A single persisted chat message.

    Attributes:
        id: Auto-generated UUID4 primary key.
        user_id: Owner of the message (``"anonymous"`` in local-trust mode).
        role: ``"user"`` or ``"assistant"``.
        content: The message text.
        citations: Source citations attached to an assistant message (note_id,
            title, snippet, ...). Empty for user messages.
        sub_questions: Multi-hop reasoning steps for an assistant message.
        created: Creation timestamp (auto-defaults to now).
        in_progress: When True, this assistant message is a placeholder for a
            turn that is still being generated. The UI shows a working indicator
            and reloads display the placeholder instead of a missing reply.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    content: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    created: datetime = Field(default_factory=datetime.utcnow)
    in_progress: bool = Field(default=False)
