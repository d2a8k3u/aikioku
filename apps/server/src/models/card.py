"""Card data model for Aikioku spaced repetition system."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class CardType(str, Enum):
    """Types of flashcards."""

    cloze = "cloze"
    qa = "qa"
    connection = "connection"


class CardStatus(str, Enum):
    """Learning status of a flashcard."""

    new = "new"
    learning = "learning"
    review = "review"
    suspended = "suspended"


class Card(BaseModel):
    """Represents a spaced repetition flashcard linked to a note.

    Uses a SM-2 inspired scheduling algorithm with ease factor,
    interval, and repetition count.

    Attributes:
        id: Auto-generated UUID4 primary key.
        note_id: UUID of the source note.
        type: Card type (cloze, qa, connection).
        front: Question or prompt text.
        back: Answer or response text.
        ease_factor: SM-2 ease factor (minimum 1.3).
        interval: Days until next review (minimum 0).
        repetitions: Successful review count (minimum 0).
        next_review: Scheduled review datetime.
        status: Current learning status.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    note_id: str
    type: CardType
    front: str = Field(min_length=1)
    back: str = ""
    ease_factor: float = Field(ge=1.3, default=2.5)
    interval: int = Field(ge=0, default=0)
    repetitions: int = Field(ge=0, default=0)
    next_review: datetime
    status: CardStatus
