"""Review API endpoints: due cards, card generation, card review, stats."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator

from src.config import settings
from src.llm.json_parse import LLMOutputParseError
from src.api.websocket import get_broadcaster
from src.models.card import Card
from src.storage.note_store import NoteStore

# Module-level import for test mocking (lazy usage in _get_spaced_repetition)
from src.augmentation.spaced_repetition import SpacedRepetition  # noqa: F401

router = APIRouter(prefix="/api/review", tags=["review"])


class ReviewRequest(BaseModel):
    rating: int

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: int) -> int:
        if v not in (1, 2, 3, 4):
            raise ValueError("rating must be 1, 2, 3, or 4")
        return v


class GenerateCardsRequest(BaseModel):
    note_id: str


def _get_note_store(request: Request) -> NoteStore:
    """Get NoteStore from app state or create a new one."""
    store = getattr(request.app.state, "note_store", None)
    if store is None:
        store = NoteStore(settings.notes_dir)
        request.app.state.note_store = store
    return store


def _get_spaced_repetition(request: Request) -> SpacedRepetition:
    """Get or create a SpacedRepetition instance from app state.

    Uses the app's configured LLM provider (app.state.llm_provider), which is the
    router wrapping Ollama Cloud chat + host embeddings. A bare OllamaProvider()
    would point at localhost:11434, unreachable inside the backend container.
    """
    sr: SpacedRepetition | None = getattr(request.app.state, "spaced_repetition", None)
    if sr is None:
        llm = getattr(request.app.state, "llm_provider", None)
        if llm is None:
            raise HTTPException(
                status_code=503,
                detail="LLM provider not configured.",
            )
        note_store = _get_note_store(request)
        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=llm,
            db_path=settings.sqlite_db_path,
        )
        request.app.state.spaced_repetition = sr
    return sr


def _card_to_dict(card: Card) -> dict[str, Any]:
    """Serialize a Card to a JSON-compatible dictionary."""
    return {
        "id": card.id,
        "note_id": card.note_id,
        "type": card.type.value if hasattr(card.type, "value") else card.type,
        "front": card.front,
        "back": card.back,
        "ease_factor": card.ease_factor,
        "interval": card.interval,
        "repetitions": card.repetitions,
        "next_review": card.next_review.isoformat(),
        "status": card.status.value if hasattr(card.status, "value") else card.status,
    }


@router.get("/due")
async def get_due_cards(request: Request) -> list[dict[str, Any]]:
    """Return cards that are due for review (next_review <= now)."""
    sr = _get_spaced_repetition(request)
    cards = await sr.get_due_cards()
    return [_card_to_dict(c) for c in cards]


@router.post("/cards")
async def generate_cards(request: Request, body: GenerateCardsRequest) -> list[dict[str, Any]]:
    """Generate flashcards from a note and return the created cards."""
    note_store = _get_note_store(request)
    sr = _get_spaced_repetition(request)

    note = note_store.get(body.note_id)
    if note is None:
        raise HTTPException(status_code=404, detail=f"Note not found: {body.note_id}")

    try:
        cards = await sr.generate_cards(note)
    except LLMOutputParseError:
        raise HTTPException(
            status_code=502,
            detail=(
                "Card generation failed: the language model returned invalid output. Please retry."
            ),
        )
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        raise HTTPException(status_code=503, detail="Language model unavailable.")

    return [_card_to_dict(c) for c in cards]


@router.post("/cards/{card_id}/review")
async def review_card(request: Request, card_id: str, body: ReviewRequest) -> dict[str, Any]:
    """Review a card with a rating (1-4) and return the updated card."""
    sr = _get_spaced_repetition(request)
    card = await sr.review_card(card_id, body.rating)
    # Broadcast review state change
    broadcaster = get_broadcaster()
    if broadcaster:
        import asyncio

        stats = await sr.get_stats()
        asyncio.ensure_future(
            broadcaster.broadcast(
                "review.due_changed",
                {
                    "due_count": stats.get("due", 0),
                    "total": stats.get("total", 0),
                },
            )
        )
    return _card_to_dict(card)


@router.get("/stats")
async def get_stats(request: Request) -> dict[str, int]:
    """Return card collection statistics."""
    sr = _get_spaced_repetition(request)
    stats = await sr.get_stats()
    # Add suspended count via SpacedRepetition abstraction
    stats["suspended"] = await sr.get_suspended_count()
    return stats
