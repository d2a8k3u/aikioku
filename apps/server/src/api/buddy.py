"""Buddy API endpoints — AI buddy profile, greeting, and dynamic home cards.

The buddy is the user-facing AI companion (default name: "KIO"). Its
configuration lives in the ``app_settings`` table alongside other runtime
settings. The greeting and cards endpoints synthesize data from multiple
subsystems (memory, review, notes, conversations, serendipity) into the
shape the frontend design expects.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, field_validator

from src import runtime_config
from src.auth import UserInDB, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/buddy", tags=["buddy"])

# ------------------------------------------------------------------ models


class BuddyProfile(BaseModel):
    name: str = "KIO"
    tone: str = "warm"
    lm_provider: str = "remote"

    @field_validator("tone")
    @classmethod
    def _validate_tone(cls, v: str) -> str:
        if v not in ("warm", "focused", "playful"):
            raise ValueError("tone must be 'warm', 'focused', or 'playful'")
        return v

    @field_validator("lm_provider")
    @classmethod
    def _validate_lm_provider(cls, v: str) -> str:
        if v not in ("local", "remote", "router"):
            raise ValueError("lm_provider must be 'local', 'remote', or 'router'")
        return v


class BuddyGreeting(BaseModel):
    greeting: str
    buddy_line: str
    about_you: list[str]


class BuddyCard(BaseModel):
    kicker: str
    title: str
    body: str
    action: str
    rgb: str
    go: str


class BuddyCards(BaseModel):
    cards: list[BuddyCard]


# ------------------------------------------------------------------ helpers


def _load_profile() -> BuddyProfile:
    """Read buddy profile from app_settings, falling back to defaults."""
    return BuddyProfile(
        name=runtime_config.get_app_setting("buddy_name") or "KIO",
        tone=runtime_config.get_app_setting("buddy_tone") or "warm",
        lm_provider=runtime_config.get_app_setting("buddy_lm_provider") or "remote",
    )


def _save_profile(profile: BuddyProfile) -> None:
    """Persist buddy profile fields to app_settings."""
    if profile.name:
        runtime_config.set_app_setting("buddy_name", profile.name)
    if profile.tone:
        runtime_config.set_app_setting("buddy_tone", profile.tone)
    if profile.lm_provider:
        runtime_config.set_app_setting("buddy_lm_provider", profile.lm_provider)


def _greeting_for(name: str) -> str:
    """Time-of-day greeting."""
    h = datetime.utcnow().hour
    if 5 <= h < 11:
        return f"Good morning, {name}."
    if 11 <= h < 17:
        return f"Good afternoon, {name}."
    if 17 <= h < 22:
        return f"Good evening, {name}."
    return f"Still up, {name}?"


_TONE_LINES: dict[str, str] = {
    "warm": "I'm here. In the meantime I've tidied up your thoughts so you can think clearly today.",
    "focused": "Ready. Your notes and memories are synced. What shall we dive into today?",
    "playful": "So what shall we crack today? Your memories are primed and I've got a few ideas where to look.",
}


def _buddy_line(tone: str) -> str:
    """Tone-based buddy welcome line."""
    return _TONE_LINES.get(tone, _TONE_LINES["warm"])


def _about_you(username: str) -> list[str]:
    """Top memories about the user, formatted as natural-language facts."""
    try:
        from src.api.memory import _load_memories

        memories = _load_memories(entity=username)
        facts: list[str] = []
        for m in memories[:5]:
            subj = m.get("subject", "")
            pred = m.get("predicate", "")
            obj = m.get("object", "")
            if subj and pred and obj:
                facts.append(f"{subj} {pred} {obj}")
        return facts
    except Exception:
        logger.warning("Failed to load about_you facts.", exc_info=True)
        return []


# ------------------------------------------------------------------ endpoints


@router.get("/profile", response_model=BuddyProfile)
async def get_profile(_user: UserInDB = Depends(require_auth)) -> BuddyProfile:
    """Return the current buddy configuration."""
    return _load_profile()


@router.put("/profile", response_model=BuddyProfile)
async def update_profile(
    body: BuddyProfile, _user: UserInDB = Depends(require_auth)
) -> BuddyProfile:
    """Update buddy configuration and persist to the database."""
    _save_profile(body)
    return _load_profile()


@router.get("/greeting", response_model=BuddyGreeting)
async def get_greeting(
    user: UserInDB = Depends(require_auth),
) -> BuddyGreeting:
    """Return the time-of-day greeting, tone-based buddy line, and facts about the user."""
    profile = _load_profile()
    return BuddyGreeting(
        greeting=_greeting_for(user.username),
        buddy_line=_buddy_line(profile.tone),
        about_you=_about_you(user.username),
    )


@router.get("/cards", response_model=BuddyCards)
async def get_cards(
    request: Request,
    user: UserInDB = Depends(require_auth),
) -> BuddyCards:
    """Return up to 4 dynamic home-screen cards based on actual system state."""
    cards: list[BuddyCard] = []

    # Card 1: Continue conversation (if recent chat exists)
    try:
        from src.api.conversations import load_messages

        msgs = load_messages(user.username, limit=1)
        if msgs and msgs[0]["role"] == "user":
            snippet = msgs[0]["content"][:80]
            cards.append(
                BuddyCard(
                    kicker="CONTINUE",
                    title="In the conversation where you left off",
                    body=f"Last message: {snippet}...",
                    action="Open",
                    rgb="95,211,224",
                    go="talk",
                )
            )
    except Exception:
        logger.warning("Failed to build continue-conversation card.", exc_info=True)

    # Card 2: Serendipity connection (if surprise score is high enough)
    try:
        from src.augmentation.serendipity import SerendipityEngine

        graph = getattr(request.app.state, "knowledge_graph", None)
        if graph is not None:
            engine = SerendipityEngine(graph)
            entity_id, score = engine.random_surprise()
            if score > 0.3:
                entity = graph.get_entity(entity_id)
                entity_name = entity.name if entity else "new connection"
                cards.append(
                    BuddyCard(
                        kicker="I NOTICED",
                        title=f"Connection: {entity_name}",
                        body="I found an interesting link in your knowledge graph.",
                        action="Explore",
                        rgb="224,163,92",
                        go="thoughts",
                    )
                )
    except Exception:
        logger.warning("Failed to build serendipity card.", exc_info=True)

    # Card 3: Review due (or offer to generate cards)
    try:
        from src.api.review import _get_spaced_repetition

        sr = _get_spaced_repetition(request)
        stats = await sr.get_stats()
        due = stats.get("due", 0)
        if due > 0:
            cards.append(
                BuddyCard(
                    kicker="TO REVIEW",
                    title=f"{due} cards waiting for review",
                    body="I'll remind you of them exactly when you'd otherwise forget.",
                    action="Review",
                    rgb="224,163,92",
                    go="recall",
                )
            )
        else:
            cards.append(
                BuddyCard(
                    kicker="TO REVIEW",
                    title="No cards yet",
                    body="I'll make them from your notes and remind you at the right time.",
                    action="Create",
                    rgb="224,163,92",
                    go="recall",
                )
            )
    except Exception:
        logger.warning("Failed to build review card.", exc_info=True)

    # Card 4: Latest note
    try:
        store = getattr(request.app.state, "note_store", None)
        if store is not None:
            recent = store.list(skip=0, limit=1)
            if recent:
                note = recent[0]
                body_text = note.content[:120].replace("\n", " ") if note.content else ""
                cards.append(
                    BuddyCard(
                        kicker="LATEST THOUGHT",
                        title=note.title or "Untitled",
                        body=body_text,
                        action="Open",
                        rgb="95,211,224",
                        go="thoughts",
                    )
                )
    except Exception:
        logger.warning("Failed to build latest-note card.", exc_info=True)

    return BuddyCards(cards=cards[:4])
