"""Integration tests for the review pipeline: card generation, SM-2 review, due cards, stats.

Uses real SQLite in a temp directory and a mocked LLMProvider for card generation.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.augmentation.spaced_repetition import SpacedRepetition
from src.models.card import Card, CardStatus, CardType
from src.models.note import Note
from src.storage.note_store import NoteStore


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for both notes and the cards database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def note_store(temp_dir: Path) -> NoteStore:
    """Create a NoteStore backed by a temporary directory."""
    return NoteStore(notes_dir=str(temp_dir / "notes"))


@pytest.fixture
def db_path(temp_dir: Path) -> str:
    """Return a path for the SQLite cards database."""
    return str(temp_dir / "cards.db")


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    """Return a mocked LLMProvider that returns three cards."""
    provider = AsyncMock()
    provider.complete.return_value = (
        '[{"type": "qa", "front": "Who created Python?", "back": "Guido van Rossum"}, '
        '{"type": "cloze", "front": "Python was created by {{Guido van Rossum}}.", '
        '"back": "Python was created by Guido van Rossum."}, '
        '{"type": "connection", "front": "Python -> Readability", '
        '"back": "Python emphasizes code readability"}]'
    )
    return provider


@pytest.fixture
def sample_note() -> Note:
    """Return a sample note for card generation."""
    return Note(
        id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        title="Python Basics",
        content=(
            "Python is a high-level programming language created by Guido van Rossum. "
            "It emphasizes code readability and uses significant indentation."
        ),
        frontmatter={"tags": ["python", "programming"]},
        links=[],
        path="/notes/python-basics.md",
    )


@pytest.fixture
def spaced_repetition(note_store, mock_llm_provider, db_path) -> SpacedRepetition:
    """Create a SpacedRepetition instance with mocked LLM and temp SQLite."""
    return SpacedRepetition(
        note_store=note_store,
        llm_provider=mock_llm_provider,
        db_path=db_path,
    )


def _insert_card(db_path: str, card: Card, sr: SpacedRepetition) -> None:
    """Helper to insert a card row directly into the database."""
    row = sr._card_to_row(card)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO cards VALUES (:id, :note_id, :type, :front, :back, "
        ":ease_factor, :interval, :repetitions, :next_review, :status)",
        row,
    )
    conn.commit()
    conn.close()


class TestGenerateCardsFromNote:
    """Test card generation from a note via the full pipeline."""

    async def test_generate_cards_from_note(
        self,
        note_store: NoteStore,
        spaced_repetition: SpacedRepetition,
        sample_note: Note,
        mock_llm_provider: AsyncMock,
    ):
        # Create the note in the store first
        note_store.create(sample_note)

        # Run card generation
        cards = await spaced_repetition.generate_cards(sample_note)

        # Verify cards were created
        assert len(cards) == 3
        assert all(isinstance(c, Card) for c in cards)
        assert all(c.note_id == sample_note.id for c in cards)

        # Verify card types
        types = {c.type for c in cards}
        assert CardType.qa in types
        assert CardType.cloze in types
        assert CardType.connection in types

        # Verify LLM was called with the note content
        mock_llm_provider.complete.assert_called_once()
        call_kwargs = mock_llm_provider.complete.call_args
        assert sample_note.title in call_kwargs.kwargs.get(
            "prompt", call_kwargs[1].get("prompt", "")
        )
        assert sample_note.content in call_kwargs.kwargs.get(
            "prompt", call_kwargs[1].get("prompt", "")
        )

        # Verify all cards have status 'new' and interval 0
        for card in cards:
            assert card.status == CardStatus.new
            assert card.interval == 0
            assert card.repetitions == 0
            assert card.ease_factor == 2.5


class TestReviewCardSm2:
    """Test SM-2 review with rating=3 (good)."""

    async def test_review_card_sm2(
        self,
        spaced_repetition: SpacedRepetition,
        sample_note: Note,
        db_path: str,
    ):
        # Create a card with interval=4, ease_factor=2.5, repetitions=2
        card = Card(
            note_id=sample_note.id,
            type=CardType.qa,
            front="What is Python?",
            back="A programming language",
            ease_factor=2.5,
            interval=4,
            repetitions=2,
            next_review=datetime(2025, 1, 1, tzinfo=timezone.utc),
            status=CardStatus.review,
        )
        _insert_card(db_path, card, spaced_repetition)

        # Review with rating=3 (good)
        reviewed = await spaced_repetition.review_card(card.id, rating=3)

        # interval = int(4 * 2.5) = 10
        assert reviewed.interval == 10
        # ease_factor unchanged for 'good'
        assert reviewed.ease_factor == 2.5
        # repetitions incremented
        assert reviewed.repetitions == 3
        # status transitions to 'review' since repetitions >= 2 before increment (now 3)
        assert reviewed.status == CardStatus.review
        # next_review should be approximately now + 10 days
        expected_review = datetime.utcnow() + timedelta(days=10)
        # Allow 5-second tolerance for test execution time
        assert abs((reviewed.next_review - expected_review).total_seconds()) < 5


class TestReviewCardAgain:
    """Test SM-2 review with rating=1 (again)."""

    async def test_review_card_again(
        self,
        spaced_repetition: SpacedRepetition,
        sample_note: Note,
        db_path: str,
    ):
        # Create a card with some progress
        card = Card(
            note_id=sample_note.id,
            type=CardType.qa,
            front="What is Python?",
            back="A programming language",
            ease_factor=2.5,
            interval=10,
            repetitions=5,
            next_review=datetime(2025, 1, 1, tzinfo=timezone.utc),
            status=CardStatus.review,
        )
        _insert_card(db_path, card, spaced_repetition)

        # Review with rating=1 (again)
        reviewed = await spaced_repetition.review_card(card.id, rating=1)

        # Interval reset to 1
        assert reviewed.interval == 1
        # Repetitions reset to 0
        assert reviewed.repetitions == 0
        # Status set to learning
        assert reviewed.status == CardStatus.learning
        # next_review should be approximately now + 1 day
        expected_review = datetime.utcnow() + timedelta(days=1)
        assert abs((reviewed.next_review - expected_review).total_seconds()) < 5


class TestDueCards:
    """Test retrieving only due cards."""

    async def test_due_cards(
        self,
        spaced_repetition: SpacedRepetition,
        sample_note: Note,
        db_path: str,
    ):
        now = datetime.utcnow()
        past = now - timedelta(days=2)
        future = now + timedelta(days=7)

        # Create 3 cards: 2 due, 1 not due
        due_card_1 = Card(
            note_id=sample_note.id,
            type=CardType.qa,
            front="Due question 1",
            back="Answer 1",
            interval=3,
            repetitions=2,
            next_review=past,
            status=CardStatus.review,
        )
        due_card_2 = Card(
            note_id=sample_note.id,
            type=CardType.cloze,
            front="Due question 2",
            back="Answer 2",
            interval=1,
            repetitions=1,
            next_review=past,
            status=CardStatus.learning,
        )
        future_card = Card(
            note_id=sample_note.id,
            type=CardType.qa,
            front="Future question",
            back="Future answer",
            interval=10,
            repetitions=5,
            next_review=future,
            status=CardStatus.review,
        )

        _insert_card(db_path, due_card_1, spaced_repetition)
        _insert_card(db_path, due_card_2, spaced_repetition)
        _insert_card(db_path, future_card, spaced_repetition)

        due_cards = await spaced_repetition.get_due_cards(limit=20)

        # Only the 2 past-due cards should be returned
        assert len(due_cards) == 2
        fronts = {c.front for c in due_cards}
        assert "Due question 1" in fronts
        assert "Due question 2" in fronts
        assert "Future question" not in fronts


class TestCardStats:
    """Test statistics retrieval across different card states."""

    async def test_card_stats(
        self,
        spaced_repetition: SpacedRepetition,
        sample_note: Note,
        db_path: str,
    ):
        now = datetime.utcnow()
        past = now - timedelta(days=1)
        future = now + timedelta(days=30)

        cards_data = [
            # (front, status, next_review)
            ("new_due", CardStatus.new, past),
            ("new_future", CardStatus.new, future),
            ("learning_due", CardStatus.learning, past),
            ("review_due", CardStatus.review, past),
            ("review_future", CardStatus.review, future),
            ("suspended_due", CardStatus.suspended, past),
        ]

        for front, status, review_time in cards_data:
            card = Card(
                note_id=sample_note.id,
                type=CardType.qa,
                front=front,
                back="Answer",
                next_review=review_time,
                status=status,
            )
            _insert_card(db_path, card, spaced_repetition)

        stats = await spaced_repetition.get_stats()

        assert stats["total"] == 6
        assert stats["due"] == 4  # all except the 2 future cards
        assert stats["new"] == 2
        assert stats["learning"] == 1
        assert stats["review"] == 2
