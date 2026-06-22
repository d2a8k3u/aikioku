"""Tests for SpacedRepetition system."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.models.card import Card, CardStatus, CardType
from src.models.note import Note


@pytest.fixture
def fixed_uuid() -> str:
    return "12345678-1234-5678-1234-567812345678"


@pytest.fixture
def fixed_uuid2() -> str:
    return "87654321-4321-8765-4321-876543210987"


@pytest.fixture
def fixed_datetime() -> datetime:
    return datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_note(fixed_uuid: str) -> Note:
    return Note(
        id=fixed_uuid,
        title="Python Basics",
        content="Python is a high-level programming language created by Guido van Rossum. "
        "It emphasizes code readability and uses significant indentation. "
        "Python supports multiple programming paradigms including procedural, "
        "object-oriented, and functional programming.",
        frontmatter={"tags": ["python", "programming"]},
        links=[],
        path="/notes/python-basics.md",
    )


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
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
def temp_db_path() -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_cards.db")


@pytest.fixture
def note_store(temp_db_path: str):
    from src.storage.note_store import NoteStore

    # Use a separate dir for notes, reuse temp dir
    notes_dir = str(Path(temp_db_path).parent / "notes")
    return NoteStore(notes_dir=notes_dir)


class TestGenerateCards:
    """Test card generation from note content using LLM."""

    @pytest.mark.asyncio
    async def test_generate_cards_returns_cards_from_note_content(
        self, note_store, mock_llm_provider, sample_note, temp_db_path
    ):
        from src.augmentation.spaced_repetition import SpacedRepetition

        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
        )
        cards = await sr.generate_cards(sample_note)

        assert len(cards) == 3
        assert all(isinstance(c, Card) for c in cards)
        assert all(c.note_id == sample_note.id for c in cards)
        # Verify LLM was called
        mock_llm_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_cards_creates_different_card_types(
        self, note_store, mock_llm_provider, sample_note, temp_db_path
    ):
        from src.augmentation.spaced_repetition import SpacedRepetition

        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
        )
        cards = await sr.generate_cards(sample_note)

        types = [c.type for c in cards]
        assert CardType.qa in types
        assert CardType.cloze in types
        assert CardType.connection in types

    @pytest.mark.asyncio
    async def test_generate_cards_from_fenced_json_persists(
        self, note_store, mock_llm_provider, sample_note, temp_db_path
    ):
        import sqlite3

        from src.augmentation.spaced_repetition import SpacedRepetition

        # Real LLMs wrap JSON in ```json fences.
        mock_llm_provider.complete.return_value = (
            "Here are the cards:\n```json\n"
            '[{"type": "qa", "front": "Who created Python?", "back": "Guido van Rossum"}]'
            "\n```"
        )

        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
        )
        cards = await sr.generate_cards(sample_note)

        assert len(cards) == 1
        assert cards[0].front == "Who created Python?"

        # Persisted to SQLite.
        conn = sqlite3.connect(temp_db_path)
        count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        conn.close()
        assert count == 1

    @pytest.mark.asyncio
    async def test_generate_cards_garbage_raises_parse_error(
        self, note_store, mock_llm_provider, sample_note, temp_db_path
    ):
        from src.augmentation.spaced_repetition import SpacedRepetition
        from src.llm.json_parse import LLMOutputParseError

        mock_llm_provider.complete.return_value = "I cannot generate cards for this."

        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
        )

        with pytest.raises(LLMOutputParseError):
            await sr.generate_cards(sample_note)


class TestGetDueCards:
    """Test retrieving due cards."""

    @pytest.mark.asyncio
    async def test_get_due_cards_returns_only_due_cards(
        self, note_store, mock_llm_provider, sample_note, temp_db_path, fixed_datetime
    ):
        from src.augmentation.spaced_repetition import SpacedRepetition

        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
        )

        # Create cards: one due, one not due
        past = datetime(2025, 1, 1, tzinfo=timezone.utc)
        future = datetime(2099, 12, 31, tzinfo=timezone.utc)

        due_card = Card(
            note_id=sample_note.id,
            type=CardType.qa,
            front="Due question",
            back="Due answer",
            next_review=past,
            status=CardStatus.review,
        )
        not_due_card = Card(
            note_id=sample_note.id,
            type=CardType.qa,
            front="Future question",
            back="Future answer",
            next_review=future,
            status=CardStatus.review,
        )

        # Manually insert cards into the database
        due_card_row = sr._card_to_row(due_card)
        not_due_card_row = sr._card_to_row(not_due_card)

        import sqlite3

        conn = sqlite3.connect(temp_db_path)
        conn.execute(
            "INSERT INTO cards VALUES (:id, :note_id, :type, :front, :back, "
            ":ease_factor, :interval, :repetitions, :next_review, :status)",
            due_card_row,
        )
        conn.execute(
            "INSERT INTO cards VALUES (:id, :note_id, :type, :front, :back, "
            ":ease_factor, :interval, :repetitions, :next_review, :status)",
            not_due_card_row,
        )
        conn.commit()
        conn.close()

        due_cards = await sr.get_due_cards(limit=20)
        assert len(due_cards) == 1
        assert due_cards[0].front == "Due question"


class TestReviewCard:
    """Test SM-2 review algorithm."""

    @pytest.mark.asyncio
    async def test_review_card_rating_1_resets_interval(
        self, note_store, mock_llm_provider, sample_note, temp_db_path
    ):
        from src.augmentation.spaced_repetition import SpacedRepetition

        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
        )

        # Create a card with some interval/repetitions
        card = Card(
            note_id=sample_note.id,
            type=CardType.qa,
            front="Q",
            back="A",
            ease_factor=2.5,
            interval=10,
            repetitions=5,
            next_review=datetime(2025, 1, 1, tzinfo=timezone.utc),
            status=CardStatus.review,
        )
        row = sr._card_to_row(card)
        import sqlite3

        conn = sqlite3.connect(temp_db_path)
        conn.execute(
            "INSERT INTO cards VALUES (:id, :note_id, :type, :front, :back, "
            ":ease_factor, :interval, :repetitions, :next_review, :status)",
            row,
        )
        conn.commit()
        conn.close()

        reviewed = await sr.review_card(card.id, rating=1)

        assert reviewed.interval == 1
        assert reviewed.repetitions == 0

    @pytest.mark.asyncio
    async def test_review_card_rating_3_increases_interval_by_ease_factor(
        self, note_store, mock_llm_provider, sample_note, temp_db_path
    ):
        from src.augmentation.spaced_repetition import SpacedRepetition

        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
        )

        card = Card(
            note_id=sample_note.id,
            type=CardType.qa,
            front="Q",
            back="A",
            ease_factor=2.5,
            interval=4,
            repetitions=2,
            next_review=datetime(2025, 1, 1, tzinfo=timezone.utc),
            status=CardStatus.review,
        )
        row = sr._card_to_row(card)
        import sqlite3

        conn = sqlite3.connect(temp_db_path)
        conn.execute(
            "INSERT INTO cards VALUES (:id, :note_id, :type, :front, :back, "
            ":ease_factor, :interval, :repetitions, :next_review, :status)",
            row,
        )
        conn.commit()
        conn.close()

        reviewed = await sr.review_card(card.id, rating=3)

        # interval *= ease_factor => 4 * 2.5 = 10
        assert reviewed.interval == 10
        assert reviewed.ease_factor == 2.5  # unchanged
        assert reviewed.repetitions == 3

    @pytest.mark.asyncio
    async def test_review_card_rating_4_increases_interval_more(
        self, note_store, mock_llm_provider, sample_note, temp_db_path
    ):
        from src.augmentation.spaced_repetition import SpacedRepetition

        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
        )

        card = Card(
            note_id=sample_note.id,
            type=CardType.qa,
            front="Q",
            back="A",
            ease_factor=2.5,
            interval=4,
            repetitions=2,
            next_review=datetime(2025, 1, 1, tzinfo=timezone.utc),
            status=CardStatus.review,
        )
        row = sr._card_to_row(card)
        import sqlite3

        conn = sqlite3.connect(temp_db_path)
        conn.execute(
            "INSERT INTO cards VALUES (:id, :note_id, :type, :front, :back, "
            ":ease_factor, :interval, :repetitions, :next_review, :status)",
            row,
        )
        conn.commit()
        conn.close()

        reviewed = await sr.review_card(card.id, rating=4)

        # interval *= ease_factor * 1.3 => 4 * 2.5 * 1.3 = 13
        assert reviewed.interval == 13
        assert reviewed.ease_factor == 2.65  # 2.5 + 0.15
        assert reviewed.repetitions == 3

    @pytest.mark.asyncio
    async def test_ease_factor_never_drops_below_1_3(
        self, note_store, mock_llm_provider, sample_note, temp_db_path
    ):
        from src.augmentation.spaced_repetition import SpacedRepetition

        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
        )

        # Start with ease_factor just above minimum
        card = Card(
            note_id=sample_note.id,
            type=CardType.qa,
            front="Q",
            back="A",
            ease_factor=1.35,  # just 0.05 above minimum
            interval=5,
            repetitions=3,
            next_review=datetime(2025, 1, 1, tzinfo=timezone.utc),
            status=CardStatus.review,
        )
        row = sr._card_to_row(card)
        import sqlite3

        conn = sqlite3.connect(temp_db_path)
        conn.execute(
            "INSERT INTO cards VALUES (:id, :note_id, :type, :front, :back, "
            ":ease_factor, :interval, :repetitions, :next_review, :status)",
            row,
        )
        conn.commit()
        conn.close()

        # Rating 2 (hard) decreases ease_factor by 0.15
        reviewed = await sr.review_card(card.id, rating=2)

        # 1.35 - 0.15 = 1.20, but minimum is 1.3
        assert reviewed.ease_factor == 1.3


class TestGetStats:
    """Test statistics retrieval."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_correct_counts(
        self, note_store, mock_llm_provider, sample_note, temp_db_path
    ):
        from src.augmentation.spaced_repetition import SpacedRepetition

        sr = SpacedRepetition(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
        )

        past = datetime(2025, 1, 1, tzinfo=timezone.utc)
        future = datetime(2099, 12, 31, tzinfo=timezone.utc)

        # Create cards with different statuses
        cards_data = [
            ("new_card", CardStatus.new, past),  # due, new
            ("learning_card", CardStatus.learning, past),  # due, learning
            ("review_card_due", CardStatus.review, past),  # due, review
            ("review_card_future", CardStatus.review, future),  # not due, review
            ("suspended_card", CardStatus.suspended, past),  # due, suspended
        ]

        import sqlite3

        conn = sqlite3.connect(temp_db_path)
        for front, status, review_time in cards_data:
            card = Card(
                note_id=sample_note.id,
                type=CardType.qa,
                front=front,
                back="A",
                next_review=review_time,
                status=status,
            )
            row = sr._card_to_row(card)
            conn.execute(
                "INSERT INTO cards VALUES (:id, :note_id, :type, :front, :back, "
                ":ease_factor, :interval, :repetitions, :next_review, :status)",
                row,
            )
        conn.commit()
        conn.close()

        stats = await sr.get_stats()

        assert stats["total"] == 5
        assert stats["due"] == 4  # all except the future one
        assert stats["new"] == 1
        assert stats["learning"] == 1
        assert stats["review"] == 2  # both review cards (due + not due)
