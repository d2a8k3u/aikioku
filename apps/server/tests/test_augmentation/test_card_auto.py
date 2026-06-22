"""Tests for CardAutoGenerator — automatic review card generation pipeline."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.events import Event, EventBus
from src.models.card import CardStatus
from src.models.note import Note


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_uuid() -> str:
    return "12345678-1234-5678-1234-567812345678"


@pytest.fixture
def fixed_uuid2() -> str:
    return "87654321-4321-8765-4321-876543210987"


@pytest.fixture
def short_note(fixed_uuid: str) -> Note:
    """A note with <200 chars of content — should be ineligible."""
    return Note(
        id=fixed_uuid,
        title="Short Note",
        content="This is a very short note.",
        frontmatter={"tags": []},
        links=[],
        path="/notes/short.md",
    )


@pytest.fixture
def long_note(fixed_uuid: str) -> Note:
    """A note with >200 chars of content — should be eligible."""
    return Note(
        id=fixed_uuid,
        title="Python Basics",
        content=(
            "Python is a high-level programming language created by Guido van Rossum. "
            "It emphasizes code readability and uses significant indentation. "
            "Python supports multiple programming paradigms including procedural, "
            "object-oriented, and functional programming. "
            "Python's design philosophy emphasizes code readability with its notable "
            "use of significant whitespace. Its language constructs and object-oriented "
            "approach aim to help programmers write clear, logical code for small and "
            "large-scale projects."
        ),
        frontmatter={"tags": ["python", "programming"]},
        links=[],
        path="/notes/python-basics.md",
    )


@pytest.fixture
def long_note_v2(fixed_uuid: str) -> Note:
    """A modified version of long_note with different content."""
    return Note(
        id=fixed_uuid,
        title="Python Basics (Revised)",
        content=(
            "Python is a versatile high-level programming language created by Guido van Rossum "
            "and first released in 1991. It emphasizes code readability and uses significant "
            "indentation. Python supports multiple programming paradigms including procedural, "
            "object-oriented, and functional programming. It is dynamically typed and "
            "garbage-collected. Python is often described as a 'batteries included' language "
            "due to its comprehensive standard library. It is widely used in web development, "
            "data science, artificial intelligence, and automation."
        ),
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
        yield str(Path(tmpdir) / "test_aikioku.db")


@pytest.fixture
def note_store(temp_db_path: str):
    from src.storage.note_store import NoteStore

    notes_dir = str(Path(temp_db_path).parent / "notes")
    return NoteStore(notes_dir=notes_dir)


@pytest.fixture
def event_bus(temp_db_path: str) -> EventBus:
    return EventBus(temp_db_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _count_cards_for_note(db_path: str, note_id: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    count = conn.execute("SELECT COUNT(*) FROM cards WHERE note_id = ?", (note_id,)).fetchone()[0]
    conn.close()
    return count


def _count_suspended_cards_for_note(db_path: str, note_id: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    count = conn.execute(
        "SELECT COUNT(*) FROM cards WHERE note_id = ? AND status = ?",
        (note_id, CardStatus.suspended.value),
    ).fetchone()[0]
    conn.close()
    return count


def _get_generation_meta(db_path: str, note_id: str) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM card_generation_meta WHERE note_id = ?", (note_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _get_daily_count(db_path: str, date_str: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT count FROM card_generation_log WHERE date = ?", (date_str,)
    ).fetchone()
    conn.close()
    return row["count"] if row else 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEligibility:
    """Tests for content-length eligibility check."""

    @pytest.mark.asyncio
    async def test_short_note_not_eligible(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, short_note
    ):
        """A note with <200 chars of content should not trigger card generation."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
        )

        # Save the short note so it can be looked up
        note_store.create(short_note)

        event = Event("note.created", {"note_id": short_note.id})
        await gen.handle_note_event(event)

        # No cards should have been generated
        assert _count_cards_for_note(temp_db_path, short_note.id) == 0
        # LLM should not have been called
        mock_llm_provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_note_is_eligible(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note
    ):
        """A note with >200 chars of content should trigger card generation."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
        )

        note_store.create(long_note)

        event = Event("note.created", {"note_id": long_note.id})
        await gen.handle_note_event(event)

        # Cards should have been generated
        assert _count_cards_for_note(temp_db_path, long_note.id) > 0
        mock_llm_provider.complete.assert_called_once()


class TestContentHash:
    """Tests for the content_hash static method."""

    def test_content_hash_stable(self, long_note):
        """Same content produces the same hash."""
        from src.augmentation.card_auto import CardAutoGenerator

        h1 = CardAutoGenerator.content_hash(long_note)
        h2 = CardAutoGenerator.content_hash(long_note)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_content_hash_different(self, long_note, long_note_v2):
        """Different content produces different hashes."""
        from src.augmentation.card_auto import CardAutoGenerator

        h1 = CardAutoGenerator.content_hash(long_note)
        h2 = CardAutoGenerator.content_hash(long_note_v2)
        assert h1 != h2


class TestContentHashObsolescence:
    """Tests for content-hash based skip/suspend logic."""

    @pytest.mark.asyncio
    async def test_same_content_hash_skips_generation(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note
    ):
        """If content hasn't changed since last generation, skip."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
        )

        note_store.create(long_note)

        # First event: should generate cards
        event1 = Event("note.created", {"note_id": long_note.id})
        await gen.handle_note_event(event1)
        first_count = _count_cards_for_note(temp_db_path, long_note.id)
        assert first_count > 0
        assert mock_llm_provider.complete.call_count == 1

        # Second event with same content: should skip
        event2 = Event("note.updated", {"note_id": long_note.id})
        await gen.handle_note_event(event2)

        # Card count should be unchanged (no new cards, no suspension)
        assert _count_cards_for_note(temp_db_path, long_note.id) == first_count
        # LLM should NOT have been called again
        assert mock_llm_provider.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_content_changed_suspends_old_and_generates_new(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note, long_note_v2
    ):
        """When content changes, old cards are suspended and new ones generated."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
        )

        note_store.create(long_note)

        # First event: generate cards for original content
        event1 = Event("note.created", {"note_id": long_note.id})
        await gen.handle_note_event(event1)
        first_count = _count_cards_for_note(temp_db_path, long_note.id)
        assert first_count > 0
        assert mock_llm_provider.complete.call_count == 1

        # Update the note with different content
        note_store.update(long_note_v2)

        # Second event with changed content
        event2 = Event("note.updated", {"note_id": long_note_v2.id})
        await gen.handle_note_event(event2)

        # Old cards should be suspended
        suspended = _count_suspended_cards_for_note(temp_db_path, long_note.id)
        assert suspended == first_count

        # New cards should be generated
        total = _count_cards_for_note(temp_db_path, long_note.id)
        assert total > first_count  # old suspended + new active

        # LLM should have been called twice
        assert mock_llm_provider.complete.call_count == 2

        # Generation meta should reflect the new hash
        meta = _get_generation_meta(temp_db_path, long_note.id)
        assert meta is not None
        assert meta["content_hash"] == _content_hash(long_note_v2.content)

    def test_suspend_old_cards(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note
    ):
        """_suspend_cards_for_note sets status to 'suspended' for non-suspended cards."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
        )

        # Insert some cards directly into the DB for this note
        conn = sqlite3.connect(temp_db_path)
        now_iso = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO cards (id, note_id, type, front, back, ease_factor, interval, repetitions, next_review, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("card-1", long_note.id, "qa", "Q1", "A1", 2.5, 0, 0, now_iso, "new"),
        )
        conn.execute(
            "INSERT INTO cards (id, note_id, type, front, back, ease_factor, interval, repetitions, next_review, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("card-2", long_note.id, "cloze", "Q2", "A2", 2.5, 0, 0, now_iso, "review"),
        )
        conn.execute(
            "INSERT INTO cards (id, note_id, type, front, back, ease_factor, interval, repetitions, next_review, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("card-3", long_note.id, "connection", "Q3", "A3", 2.5, 0, 0, now_iso, "suspended"),
        )
        conn.commit()
        conn.close()

        # Suspend cards for this note
        count = gen._suspend_cards_for_note(long_note.id)

        # Should have suspended 2 cards (the 'new' and 'review' ones, not the already-suspended one)
        assert count == 2

        # Verify: card-1 and card-2 should now be suspended, card-3 was already suspended
        suspended_count = _count_suspended_cards_for_note(temp_db_path, long_note.id)
        assert suspended_count == 3  # all 3 are now suspended


class TestDailyBalancer:
    """Tests for daily card generation quota."""

    @pytest.mark.asyncio
    async def test_daily_quota_not_exceeded(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note
    ):
        """With 0 generations today, check_daily_quota returns True."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
            max_per_day=20,
        )

        # No generations recorded yet — quota should be available
        assert await gen.check_daily_quota() is True

    @pytest.mark.asyncio
    async def test_daily_quota_exceeded(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note
    ):
        """With max_per_day generations today, check_daily_quota returns False."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
            max_per_day=1,
        )

        # Record one generation to fill the quota
        await gen.record_generation(long_note.id, _content_hash(long_note.content))

        # Quota should now be exceeded
        assert await gen.check_daily_quota() is False

    @pytest.mark.asyncio
    async def test_record_generation_increments(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note
    ):
        """record_generation increments the daily count."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
            max_per_day=20,
        )

        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Before: count should be 0
        assert _get_daily_count(temp_db_path, today) == 0

        # Record a generation
        await gen.record_generation(long_note.id, _content_hash(long_note.content))

        # After: count should be 1
        assert _get_daily_count(temp_db_path, today) == 1

        # Record another
        await gen.record_generation(long_note.id, _content_hash(long_note.content))

        # After: count should be 2
        assert _get_daily_count(temp_db_path, today) == 2

    @pytest.mark.asyncio
    async def test_daily_quota_exceeded_skips_generation(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note
    ):
        """When daily quota is exceeded, skip card generation."""
        from src.augmentation.card_auto import CardAutoGenerator

        # Set max_per_day to 1 so we can easily exceed it
        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
            max_per_day=1,
        )

        note_store.create(long_note)

        # First event: should generate (within quota)
        event1 = Event("note.created", {"note_id": long_note.id})
        await gen.handle_note_event(event1)
        assert mock_llm_provider.complete.call_count == 1

        # Create a second note with different content
        note2 = Note(
            id="22222222-2222-2222-2222-222222222222",
            title="Another Long Note",
            content=(
                "JavaScript is a high-level, often just-in-time compiled language that "
                "conforms to the ECMAScript specification. JavaScript has dynamic typing, "
                "prototype-based object-orientation, and first-class functions. It is "
                "multi-paradigm, supporting event-driven, functional, and imperative "
                "programming styles. It has application programming interfaces for working "
                "with text, dates, regular expressions, standard data structures, and the "
                "Document Object Model."
            ),
            frontmatter={"tags": ["javascript"]},
            links=[],
            path="/notes/javascript.md",
        )
        note_store.create(note2)

        # Second event: should skip (quota exceeded)
        event2 = Event("note.created", {"note_id": note2.id})
        await gen.handle_note_event(event2)

        # LLM should NOT have been called again
        assert mock_llm_provider.complete.call_count == 1
        # No cards for note2
        assert _count_cards_for_note(temp_db_path, note2.id) == 0

    @pytest.mark.asyncio
    async def test_daily_quota_resets_next_day(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note
    ):
        """Daily quota should reset when the date changes."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
            max_per_day=1,
        )

        note_store.create(long_note)

        # First event today: should generate
        event1 = Event("note.created", {"note_id": long_note.id})
        await gen.handle_note_event(event1)
        assert mock_llm_provider.complete.call_count == 1

        # Set the daily log to a fixed past date so today's quota is free.
        conn = sqlite3.connect(temp_db_path)
        conn.execute(
            "INSERT OR REPLACE INTO card_generation_log (date, count) VALUES (?, ?)",
            ("2020-01-01", 1),
        )
        # Also delete today's entry
        today = datetime.utcnow().strftime("%Y-%m-%d")
        conn.execute("DELETE FROM card_generation_log WHERE date = ?", (today,))
        conn.commit()
        conn.close()

        # Create a second note
        note2 = Note(
            id="33333333-3333-3333-3333-333333333333",
            title="Second Long Note",
            content=(
                "TypeScript is a strongly typed programming language that builds on "
                "JavaScript, giving you better tooling at any scale. TypeScript adds "
                "additional syntax to JavaScript to support a tighter integration with "
                "your editor. Catch errors early in your editor. TypeScript code converts "
                "to JavaScript, which runs anywhere JavaScript runs: in a browser, on "
                "Node.js or Deno, and in your apps."
            ),
            frontmatter={"tags": ["typescript"]},
            links=[],
            path="/notes/typescript.md",
        )
        note_store.create(note2)

        # Second event: should now generate (quota reset)
        event2 = Event("note.created", {"note_id": note2.id})
        await gen.handle_note_event(event2)

        assert mock_llm_provider.complete.call_count == 2
        assert _count_cards_for_note(temp_db_path, note2.id) > 0


class TestEventHandling:
    """Tests for event-driven card generation."""

    @pytest.mark.asyncio
    async def test_note_created_event_triggers_generation(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note
    ):
        """A note.created event should trigger card generation."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
        )

        note_store.create(long_note)

        event = Event("note.created", {"note_id": long_note.id})
        await gen.handle_note_event(event)

        assert _count_cards_for_note(temp_db_path, long_note.id) > 0

    @pytest.mark.asyncio
    async def test_note_updated_event_triggers_generation(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note, long_note_v2
    ):
        """A note.updated event should trigger card generation when content changed."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
        )

        note_store.create(long_note)

        # First generate cards for original
        event1 = Event("note.created", {"note_id": long_note.id})
        await gen.handle_note_event(event1)
        first_count = _count_cards_for_note(temp_db_path, long_note.id)

        # Update note content
        note_store.update(long_note_v2)

        # Updated event
        event2 = Event("note.updated", {"note_id": long_note_v2.id})
        await gen.handle_note_event(event2)

        # Should have suspended old + generated new
        suspended = _count_suspended_cards_for_note(temp_db_path, long_note.id)
        assert suspended == first_count
        total = _count_cards_for_note(temp_db_path, long_note.id)
        assert total > first_count

    @pytest.mark.asyncio
    async def test_event_subscription_wiring(
        self, note_store, mock_llm_provider, temp_db_path, event_bus, long_note
    ):
        """Verify that the generator subscribes to note.created and note.updated."""
        from src.augmentation.card_auto import CardAutoGenerator

        CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
        )

        # Check that handlers are registered
        assert "note.created" in event_bus._subscribers
        assert "note.updated" in event_bus._subscribers
        assert len(event_bus._subscribers["note.created"]) >= 1
        assert len(event_bus._subscribers["note.updated"]) >= 1

    @pytest.mark.asyncio
    async def test_note_not_found_handled_gracefully(
        self, note_store, mock_llm_provider, temp_db_path, event_bus
    ):
        """If the note_id in the event doesn't exist, handle gracefully."""
        from src.augmentation.card_auto import CardAutoGenerator

        gen = CardAutoGenerator(
            note_store=note_store,
            llm_provider=mock_llm_provider,
            db_path=temp_db_path,
            event_bus=event_bus,
        )

        event = Event("note.created", {"note_id": "nonexistent-id"})
        # Should not raise
        await gen.handle_note_event(event)

        # LLM should not have been called
        mock_llm_provider.complete.assert_not_called()
