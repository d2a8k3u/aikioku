"""Automatic review card generation pipeline.

Generates spaced repetition flashcards when notes are created or updated,
with eligibility checks, daily balancing, and content-hash obsolescence.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime

import structlog

from src.events import Event, EventBus
from src.llm.base import LLMProvider
from src.models.card import CardStatus
from src.models.note import Note
from src.storage.note_store import NoteStore

logger = structlog.get_logger(__name__)


class CardAutoGenerator:
    """Automatically generates spaced repetition flashcards on note events.

    Pipeline:
    1. Eligibility check: only notes with >200 chars of content
    2. Daily balancer: limit to max cards per day (default 20)
    3. Content-hash obsolescence: skip if content unchanged; suspend old
       cards and generate new ones if content changed
    4. Event-driven: subscribes to note.created and note.updated
    """

    _CREATE_META_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS card_generation_meta (
            note_id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            generated_at TEXT NOT NULL
        )
    """

    _CREATE_LOG_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS card_generation_log (
            date TEXT PRIMARY KEY,
            count INTEGER DEFAULT 0
        )
    """

    _CREATE_CARDS_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS cards (
            id TEXT PRIMARY KEY,
            note_id TEXT NOT NULL,
            type TEXT NOT NULL,
            front TEXT NOT NULL,
            back TEXT NOT NULL,
            ease_factor REAL NOT NULL DEFAULT 2.5,
            interval INTEGER NOT NULL DEFAULT 0,
            repetitions INTEGER NOT NULL DEFAULT 0,
            next_review TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new'
        )
    """

    def __init__(
        self,
        note_store: NoteStore,
        llm_provider: LLMProvider,
        db_path: str,
        event_bus: EventBus,
        max_per_day: int = 20,
        min_content_length: int = 200,
    ) -> None:
        """Initialize the CardAutoGenerator.

        Args:
            note_store: The NoteStore for looking up notes by ID.
            llm_provider: LLM provider for card generation.
            db_path: Path to the SQLite database.
            event_bus: EventBus for subscribing to note events.
            max_per_day: Maximum cards to generate per day (default 20).
            min_content_length: Minimum content length for eligibility (default 200).
        """
        self._note_store = note_store
        self._llm = llm_provider
        self._db_path = db_path
        self._event_bus = event_bus
        self._max_per_day = max_per_day
        self._min_content_length = min_content_length
        self._init_db()
        self._subscribe()

    # ------------------------------------------------------------------ DB init

    def _init_db(self) -> None:
        """Create the card_generation_meta, card_generation_log, and cards tables."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(self._CREATE_META_TABLE_SQL)
        conn.execute(self._CREATE_LOG_TABLE_SQL)
        conn.execute(self._CREATE_CARDS_TABLE_SQL)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a SQLite connection with row factory."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ Event subscription

    def _subscribe(self) -> None:
        """Subscribe to note.created and note.updated events."""
        self._event_bus.subscribe("note.created", self.handle_note_event)
        self._event_bus.subscribe("note.updated", self.handle_note_event)
        logger.info("card_auto.subscribed", events=["note.created", "note.updated"])

    # ------------------------------------------------------------------ Content hash

    @staticmethod
    def content_hash(note: Note) -> str:
        """Compute SHA-256 hash of note content.

        Args:
            note: The note to hash.

        Returns:
            Hex-encoded SHA-256 digest of note.content.
        """
        return hashlib.sha256(note.content.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ Eligibility

    async def is_eligible(self, note: Note) -> bool:
        """Check if a note is eligible for card generation.

        A note is eligible if its stripped content length exceeds
        the minimum threshold.

        Args:
            note: The note to check.

        Returns:
            True if the note is eligible, False otherwise.
        """
        return len(note.content.strip()) >= self._min_content_length

    # ------------------------------------------------------------------ Daily balancer

    async def check_daily_quota(self) -> bool:
        """Check if we're still within the daily generation quota.

        Returns:
            True if we can still generate cards today, False if quota exceeded.
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        conn = self._get_conn()
        row = conn.execute(
            "SELECT count FROM card_generation_log WHERE date = ?", (today,)
        ).fetchone()
        conn.close()

        current = row["count"] if row else 0
        return current < self._max_per_day

    async def record_generation(self, note_id: str, content_hash_val: str) -> None:
        """Record a card generation event: increment daily count + store content hash.

        Args:
            note_id: The note ID that had cards generated.
            content_hash_val: The SHA-256 hash of the note content at generation time.
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        now_iso = datetime.utcnow().isoformat()

        conn = sqlite3.connect(self._db_path)
        # Increment daily count
        conn.execute(
            "INSERT INTO card_generation_log (date, count) VALUES (?, 1) "
            "ON CONFLICT(date) DO UPDATE SET count = count + 1",
            (today,),
        )
        # Store content hash for this note
        conn.execute(
            "INSERT OR REPLACE INTO card_generation_meta (note_id, content_hash, generated_at) "
            "VALUES (?, ?, ?)",
            (note_id, content_hash_val, now_iso),
        )
        conn.commit()
        conn.close()

        logger.debug(
            "card_auto.recorded",
            note_id=note_id,
            date=today,
        )

    # ------------------------------------------------------------------ Stored hash lookup

    def _get_stored_hash(self, note_id: str) -> str | None:
        """Get the stored content hash for a note, if any.

        Args:
            note_id: The note ID to look up.

        Returns:
            The stored hash string, or None if no generation recorded.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT content_hash FROM card_generation_meta WHERE note_id = ?",
            (note_id,),
        ).fetchone()
        conn.close()
        return row["content_hash"] if row else None

    # ------------------------------------------------------------------ Card suspension

    def _suspend_cards_for_note(self, note_id: str) -> int:
        """Suspend all non-suspended cards for a given note.

        Args:
            note_id: The note whose cards should be suspended.

        Returns:
            Number of cards suspended.
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute(
            "UPDATE cards SET status = ? WHERE note_id = ? AND status != ?",
            (CardStatus.suspended.value, note_id, CardStatus.suspended.value),
        )
        count = cursor.rowcount
        conn.commit()
        conn.close()
        if count > 0:
            logger.info("card_auto.suspended", note_id=note_id, count=count)
        return count

    # ------------------------------------------------------------------ Main handler

    async def handle_note_event(self, event: Event) -> None:
        """Handle a note.created or note.updated event.

        Pipeline:
        1. Look up the note from the store
        2. Check eligibility (content length)
        3. Check daily quota
        4. Compare content hash with stored hash
        5. If changed: suspend old cards, generate new ones
        6. If unchanged: skip

        Args:
            event: The event with note_id in event.data.
        """
        note_id = event.data.get("note_id")
        if not note_id:
            logger.warning("card_auto.missing_note_id", event_type=event.type)
            return

        # Look up the note
        note = self._note_store.get(note_id)
        if note is None:
            logger.warning("card_auto.note_not_found", note_id=note_id)
            return

        # Eligibility check
        if not await self.is_eligible(note):
            logger.debug("card_auto.ineligible", note_id=note_id, reason="content too short")
            return

        # Daily quota check
        if not await self.check_daily_quota():
            logger.info("card_auto.quota_exceeded", note_id=note_id)
            return

        # Content hash check
        current_hash = self.content_hash(note)
        stored_hash = self._get_stored_hash(note_id)

        if stored_hash is not None and stored_hash == current_hash:
            logger.debug("card_auto.skipped", note_id=note_id, reason="content unchanged")
            return

        # Content changed or first generation: generate cards
        await self.generate_for_note(note)

    # ------------------------------------------------------------------ Card generation

    async def generate_for_note(self, note: Note) -> None:
        """Suspend old cards for this note and generate new ones.

        Args:
            note: The note to generate cards for.
        """
        from src.augmentation.spaced_repetition import SpacedRepetition

        # Suspend existing cards for this note
        self._suspend_cards_for_note(note.id)

        # Generate new cards
        sr = SpacedRepetition(
            note_store=self._note_store,
            llm_provider=self._llm,
            db_path=self._db_path,
        )
        try:
            cards = await sr.generate_cards(note)
            logger.info(
                "card_auto.generated",
                note_id=note.id,
                card_count=len(cards),
            )
        except Exception:
            logger.exception("card_auto.generation_failed", note_id=note.id)
            return

        # Record the generation
        await self.record_generation(note.id, self.content_hash(note))
