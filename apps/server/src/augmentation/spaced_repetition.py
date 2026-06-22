"""Spaced Repetition system using SM-2 algorithm with SQLite storage."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta

from src.llm.base import LLMProvider
from src.llm.json_parse import parse_llm_json
from src.models.card import Card, CardStatus, CardType
from src.models.note import Note
from src.storage.note_store import NoteStore

logger = logging.getLogger(__name__)


class SpacedRepetition:
    """Manages spaced repetition flashcards using the SM-2 algorithm.

    Cards are stored in a SQLite database. Card generation is powered
    by an LLM provider that extracts cloze, Q-A, and connection cards
    from note content.
    """

    _CREATE_TABLE_SQL = """
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

    _GENERATE_PROMPT_SYSTEM = (
        "You are a spaced repetition card generator. "
        "Given a note's content, generate 2-4 high-quality flashcards. "
        "Return ONLY a JSON array of objects, each with keys: "
        '"type" (one of "cloze", "qa", "connection"), "front", "back". '
        "Do not include any other text or explanation."
    )

    def __init__(
        self,
        note_store: NoteStore,
        llm_provider: LLMProvider,
        db_path: str,
    ) -> None:
        """Initialize SpacedRepetition with storage, LLM, and database path.

        Creates the cards table if it does not exist.
        """
        self._note_store = note_store
        self._llm = llm_provider
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the cards table if it doesn't exist."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(self._CREATE_TABLE_SQL)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a SQLite connection with row factory."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def generate_cards(self, note: Note) -> list[Card]:
        """Generate flashcards from note content using the LLM and persist them.

        Args:
            note: The source note to generate cards from.

        Returns:
            A list of Card objects created from the LLM response.
        """
        prompt = (
            f"Title: {note.title}\n\n"
            f"Content:\n{note.content}\n\n"
            f"Generate flashcards as a JSON array."
        )
        response = await self._llm.complete(
            prompt=prompt,
            system=self._GENERATE_PROMPT_SYSTEM,
        )
        # Propagates LLMOutputParseError when the model returns unparseable output;
        # the API layer maps that to a 502.
        raw_cards = parse_llm_json(response, expect="list")

        cards: list[Card] = []
        now = datetime.utcnow()
        for raw in raw_cards:
            try:
                card = Card(
                    note_id=note.id,
                    type=CardType(raw["type"]),
                    front=raw["front"],
                    back=raw.get("back", ""),
                    next_review=now,
                    status=CardStatus.new,
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Card generation: skipping malformed card item: %s", exc)
                continue
            cards.append(card)

        # Persist generated cards to SQLite
        conn = sqlite3.connect(self._db_path)
        for card in cards:
            row = self._card_to_row(card)
            conn.execute(
                "INSERT OR REPLACE INTO cards (id, note_id, type, front, back, ease_factor, interval, repetitions, next_review, status) "
                "VALUES (:id, :note_id, :type, :front, :back, :ease_factor, :interval, :repetitions, :next_review, :status)",
                row,
            )
        conn.commit()
        conn.close()

        return cards

    async def get_suspended_count(self) -> int:
        """Return the number of suspended cards."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM cards WHERE status = ?", (CardStatus.suspended,)
        ).fetchone()
        conn.close()
        return row[0] or 0

    async def get_due_cards(self, limit: int = 20) -> list[Card]:
        """Retrieve cards that are due for review.

        Args:
            limit: Maximum number of cards to return.

        Returns:
            A list of Card objects where next_review <= now.
        """
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM cards WHERE next_review <= ? ORDER BY next_review ASC LIMIT ?",
            (now, limit),
        ).fetchall()
        conn.close()

        return [self._row_to_card(dict(row)) for row in rows]

    async def review_card(self, card_id: str, rating: int) -> Card:
        """Review a card and update its SM-2 parameters.

        Args:
            card_id: The ID of the card to review.
            rating: SM-2 quality rating (1=again, 2=hard, 3=good, 4=easy).

        Returns:
            The updated Card.

        Raises:
            ValueError: If the card is not found.
        """
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
        if row is None:
            conn.close()
            raise ValueError(f"Card not found: {card_id}")

        card = self._row_to_card(dict(row))
        conn.close()

        # SM-2 algorithm
        if rating == 1:
            # Again: reset
            card.interval = 1
            card.repetitions = 0
            card.status = CardStatus.learning
        elif rating == 2:
            # Hard
            card.interval = max(1, int(card.interval * 1.2))
            card.ease_factor = max(1.3, card.ease_factor - 0.15)
            card.repetitions += 1
            card.status = CardStatus.learning
        elif rating == 3:
            # Good
            card.interval = int(card.interval * card.ease_factor)
            if card.interval == 0:
                card.interval = 1
            card.repetitions += 1
            if card.repetitions >= 2:
                card.status = CardStatus.review
            else:
                card.status = CardStatus.learning
        elif rating == 4:
            # Easy
            card.interval = int(card.interval * card.ease_factor * 1.3)
            if card.interval == 0:
                card.interval = 1
            card.ease_factor = card.ease_factor + 0.15
            card.repetitions += 1
            card.status = CardStatus.review

        card.next_review = datetime.utcnow() + timedelta(days=card.interval)

        # Persist
        row_data = self._card_to_row(card)
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "UPDATE cards SET note_id=:note_id, type=:type, front=:front, back=:back, "
            "ease_factor=:ease_factor, interval=:interval, repetitions=:repetitions, "
            "next_review=:next_review, status=:status WHERE id=:id",
            row_data,
        )
        conn.commit()
        conn.close()

        return card

    async def get_stats(self) -> dict:
        """Return statistics about the card collection.

        Returns:
            A dict with keys: total, due, new, learning, review.
        """
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()

        total = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        due = conn.execute(
            "SELECT COUNT(*) FROM cards WHERE next_review <= ?", (now,)
        ).fetchone()[0]
        new_count = conn.execute(
            "SELECT COUNT(*) FROM cards WHERE status = ?", (CardStatus.new,)
        ).fetchone()[0]
        learning = conn.execute(
            "SELECT COUNT(*) FROM cards WHERE status = ?", (CardStatus.learning,)
        ).fetchone()[0]
        review = conn.execute(
            "SELECT COUNT(*) FROM cards WHERE status = ?", (CardStatus.review,)
        ).fetchone()[0]
        conn.close()

        return {
            "total": total,
            "due": due,
            "new": new_count,
            "learning": learning,
            "review": review,
        }

    def _card_to_row(self, card: Card) -> dict:
        """Serialize a Card to a SQLite-compatible dictionary.

        Args:
            card: The Card to serialize.

        Returns:
            A dictionary suitable for SQLite INSERT/UPDATE.
        """
        return {
            "id": card.id,
            "note_id": card.note_id,
            "type": card.type.value,
            "front": card.front,
            "back": card.back,
            "ease_factor": card.ease_factor,
            "interval": card.interval,
            "repetitions": card.repetitions,
            "next_review": card.next_review.isoformat(),
            "status": card.status.value,
        }

    def _row_to_card(self, row: dict) -> Card:
        """Deserialize a SQLite row into a Card.

        Args:
            row: A dictionary from a SQLite row.

        Returns:
            A Card instance.
        """
        return Card(
            id=row["id"],
            note_id=row["note_id"],
            type=CardType(row["type"]),
            front=row["front"],
            back=row["back"],
            ease_factor=row["ease_factor"],
            interval=row["interval"],
            repetitions=row["repetitions"],
            next_review=datetime.fromisoformat(row["next_review"]),
            status=CardStatus(row["status"]),
        )
