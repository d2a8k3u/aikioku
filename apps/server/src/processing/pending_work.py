"""Durable queue of LLM-backed work deferred while the daily budget is exhausted.

When the daily LLM budget is reached, ingestion still persists the raw note /
memory but its LLM processing is enqueued here and drained once the budget
resets (UTC midnight) or is raised. Backed by SQLite (one ``pending_llm_work``
table) on the shared app database, like ``llm_costs`` and ``events``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import settings

logger = logging.getLogger(__name__)

# Drop an item after this many failed drain attempts so a permanently-failing
# payload can't wedge the queue.
_MAX_ATTEMPTS = 5

_DB_TABLE = """
CREATE TABLE IF NOT EXISTS pending_llm_work (
    id TEXT PRIMARY KEY,
    work_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
)
"""

# De-dupe key: re-enqueuing the same (work_type, entity_id) refreshes the payload
# instead of creating a second row.
_DB_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_work_dedupe "
    "ON pending_llm_work(work_type, entity_id)"
)


def _connect() -> sqlite3.Connection:
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table() -> None:
    with _connect() as conn:
        conn.execute(_DB_TABLE)
        conn.execute(_DB_INDEX)


def enqueue(work_type: str, entity_id: str, payload: dict[str, Any]) -> None:
    """Add a work item, de-duplicating on ``(work_type, entity_id)``.

    A repeat enqueue for the same key refreshes the payload and resets the
    attempt counter rather than creating a duplicate row.
    """
    _ensure_table()
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO pending_llm_work
                (id, work_type, entity_id, payload, created, attempts, last_error)
            VALUES (?, ?, ?, ?, ?, 0, NULL)
            ON CONFLICT(work_type, entity_id) DO UPDATE SET
                payload = excluded.payload, created = excluded.created,
                attempts = 0, last_error = NULL
            """,
            (str(uuid4()), work_type, entity_id, json.dumps(payload), now),
        )
    logger.info("pending_work.enqueued type=%s entity=%s", work_type, entity_id)


def count() -> int:
    _ensure_table()
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM pending_llm_work").fetchone()
    return int(row[0])


def list_pending(limit: int = 100) -> list[dict[str, Any]]:
    """Oldest-first pending items; each item's ``payload`` is decoded to a dict."""
    _ensure_table()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, work_type, entity_id, payload, created, attempts "
            "FROM pending_llm_work ORDER BY created LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "work_type": r["work_type"],
            "entity_id": r["entity_id"],
            "payload": json.loads(r["payload"]),
            "created": r["created"],
            "attempts": r["attempts"],
        }
        for r in rows
    ]


def delete(item_id: str) -> None:
    _ensure_table()
    with _connect() as conn:
        conn.execute("DELETE FROM pending_llm_work WHERE id = ?", (item_id,))


def bump_attempt(item_id: str, error: str) -> None:
    """Record a failed drain attempt; drop the item once it hits the cap."""
    _ensure_table()
    with _connect() as conn:
        row = conn.execute(
            "SELECT attempts FROM pending_llm_work WHERE id = ?", (item_id,)
        ).fetchone()
        if row is None:
            return
        attempts = row["attempts"] + 1
        if attempts >= _MAX_ATTEMPTS:
            conn.execute("DELETE FROM pending_llm_work WHERE id = ?", (item_id,))
            logger.warning(
                "pending_work.dropped id=%s after %d attempts: %s",
                item_id,
                attempts,
                error,
            )
        else:
            conn.execute(
                "UPDATE pending_llm_work SET attempts = ?, last_error = ? WHERE id = ?",
                (attempts, error[:500], item_id),
            )
