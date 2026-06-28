"""Conversation history API: persist and paginate the user's chat messages.

Single continuous thread per user (keyed by ``user_id``). Messages are appended
by the chat endpoints (see :mod:`src.api.chat`) and read back here for the UI,
newest-first with cursor pagination for scroll-up loading.

Follows the raw-sqlite3 pattern of :mod:`src.api.memory` (``_ensure_table`` +
``with sqlite3.connect(settings.sqlite_db_path)``, no ORM).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query

from src.auth import UserInDB, require_auth
from src.config import settings
from src.models.conversation import ConversationMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT,
    sub_questions TEXT,
    created TEXT NOT NULL,
    in_progress INTEGER NOT NULL DEFAULT 0
)
"""

_DB_INDEX = """
CREATE INDEX IF NOT EXISTS idx_convmsg_user_created
    ON conversation_messages(user_id, created)
"""


def _ensure_table() -> None:
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(_DB_SCHEMA)
        conn.execute(_DB_INDEX)
        # Migration: add in_progress column for deployments that pre-date it.
        try:
            conn.execute(
                "ALTER TABLE conversation_messages ADD COLUMN in_progress INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass  # column already exists


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "user_id": row[1],
        "role": row[2],
        "content": row[3],
        "citations": json.loads(row[4]) if row[4] else [],
        "sub_questions": json.loads(row[5]) if row[5] else [],
        "created": row[6],
        "in_progress": bool(row[7]) if len(row) > 7 else False,
    }


def store_message(message: ConversationMessage) -> ConversationMessage:
    """Insert (or replace) a chat message. Returns the stored message."""
    _ensure_table()
    in_progress = 1 if getattr(message, "in_progress", False) else 0
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO conversation_messages
               (id, user_id, role, content, citations, sub_questions, created, in_progress)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message.id,
                message.user_id,
                message.role,
                message.content,
                json.dumps(message.citations),
                json.dumps(message.sub_questions),
                message.created.isoformat(),
                in_progress,
            ),
        )
    return message


def update_message(
    message_id: str,
    content: str | None = None,
    citations: list[Any] | None = None,
    sub_questions: list[Any] | None = None,
    in_progress: bool | None = None,
) -> dict[str, Any] | None:
    """Update an existing assistant message in place.

    Used to promote an ``in_progress`` placeholder to its final content once
    generation completes. Returns the updated row, or ``None`` if no row was
    found.
    """
    _ensure_table()
    fields: list[str] = []
    params: list[Any] = []
    if content is not None:
        fields.append("content = ?")
        params.append(content)
    if citations is not None:
        fields.append("citations = ?")
        params.append(json.dumps(citations))
    if sub_questions is not None:
        fields.append("sub_questions = ?")
        params.append(json.dumps(sub_questions))
    if in_progress is not None:
        fields.append("in_progress = ?")
        params.append(1 if in_progress else 0)
    if not fields:
        return _get_message(message_id)

    params.append(message_id)
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(
            f"UPDATE conversation_messages SET {', '.join(fields)} WHERE id = ?",
            params,
        )
    return _get_message(message_id)


def _get_message(message_id: str) -> dict[str, Any] | None:
    _ensure_table()
    columns = "id, user_id, role, content, citations, sub_questions, created, in_progress"
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        row = conn.execute(
            f"SELECT {columns} FROM conversation_messages WHERE id = ?",
            (message_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def delete_message(message_id: str) -> bool:
    """Delete a single message by id. Returns True if a row was removed."""
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        cursor = conn.execute("DELETE FROM conversation_messages WHERE id = ?", (message_id,))
    return cursor.rowcount > 0


def load_messages(user_id: str, limit: int = 10, before: str | None = None) -> list[dict[str, Any]]:
    """Load a page of the user's messages, newest-first.

    Args:
        user_id: The thread owner.
        limit: Maximum rows to return.
        before: ISO ``created`` cursor of the oldest message already loaded.
            When set, only rows strictly older than it are returned — this is
            how the UI pages further back on scroll-up loading.

    Returns:
        A list of message dicts ordered newest-first.
    """
    _ensure_table()
    columns = "id, user_id, role, content, citations, sub_questions, created, in_progress"
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        if before:
            cursor = conn.execute(
                f"""SELECT {columns}
                   FROM conversation_messages
                   WHERE user_id = ? AND created < ?
                   ORDER BY created DESC LIMIT ?""",
                (user_id, before, limit),
            )
        else:
            cursor = conn.execute(
                f"""SELECT {columns}
                   FROM conversation_messages
                   WHERE user_id = ?
                   ORDER BY created DESC LIMIT ?""",
                (user_id, limit),
            )
        return [_row_to_dict(r) for r in cursor.fetchall()]


def recent_history(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return the last ``limit`` messages in chronological (old->new) order.

    Used to ground generation with short-term conversation context so the
    assistant can answer in-session questions like "what was my previous
    question?".

    In-progress assistant placeholders are excluded: they have no content yet
    and would only confuse the LLM context.
    """
    rows = load_messages(user_id, limit=limit)
    return [
        r for r in reversed(rows) if not (r.get("role") == "assistant" and r.get("in_progress"))
    ]


def iter_turns_for_reembed() -> list[tuple[str, str]]:
    """Reconstruct ``(assistant_turn_id, embed_text)`` for every completed turn.

    Mirrors :func:`src.api.chat._embed_turn`: each assistant message is paired
    with the most recent preceding user message (per ``user_id``, ordered by
    ``created``), and the embed text is
    ``f"[{date}] user: {u}\\nassistant: {a}"`` using the assistant row's date.
    The assistant row ``id`` is the vector-store key. Best-effort: orphan user
    messages (no assistant reply) are skipped — they were never embedded.

    In-progress assistant placeholders are also skipped because they do not yet
    have final content.
    """
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        rows = conn.execute(
            "SELECT id, user_id, role, content, created, in_progress FROM conversation_messages "
            "ORDER BY user_id, created ASC, id ASC"
        ).fetchall()
    out: list[tuple[str, str]] = []
    last_user: dict[str, str] = {}
    for _id, user_id, role, content, created, in_progress in rows:
        if role == "user":
            last_user[user_id] = content
        elif role == "assistant" and not in_progress:
            u = last_user.get(user_id, "")
            date = (created or "")[:10]
            out.append((_id, f"[{date}] user: {u}\nassistant: {content}"))
    return out


def clear_messages(user_id: str) -> int:
    """Delete all of the user's messages. Returns the number deleted."""
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        cursor = conn.execute("DELETE FROM conversation_messages WHERE user_id = ?", (user_id,))
        return cursor.rowcount


@router.get("/messages")
async def get_messages(
    user: UserInDB = Depends(require_auth),
    limit: int = Query(10, ge=1, le=100),
    before: str | None = None,
) -> list[dict[str, Any]]:
    """Return a page of the authenticated user's chat history, newest-first.

    Pass the oldest loaded message's ``created`` as ``before`` to page further
    back (scroll-up loading).
    """
    return load_messages(user.username, limit=limit, before=before)


@router.get("/messages/{message_id}")
async def get_message_endpoint(
    message_id: str,
    user: UserInDB = Depends(require_auth),
) -> dict[str, Any]:
    """Return a single chat message by id.

    Used by the frontend to poll in-progress placeholders after a mid-generation
    reload — the client re-fetches the placeholder until it is promoted to its
    final content.
    """
    message = _get_message(message_id)
    if message is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Message not found")
    if message.get("user_id") != user.username:
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Not authorized")
    return message


@router.delete("/messages/{message_id}")
async def delete_message_endpoint(
    message_id: str,
    user: UserInDB = Depends(require_auth),
) -> dict[str, Any]:
    """Delete a single chat message by id.

    Only messages owned by the authenticated user can be deleted.
    """
    message = _get_message(message_id)
    if message is None:
        return {"status": "not_found", "deleted": False}
    if message.get("user_id") != user.username:
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Not authorized to delete this message")
    deleted = delete_message(message_id)
    return {"status": "deleted" if deleted else "not_found", "deleted": deleted}


@router.delete("/messages")
async def delete_messages(user: UserInDB = Depends(require_auth)) -> dict[str, Any]:
    """Delete the authenticated user's entire chat history."""
    deleted = clear_messages(user.username)
    return {"status": "cleared", "deleted": deleted}
