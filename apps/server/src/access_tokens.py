"""Personal access tokens (PATs) for machine clients (MCP, external apps).

Unlike ``secrets_store`` (reversible Fernet encryption for provider API keys), auth
tokens are stored **one-way hashed** and shown to the user only once at creation —
a leaked DB read must not reveal usable tokens. Each token carries a ``scope``
(``read`` or ``full``) and the ``username`` of its creator, which the MCP layer
uses to mint a short-lived internal JWT when proxying calls to the REST API.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings

_DB_TABLE = """
CREATE TABLE IF NOT EXISTS access_tokens (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    username TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    scope TEXT NOT NULL,
    created TEXT NOT NULL,
    last_used TEXT
)
"""

TOKEN_PREFIX = "sbk_"
VALID_SCOPES = ("read", "full")


@dataclass
class AccessToken:
    id: str
    name: str
    username: str
    prefix: str
    scope: str
    created: str
    last_used: str | None


def _ensure_table() -> None:
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(_DB_TABLE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def _row_to_token(row: tuple[Any, ...]) -> AccessToken:
    return AccessToken(
        id=row[0],
        name=row[1],
        username=row[2],
        prefix=row[3],
        scope=row[4],
        created=row[5],
        last_used=row[6],
    )


def create_token(name: str, username: str, scope: str) -> tuple[AccessToken, str]:
    """Create a token, returning the record and the plaintext (shown only once)."""
    if scope not in VALID_SCOPES:
        raise ValueError(f"Invalid scope: {scope!r} (expected one of {VALID_SCOPES})")
    plaintext = TOKEN_PREFIX + secrets.token_urlsafe(32)
    token_id = uuid.uuid4().hex
    prefix = plaintext[:12]
    now = _now()
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(
            "INSERT INTO access_tokens "
            "(id, name, username, token_hash, prefix, scope, created, last_used) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            (token_id, name, username, _hash(plaintext), prefix, scope, now),
        )
    record = AccessToken(
        id=token_id,
        name=name,
        username=username,
        prefix=prefix,
        scope=scope,
        created=now,
        last_used=None,
    )
    return record, plaintext


def verify_token(plaintext: str) -> AccessToken | None:
    """Return the matching token (and stamp ``last_used``), or None if invalid."""
    if not plaintext or not plaintext.startswith(TOKEN_PREFIX):
        return None
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        row = conn.execute(
            "SELECT id, name, username, prefix, scope, created, last_used "
            "FROM access_tokens WHERE token_hash = ?",
            (_hash(plaintext),),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE access_tokens SET last_used = ? WHERE id = ?",
            (_now(), row[0]),
        )
    return _row_to_token(row)


def list_tokens() -> list[AccessToken]:
    """List tokens (never includes the hash or plaintext)."""
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, username, prefix, scope, created, last_used "
            "FROM access_tokens ORDER BY created DESC"
        ).fetchall()
    return [_row_to_token(r) for r in rows]


def delete_token(token_id: str) -> bool:
    """Revoke a token by id. Returns True if a row was removed."""
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        cur = conn.execute("DELETE FROM access_tokens WHERE id = ?", (token_id,))
        return cur.rowcount > 0
