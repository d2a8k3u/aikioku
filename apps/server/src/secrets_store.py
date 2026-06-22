"""Encrypted secrets store.

Replaces the old ``.env`` file for sensitive values (API keys, JWT signing
secret). Secrets are encrypted with Fernet (AES-128-CBC + HMAC) and stored in the
``app_secrets`` SQLite table. The master key lives in a key FILE outside the DB —
a chicken-and-egg constraint: the DB can't decrypt itself. The key file is
auto-generated on first run and persisted in the data volume.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from src.config import settings

_DB_TABLE = """
CREATE TABLE IF NOT EXISTS app_secrets (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL,
    created TEXT NOT NULL,
    modified TEXT NOT NULL
)
"""

_lock = threading.Lock()
_fernet: Fernet | None = None


def _key_file_path() -> Path:
    """Resolve the master key file path (next to the SQLite DB by default)."""
    if settings.secret_key_file:
        return Path(settings.secret_key_file)
    return Path(settings.sqlite_db_path).parent / "secret.key"


def _load_or_create_key() -> bytes:
    """Load the Fernet master key, generating + persisting it on first run."""
    path = _key_file_path()
    if path.exists():
        return path.read_bytes().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    # Write restrictively (owner-only) so the key isn't world-readable.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key)
    finally:
        os.close(fd)
    return key


def _cipher() -> Fernet:
    global _fernet
    if _fernet is None:
        with _lock:
            if _fernet is None:
                _fernet = Fernet(_load_or_create_key())
    return _fernet


def _ensure_table() -> None:
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(_DB_TABLE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_secret(key: str, plaintext: str) -> None:
    """Encrypt and upsert a secret. Empty plaintext deletes the secret instead."""
    if plaintext == "":
        delete_secret(key)
        return
    _ensure_table()
    token = _cipher().encrypt(plaintext.encode("utf-8"))
    now = _now()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(
            "INSERT INTO app_secrets (key, value, created, modified) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, modified=excluded.modified",
            (key, token, now, now),
        )


def get_secret(key: str) -> str | None:
    """Return the decrypted secret, or None if absent/undecryptable."""
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        row = conn.execute(
            "SELECT value FROM app_secrets WHERE key = ?", (key,)
        ).fetchone()
    if row is None:
        return None
    try:
        return _cipher().decrypt(row[0]).decode("utf-8")
    except (InvalidToken, ValueError):
        return None


def has_secret(key: str) -> bool:
    return get_secret(key) is not None


def delete_secret(key: str) -> None:
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute("DELETE FROM app_secrets WHERE key = ?", (key,))


def list_secret_keys() -> list[str]:
    """Return the names of stored secrets (never the values)."""
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        rows = conn.execute("SELECT key FROM app_secrets ORDER BY key").fetchall()
    return [r[0] for r in rows]
