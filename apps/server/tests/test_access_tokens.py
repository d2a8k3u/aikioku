"""Tests for the personal access token (PAT) store."""
from __future__ import annotations

import sqlite3

import pytest

from src import access_tokens
from src.config import settings


class TestCreateAndVerify:
    def test_create_returns_plaintext_once(self):
        record, plaintext = access_tokens.create_token("ci", "alice", "full")
        assert plaintext.startswith("sbk_")
        assert record.prefix == plaintext[:12]
        assert record.scope == "full"
        assert record.username == "alice"

    def test_verify_matches_created_token(self):
        _, plaintext = access_tokens.create_token("ci", "alice", "read")
        record = access_tokens.verify_token(plaintext)
        assert record is not None
        assert record.scope == "read"
        assert record.username == "alice"

    def test_plaintext_not_stored(self):
        _, plaintext = access_tokens.create_token("ci", "alice", "full")
        raw = sqlite3.connect(settings.sqlite_db_path).execute(
            "SELECT token_hash FROM access_tokens"
        ).fetchone()[0]
        assert plaintext not in raw  # only the sha256 hash is persisted

    def test_verify_rejects_unknown_token(self):
        assert access_tokens.verify_token("sbk_does_not_exist") is None

    def test_verify_rejects_non_prefixed(self):
        assert access_tokens.verify_token("random-jwt-like-thing") is None

    def test_verify_stamps_last_used(self):
        _, plaintext = access_tokens.create_token("ci", "alice", "full")
        assert access_tokens.verify_token(plaintext).last_used is None
        # Second verify sees the stamp written by the first.
        access_tokens.verify_token(plaintext)
        assert access_tokens.verify_token(plaintext).last_used is not None

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValueError):
            access_tokens.create_token("ci", "alice", "admin")


class TestListAndDelete:
    def test_list_excludes_secret(self):
        access_tokens.create_token("ci", "alice", "full")
        tokens = access_tokens.list_tokens()
        assert len(tokens) == 1
        assert not hasattr(tokens[0], "token_hash")

    def test_delete_revokes(self):
        record, plaintext = access_tokens.create_token("ci", "alice", "full")
        assert access_tokens.delete_token(record.id) is True
        assert access_tokens.verify_token(plaintext) is None

    def test_delete_unknown_returns_false(self):
        assert access_tokens.delete_token("nope") is False
