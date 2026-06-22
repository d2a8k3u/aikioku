"""Tests for the reembed status endpoint and conversation turn reconstruction."""
from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

from src.api.conversations import iter_turns_for_reembed, store_message
from src.models.conversation import ConversationMessage


def test_reembed_status_endpoint():
    from src.main import app

    with TestClient(app) as client:
        resp = client.get("/api/admin/reembed/status")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("state", "processed_notes", "total_notes", "processed_convs", "total_convs"):
            assert key in body


def test_iter_turns_reproduces_embed_text():
    store_message(ConversationMessage(
        id="u1", user_id="alice", role="user", content="hi there",
        created=datetime(2026, 6, 14, 10, 0, 0),
    ))
    store_message(ConversationMessage(
        id="a1", user_id="alice", role="assistant", content="hello back",
        created=datetime(2026, 6, 14, 10, 0, 1),
    ))

    turns = iter_turns_for_reembed()

    assert ("a1", "[2026-06-14] user: hi there\nassistant: hello back") in turns


def test_iter_turns_skips_orphan_user_message():
    store_message(ConversationMessage(
        id="u-only", user_id="bob", role="user", content="no reply",
        created=datetime(2026, 6, 14, 11, 0, 0),
    ))
    ids = [tid for tid, _ in iter_turns_for_reembed()]
    assert "u-only" not in ids
