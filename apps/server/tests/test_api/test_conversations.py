"""Tests for the conversation history API (persist + paginate chat messages)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

_BASE = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def client():
    from src.main import app

    return TestClient(app)


def _store(user_id: str, role: str, content: str, created: datetime, **kw):
    from src.api import conversations as conv
    from src.models.conversation import ConversationMessage

    return conv.store_message(
        ConversationMessage(user_id=user_id, role=role, content=content, created=created, **kw)
    )


class TestConversationStore:
    def test_load_is_newest_first(self):
        from src.api import conversations as conv

        for i in range(3):
            _store("anonymous", "user", f"q{i}", _BASE + timedelta(minutes=i))

        rows = conv.load_messages("anonymous", limit=10)
        assert [r["content"] for r in rows] == ["q2", "q1", "q0"]

    def test_pagination_before_cursor(self):
        from src.api import conversations as conv

        for i in range(5):
            _store("anonymous", "user", f"q{i}", _BASE + timedelta(minutes=i))

        first = conv.load_messages("anonymous", limit=2)
        assert [r["content"] for r in first] == ["q4", "q3"]

        cursor = first[-1]["created"]  # oldest loaded so far
        older = conv.load_messages("anonymous", limit=2, before=cursor)
        assert [r["content"] for r in older] == ["q2", "q1"]

    def test_user_scoping(self):
        from src.api import conversations as conv

        _store("anonymous", "user", "mine", _BASE)
        _store("someone_else", "user", "theirs", _BASE)

        rows = conv.load_messages("anonymous", limit=10)
        assert [r["content"] for r in rows] == ["mine"]

    def test_recent_history_is_chronological(self):
        from src.api import conversations as conv

        for i in range(3):
            _store("anonymous", "user", f"q{i}", _BASE + timedelta(minutes=i))

        hist = conv.recent_history("anonymous", limit=10)
        assert [r["content"] for r in hist] == ["q0", "q1", "q2"]

    def test_citations_and_sub_questions_roundtrip(self):
        from src.api import conversations as conv

        _store(
            "anonymous",
            "assistant",
            "answer",
            _BASE,
            citations=[{"note_id": "n1", "title": "T", "snippet": "s"}],
            sub_questions=["sq1", "sq2"],
        )

        rows = conv.load_messages("anonymous", limit=10)
        assert rows[0]["citations"] == [{"note_id": "n1", "title": "T", "snippet": "s"}]
        assert rows[0]["sub_questions"] == ["sq1", "sq2"]


class TestConversationEndpoints:
    def test_get_messages(self, client):
        _store("anonymous", "user", "hello", _BASE)

        resp = client.get("/api/conversations/messages?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "hello"
        assert data[0]["role"] == "user"

    def test_delete_clears_history(self, client):
        from src.api import conversations as conv

        _store("anonymous", "user", "hello", _BASE)

        resp = client.delete("/api/conversations/messages")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 1
        assert conv.load_messages("anonymous", limit=10) == []
