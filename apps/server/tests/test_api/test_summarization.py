"""Tests for Summarization API endpoint."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.models.note import Note


@pytest.fixture
def client(tmp_path):
    """Create a FastAPI TestClient with a NoteStore containing a sample note."""
    from src.main import app

    with tempfile.TemporaryDirectory() as tmpdir:
        from src.storage.note_store import NoteStore

        store = NoteStore(tmpdir)
        note = Note(
            title="Python Programming",
            content="Python is a high-level programming language. It supports multiple paradigms.",
            path="python.md",
        )
        created = store.create(note)
        app.state.note_store = store

        # Patch summarizer to avoid real LLM calls
        mock_summarizer = AsyncMock()
        mock_summarizer.summarize.return_value = {
            "brief": "- Bullet one\n- Bullet two",
            "detailed": "Python is a language with many paradigms.",
            "one-liner": "Python is a versatile language.",
        }
        app.state.summarizer = mock_summarizer

        yield TestClient(app), created


class TestSummarizeNote:
    def test_returns_summary(self, client):
        cli, note = client
        response = cli.post(f"/api/notes/{note.id}/summarize")
        assert response.status_code == 200
        data = response.json()
        assert data["note_id"] == note.id
        assert "summary" in data
        assert "brief" in data["summary"]
        assert "detailed" in data["summary"]
        assert "one-liner" in data["summary"]

    def test_uses_real_summarizer_when_not_patched(self, client):
        cli, note = client
        # Remove the patched summarizer so endpoint instantiates one
        del cli.app.state.summarizer
        # Provide a mock llm_provider on app.state
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value="summary text")
        cli.app.state.llm_provider = mock_llm
        response = cli.post(f"/api/notes/{note.id}/summarize")
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data

    def test_nonexistent_note_returns_404(self, client):
        cli, note = client
        response = cli.post("/api/notes/00000000-0000-0000-0000-000000000000/summarize")
        assert response.status_code == 404

    def test_invalid_uuid_returns_422(self, client):
        cli, note = client
        response = cli.post("/api/notes/not-a-uuid/summarize")
        assert response.status_code == 422
