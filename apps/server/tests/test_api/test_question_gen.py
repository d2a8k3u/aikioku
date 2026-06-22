"""Tests for Question Generation API endpoint."""
from __future__ import annotations

import tempfile

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
            content="Python is a high-level programming language. It supports object-oriented and functional programming.",
            path="python.md",
        )
        created = store.create(note)
        app.state.note_store = store
        yield TestClient(app), created


class TestGenerateQuestions:
    def test_returns_questions(self, client):
        cli, note = client
        response = cli.get(f"/api/notes/{note.id}/questions")
        assert response.status_code == 200
        data = response.json()
        assert data["note_id"] == note.id
        assert "questions" in data
        assert isinstance(data["questions"], list)
        assert len(data["questions"]) > 0
        for q in data["questions"]:
            assert "type" in q
            assert "question" in q
            assert "answer" in q

    def test_respects_count_param(self, client):
        cli, note = client
        response = cli.get(f"/api/notes/{note.id}/questions?count=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["questions"]) <= 2

    def test_nonexistent_note_returns_404(self, client):
        cli, note = client
        response = cli.get("/api/notes/00000000-0000-0000-0000-000000000000/questions")
        assert response.status_code == 404

    def test_invalid_uuid_returns_422(self, client):
        cli, note = client
        response = cli.get("/api/notes/not-a-uuid/questions")
        assert response.status_code == 422
