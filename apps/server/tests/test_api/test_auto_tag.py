"""Tests for the auto-tag API endpoint — generated tags must persist to the note."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.augmentation.auto_tag import AutoTagger
from src.models.note import Note
from src.storage.note_store import NoteStore


@pytest.fixture
def store(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    return NoteStore(str(notes_dir))


@pytest.fixture
def client(store):
    from src.main import app

    prev_store = getattr(app.state, "note_store", None)
    prev_tagger = getattr(app.state, "auto_tagger", None)
    app.state.note_store = store
    # llm_provider=None -> rule-based tags only, no real LLM call in tests.
    app.state.auto_tagger = AutoTagger(llm_provider=None)
    try:
        yield TestClient(app)
    finally:
        app.state.note_store = prev_store
        app.state.auto_tagger = prev_tagger


def _seed_note(store: NoteStore) -> Note:
    note = Note(
        title="Tagging target",
        content="This note covers #python and #testing.",
        path="tagging-target.md",
    )
    store.create(note)
    return note


class TestAutoTagPersistence:
    """POST /api/tags/auto/{id} must write generated tags, not just return them."""

    def test_generated_tags_persisted_to_note(self, client, store):
        note = _seed_note(store)

        response = client.post(f"/api/tags/auto/{note.id}")

        assert response.status_code == 200
        returned = response.json()["tags"]
        assert returned, "endpoint returned no tags"
        assert "python" in returned and "testing" in returned

        reread = store.get(note.id)
        assert reread is not None
        assert reread.frontmatter["tags"] == returned

    def test_persisted_tags_are_findable_by_tag_filter(self, client, store):
        note = _seed_note(store)

        client.post(f"/api/tags/auto/{note.id}")

        matches = store.get_by_tag("python")
        assert [n.id for n in matches] == [note.id]

    def test_missing_note_returns_404(self, client):
        response = client.post("/api/tags/auto/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404


class TestAutoTagAuthConsistency:
    """The auto-tag route mutates state, so it carries require_auth like PUT /api/notes/."""

    def test_works_without_token_when_auth_not_required(self, client, store):
        note = _seed_note(store)
        response = client.post(f"/api/tags/auto/{note.id}")
        assert response.status_code == 200

    def test_route_declares_require_auth_dependency(self):
        from src.api import auto_tag
        from src.auth import require_auth

        route = next(
            r
            for r in auto_tag.router.routes
            if getattr(r, "path", None) == "/api/tags/auto/{note_id}"
        )
        dep_calls = [d.call for d in route.dependant.dependencies]
        assert require_auth in dep_calls
