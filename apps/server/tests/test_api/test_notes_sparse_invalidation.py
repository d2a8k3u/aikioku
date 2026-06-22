"""PHASE 6b: creating/updating/deleting a note must invalidate the shared
sparse index so the change is reflected in the next search.

The notes router calls ``app.state.hybrid_fusion.sparse.mark_dirty()`` on
create/update/delete (guarded with getattr so it is a no-op when fusion is
unset). These tests assert the wiring via a spy on the shared sparse instance.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.storage.note_store import NoteStore


@pytest.fixture
def wired_client(tmp_path, monkeypatch):
    """TestClient with a temp NoteStore and a spyable sparse on app.state.

    Background entity-extraction and embedding tasks are stubbed so the test
    isolates the sparse-invalidation wiring.
    """
    from src.main import app
    from src.api import notes as notes_mod

    notes_dir = str(tmp_path / "notes")
    store = NoteStore(notes_dir)
    app.state.note_store = store
    notes_mod._note_store = store

    # Stub the fire-and-forget background tasks.
    monkeypatch.setattr(notes_mod, "_extract_and_store_entities", lambda *a, **k: None)
    monkeypatch.setattr(notes_mod, "_store_note_embeddings", lambda *a, **k: None)

    # Install a fusion stub whose sparse exposes a spyable mark_dirty.
    fusion = MagicMock()
    fusion.sparse.mark_dirty = MagicMock()
    app.state.hybrid_fusion = fusion

    yield TestClient(app), store, fusion.sparse

    notes_mod._note_store = None
    for attr in ("note_store", "hybrid_fusion"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


_NOTE = {"title": "Docker Note", "content": "containers and zorptastic tokens", "path": "n.md"}


def test_create_note_marks_sparse_dirty(wired_client):
    client, store, sparse = wired_client
    resp = client.post("/api/notes/", json=_NOTE)
    assert resp.status_code == 200
    sparse.mark_dirty.assert_called()


def test_update_note_marks_sparse_dirty(wired_client):
    client, store, sparse = wired_client
    created = client.post("/api/notes/", json=_NOTE).json()
    sparse.mark_dirty.reset_mock()

    note_id = created["id"]
    resp = client.put(f"/api/notes/{note_id}", json={"content": "updated body"})
    assert resp.status_code == 200
    sparse.mark_dirty.assert_called()


def test_delete_note_marks_sparse_dirty(wired_client):
    client, store, sparse = wired_client
    created = client.post("/api/notes/", json=_NOTE).json()
    sparse.mark_dirty.reset_mock()

    note_id = created["id"]
    resp = client.delete(f"/api/notes/{note_id}")
    assert resp.status_code == 200
    sparse.mark_dirty.assert_called()


def test_create_note_without_fusion_does_not_error(tmp_path, monkeypatch):
    """When app.state.hybrid_fusion is unset, create still succeeds (getattr guard)."""
    from src.main import app
    from src.api import notes as notes_mod

    notes_dir = str(tmp_path / "notes")
    store = NoteStore(notes_dir)
    app.state.note_store = store
    notes_mod._note_store = store
    monkeypatch.setattr(notes_mod, "_extract_and_store_entities", lambda *a, **k: None)
    monkeypatch.setattr(notes_mod, "_store_note_embeddings", lambda *a, **k: None)
    if hasattr(app.state, "hybrid_fusion"):
        delattr(app.state, "hybrid_fusion")

    try:
        client = TestClient(app)
        resp = client.post("/api/notes/", json=_NOTE)
        assert resp.status_code == 200
    finally:
        notes_mod._note_store = None
        if hasattr(app.state, "note_store"):
            delattr(app.state, "note_store")
