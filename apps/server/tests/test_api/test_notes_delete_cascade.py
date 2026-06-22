"""Deleting a note must cascade to the embedding store (and graph), and must never
500 the request if a downstream store fails — the note file/index is already gone.

Spies on ``app.state.embedding_store`` assert the wiring; the graph is left unset
so these tests isolate the embedding cascade and the error-isolation guarantee.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.storage.note_store import NoteStore

_NOTE = {"title": "Cascade Note", "content": "zorptastic tokens", "path": "n.md"}


@pytest.fixture
def wired_client(tmp_path, monkeypatch):
    from src.main import app
    from src.api import notes as notes_mod

    store = NoteStore(str(tmp_path / "notes"))
    app.state.note_store = store
    notes_mod._note_store = store

    monkeypatch.setattr(notes_mod, "_extract_and_store_entities", lambda *a, **k: None)
    monkeypatch.setattr(notes_mod, "_store_note_embeddings", lambda *a, **k: None)

    fusion = MagicMock()
    fusion.sparse.mark_dirty = MagicMock()
    app.state.hybrid_fusion = fusion

    embedding_store = MagicMock()
    app.state.embedding_store = embedding_store
    if hasattr(app.state, "knowledge_graph"):
        delattr(app.state, "knowledge_graph")

    yield TestClient(app), store, embedding_store, fusion.sparse

    notes_mod._note_store = None
    for attr in ("note_store", "hybrid_fusion", "embedding_store"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def test_delete_calls_embedding_delete(wired_client):
    client, store, embedding_store, sparse = wired_client
    note_id = client.post("/api/notes/", json=_NOTE).json()["id"]

    resp = client.delete(f"/api/notes/{note_id}")

    assert resp.status_code == 200
    embedding_store.delete.assert_called_once_with(note_id)


def test_delete_with_stores_unset_still_succeeds(wired_client):
    client, store, embedding_store, sparse = wired_client
    from src.main import app

    note_id = client.post("/api/notes/", json=_NOTE).json()["id"]
    delattr(app.state, "embedding_store")  # both stores now unset → cascade no-op
    sparse.mark_dirty.reset_mock()

    resp = client.delete(f"/api/notes/{note_id}")

    assert resp.status_code == 200
    sparse.mark_dirty.assert_called()


def test_delete_when_embedding_delete_raises_still_succeeds(wired_client):
    client, store, embedding_store, sparse = wired_client
    embedding_store.delete.side_effect = RuntimeError("chroma down")
    note_id = client.post("/api/notes/", json=_NOTE).json()["id"]

    resp = client.delete(f"/api/notes/{note_id}")

    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    assert store.get(note_id) is None
