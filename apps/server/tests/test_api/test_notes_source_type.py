"""Tests for the source_type field distinguishing user notes from MCP memories.

BOD 1: Notes vs Memories distinction.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.models.note import Note
from src.storage.note_store import NoteStore


def _make_note(suffix: str, *, modified=None) -> Note:
    nid = f"00000000-0000-0000-0000-{int(suffix):012d}"
    base = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    return Note(
        id=nid,
        title=f"Note {suffix}",
        content=f"Body {suffix}",
        frontmatter={"tags": []},
        links=[],
        path=f"/notes/note-{suffix}.md",
        created=base,
        modified=modified or base,
    )


@pytest.fixture
def wired_client(tmp_path):
    """A TestClient whose notes router uses a real temp-dir NoteStore."""
    from src.main import app
    from src.api import notes as notes_mod

    notes_dir = str(tmp_path / "notes")
    store = NoteStore(notes_dir)
    app.state.note_store = store
    notes_mod._note_store = store
    yield TestClient(app), store
    notes_mod._note_store = None
    if hasattr(app.state, "note_store"):
        delattr(app.state, "note_store")


class TestUserCreatedNoteHasSourceTypeNote:
    """User-created notes (via REST API) should have source_type='note'."""

    def test_create_note_via_api_has_source_type_note(self, wired_client):
        client, store = wired_client
        note = _make_note("1")
        resp = client.post("/api/notes/", json=note.model_dump(mode="json"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_type"] == "note", f"expected 'note', got {body.get('source_type')}"

    def test_note_persisted_to_disk_has_source_type_in_frontmatter(self, wired_client):
        client, store = wired_client
        note = _make_note("2")
        resp = client.post("/api/notes/", json=note.model_dump(mode="json"))
        assert resp.status_code == 200
        # Read the file directly to verify frontmatter serialization
        file_path = store._file_path(note.id)
        content = file_path.read_text(encoding="utf-8")
        assert "source_type: note" in content, f"frontmatter missing source_type:\n{content}"

    def test_note_survives_roundtrip_through_disk(self, wired_client):
        client, store = wired_client
        note = _make_note("3")
        resp = client.post("/api/notes/", json=note.model_dump(mode="json"))
        assert resp.status_code == 200
        # Re-read from disk via store.get()
        reloaded = store.get(note.id)
        assert reloaded is not None
        assert reloaded.source_type == "note", f"roundtrip lost source_type: {reloaded.source_type}"


class TestListNotesSourceTypeFilter:
    """list_notes endpoint should support filtering by source_type."""

    def test_list_notes_filter_by_source_type_note(self, wired_client):
        client, store = wired_client
        # Create a user note (source_type="note")
        note1 = _make_note("10")
        store.create(note1)
        # Create a memory note (source_type="memory") directly via store
        note2 = Note(
            id="00000000-0000-0000-0000-000000000020",
            title="Memory 20",
            content="Memory body",
            frontmatter={"tags": []},
            links=[],
            path="/notes/memory-20.md",
            source_type="memory",
            created=datetime(2026, 6, 1, tzinfo=timezone.utc),
            modified=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        store.create(note2)

        # Filter by source_type=note
        resp = client.get("/api/notes/?source_type=note")
        assert resp.status_code == 200
        body = resp.json()
        ids = {n["id"] for n in body}
        assert note1.id in ids
        assert note2.id not in ids, "memory note should not appear in note filter"

    def test_list_notes_filter_by_source_type_memory(self, wired_client):
        client, store = wired_client
        note1 = _make_note("11")
        store.create(note1)
        note2 = Note(
            id="00000000-0000-0000-0000-000000000021",
            title="Memory 21",
            content="Memory body",
            frontmatter={"tags": []},
            links=[],
            path="/notes/memory-21.md",
            source_type="memory",
            created=datetime(2026, 6, 1, tzinfo=timezone.utc),
            modified=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        store.create(note2)

        resp = client.get("/api/notes/?source_type=memory")
        assert resp.status_code == 200
        body = resp.json()
        ids = {n["id"] for n in body}
        assert note2.id in ids
        assert note1.id not in ids, "user note should not appear in memory filter"

    def test_list_notes_without_source_type_returns_all(self, wired_client):
        client, store = wired_client
        note1 = _make_note("12")
        store.create(note1)
        note2 = Note(
            id="00000000-0000-0000-0000-000000000022",
            title="Memory 22",
            content="Memory body",
            frontmatter={"tags": []},
            links=[],
            path="/notes/memory-22.md",
            source_type="memory",
            created=datetime(2026, 6, 1, tzinfo=timezone.utc),
            modified=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        store.create(note2)

        resp = client.get("/api/notes/")
        assert resp.status_code == 200
        body = resp.json()
        ids = {n["id"] for n in body}
        assert note1.id in ids
        assert note2.id in ids


def _make_hidden(suffix: str) -> Note:
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    return Note(
        id=f"00000000-0000-0000-0000-9{int(suffix):011d}",
        title=f"Hidden {suffix}",
        content=f"Hidden body {suffix}",
        frontmatter={"tags": []},
        links=[],
        path=f"/notes/hidden-{suffix}.md",
        source_type="hidden",
        created=base,
        modified=base,
    )


class TestHiddenNotesExcludedFromUiList:
    """Hidden notes are retrievable internally but absent from the user-facing list."""

    def test_hidden_note_absent_from_default_list(self, wired_client):
        client, store = wired_client
        visible = _make_note("50")
        store.create(visible)
        hidden = _make_hidden("50")
        store.create(hidden)

        resp = client.get("/api/notes/")
        assert resp.status_code == 200
        ids = {n["id"] for n in resp.json()}
        assert visible.id in ids
        assert hidden.id not in ids, "hidden note must not appear in the default list"

    def test_hidden_note_present_with_explicit_filter(self, wired_client):
        client, store = wired_client
        hidden = _make_hidden("51")
        store.create(hidden)

        resp = client.get("/api/notes/?source_type=hidden")
        assert resp.status_code == 200
        ids = {n["id"] for n in resp.json()}
        assert hidden.id in ids

    def test_hidden_note_still_fetchable_by_id(self, wired_client):
        client, store = wired_client
        hidden = _make_hidden("52")
        store.create(hidden)

        resp = client.get(f"/api/notes/{hidden.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == hidden.id


class TestReindexPreservesSourceType:
    """Reindex must preserve source_type from YAML frontmatter."""

    def test_reindex_restores_source_type(self, wired_client):
        client, store = wired_client
        note = Note(
            id="00000000-0000-0000-0000-000000000030",
            title="Memory 30",
            content="Memory body",
            frontmatter={"tags": []},
            links=[],
            path="/notes/memory-30.md",
            source_type="memory",
            created=datetime(2026, 6, 1, tzinfo=timezone.utc),
            modified=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )
        store.create(note)

        # Force reindex
        store.reindex()

        # Read back
        reloaded = store.get(note.id)
        assert reloaded is not None
        assert reloaded.source_type == "memory", f"reindex lost source_type: {reloaded.source_type}"


class TestBackwardCompatibility:
    """Old notes without source_type in frontmatter should default to 'note'."""

    def test_old_note_without_source_type_defaults_to_note(self, wired_client):
        client, store = wired_client
        # Write a markdown file without source_type in frontmatter
        note_id = "00000000-0000-0000-0000-000000000040"
        file_path = store._file_path(note_id)
        old_md = """---
id: 00000000-0000-0000-0000-000000000040
title: Old Note
tags: []
aliases: []
links: []
path: /notes/old-note.md
created: '2026-06-01T00:00:00+00:00'
modified: '2026-06-01T00:00:00+00:00'
---
Old content without source_type."""
        file_path.write_text(old_md, encoding="utf-8")

        # Reindex to pick it up
        store.reindex()

        # Read back
        reloaded = store.get(note_id)
        assert reloaded is not None
        assert reloaded.source_type == "note", (
            f"old note should default to 'note', got {reloaded.source_type}"
        )
