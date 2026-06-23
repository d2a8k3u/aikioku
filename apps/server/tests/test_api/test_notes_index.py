"""API-level tests proving the notes list/stats endpoints use the metadata index.

The list endpoint must page via NoteStore.list (reading only
the page's files), and /api/stats must count via the O(1) index count rather
than scanning the whole corpus.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
    """A TestClient whose notes router and stats use a real temp-dir NoteStore."""
    from src.main import app
    from src.api import notes as notes_mod

    notes_dir = str(tmp_path / "notes")
    store = NoteStore(notes_dir)
    app.state.note_store = store
    # The list/get/delete endpoints use the module-level singleton.
    notes_mod._note_store = store
    yield TestClient(app), store
    notes_mod._note_store = None
    if hasattr(app.state, "note_store"):
        delattr(app.state, "note_store")


class TestListEndpointPaginates:
    def test_list_returns_at_most_limit(self, wired_client):
        client, store = wired_client
        for i in range(1, 11):
            store.create(
                _make_note(str(i), modified=datetime(2026, 6, i, tzinfo=timezone.utc))
            )
        resp = client.get("/api/notes/?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 5

    def test_list_reads_only_page_files(self, wired_client, monkeypatch):
        client, store = wired_client
        for i in range(1, 21):
            store.create(
                _make_note(str(i), modified=datetime(2026, 6, (i % 28) + 1, tzinfo=timezone.utc))
            )

        counter = {"count": 0}
        original = Path.read_text

        def _counting(self, *a, **k):
            if str(self).endswith(".md"):
                counter["count"] += 1
            return original(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", _counting)
        resp = client.get("/api/notes/?limit=5")
        assert resp.status_code == 200
        assert len(resp.json()) == 5
        assert counter["count"] <= 6, f"list endpoint read {counter['count']} files for limit=5"

    def test_list_skip_and_limit(self, wired_client):
        client, store = wired_client
        for i in range(1, 11):
            store.create(
                _make_note(str(i), modified=datetime(2026, 6, i, tzinfo=timezone.utc))
            )
        resp = client.get("/api/notes/?skip=2&limit=2")
        assert resp.status_code == 200
        titles = [n["title"] for n in resp.json()]
        assert titles == ["Note 8", "Note 7"]


class TestStatsUsesIndexCount:
    def test_stats_note_count_does_not_scan_corpus(self, wired_client, monkeypatch):
        client, store = wired_client
        for i in range(1, 11):
            store.create(_make_note(str(i)))

        counter = {"count": 0}
        original = Path.read_text

        def _counting(self, *a, **k):
            if str(self).endswith(".md"):
                counter["count"] += 1
            return original(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", _counting)
        # The non-note count helpers need services this lightweight fixture does
        # not stand up; stub them so we isolate the note-count path under test.
        with (
            patch("src.api.stats._get_entity_count", new_callable=AsyncMock, return_value=0),
            patch("src.api.stats._get_relation_count", new_callable=AsyncMock, return_value=0),
            patch("src.api.stats._get_memory_count", new_callable=AsyncMock, return_value=0),
            patch("src.api.stats._get_card_count", new_callable=AsyncMock, return_value=0),
        ):
            resp = client.get("/api/stats/")
        assert resp.status_code == 200
        assert resp.json()["notes"] == 10
        # O(1) count -> no markdown files read at all.
        assert counter["count"] == 0, f"stats read {counter['count']} markdown files"
