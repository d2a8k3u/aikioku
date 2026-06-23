"""Tests for the persistent SQLite metadata index in NoteStore.

These verify the optimization layer: list/count/tag/path operations are served
from a SQLite index of note metadata, so the markdown corpus is NOT fully
scanned on every request. Markdown files remain the source of truth; the index
is a cache that is kept in sync incrementally and repaired by reindex().
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models.note import Note
from src.storage.note_store import NoteStore


def _make_note(suffix: str, *, tags=None, modified=None, title=None) -> Note:
    """Build a Note with deterministic, distinguishable fields."""
    nid = f"00000000-0000-0000-0000-{int(suffix):012d}"
    base = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    return Note(
        id=nid,
        title=title or f"Note {suffix}",
        content=f"Body of note {suffix}.",
        frontmatter={"tags": tags or []},
        links=[],
        path=f"/notes/note-{suffix}.md",
        created=base,
        modified=modified or base,
    )


@pytest.fixture
def tmp_notes_dir(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    return str(notes_dir)


# --------------------------------------------------------------------------- #
# 1. Sync: create/update/delete keep the index row in lockstep with disk.
# --------------------------------------------------------------------------- #


class TestIndexSync:
    def test_create_inserts_index_row(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        note = _make_note("1", tags=["alpha"])
        store.create(note)

        rows = _read_index(store)
        assert len(rows) == 1
        assert rows[0]["id"] == note.id
        assert rows[0]["path"] == note.path
        assert rows[0]["title"] == note.title

    def test_update_updates_index_row(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        note = _make_note("1", title="Original")
        store.create(note)
        note.title = "Renamed"
        store.update(note)

        rows = _read_index(store)
        assert len(rows) == 1
        assert rows[0]["title"] == "Renamed"

    def test_delete_removes_index_row(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        note = _make_note("1")
        store.create(note)
        store.delete(note.id)

        assert _read_index(store) == []
        assert store.count() == 0


# --------------------------------------------------------------------------- #
# 2. Count is O(1) from the index and matches reality.
# --------------------------------------------------------------------------- #


class TestCount:
    def test_count_matches_number_of_notes(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        for i in range(1, 6):
            store.create(_make_note(str(i)))
        assert store.count() == 5
        assert store.count() == len(store.list_all())

    def test_count_empty(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        assert store.count() == 0


# --------------------------------------------------------------------------- #
# 3. Pagination: list(skip, limit) returns the most-recently-modified page and
#    reads ONLY the page's files (instrumented read counter).
# --------------------------------------------------------------------------- #


class TestPagination:
    def test_list_returns_most_recent_first(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        # modified ascending with index -> newest is "10"
        for i in range(1, 11):
            store.create(
                _make_note(
                    str(i),
                    modified=datetime(2026, 6, i, 0, 0, 0, tzinfo=timezone.utc),
                )
            )
        page = store.list(skip=0, limit=3)
        assert [n.title for n in page] == ["Note 10", "Note 9", "Note 8"]

    def test_list_respects_skip(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        for i in range(1, 11):
            store.create(
                _make_note(
                    str(i),
                    modified=datetime(2026, 6, i, 0, 0, 0, tzinfo=timezone.utc),
                )
            )
        page = store.list(skip=2, limit=2)
        assert [n.title for n in page] == ["Note 8", "Note 7"]

    def test_list_reads_only_page_files(self, tmp_notes_dir, monkeypatch):
        store = NoteStore(tmp_notes_dir)
        for i in range(1, 21):
            store.create(
                _make_note(
                    str(i),
                    modified=datetime(2026, 6, (i % 28) + 1, 0, 0, 0, tzinfo=timezone.utc),
                )
            )

        reads = _count_markdown_reads(monkeypatch)
        page = store.list(skip=0, limit=5)
        assert len(page) == 5
        # Only the 5 page files should be read, not all 20.
        assert reads["count"] <= 6, f"read {reads['count']} files for a 5-item page"


# --------------------------------------------------------------------------- #
# 4. get_by_tag / get_by_path use the index and read only matching files.
# --------------------------------------------------------------------------- #


class TestTagAndPathUseIndex:
    def test_get_by_tag_returns_correct_notes(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        store.create(_make_note("1", tags=["python", "coding"]))
        store.create(_make_note("2", tags=["rust"]))
        store.create(_make_note("3", tags=["python"]))

        result = store.get_by_tag("python")
        assert {n.title for n in result} == {"Note 1", "Note 3"}

    def test_get_by_tag_case_insensitive(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        store.create(_make_note("1", tags=["Python"]))
        result = store.get_by_tag("python")
        assert len(result) == 1

    def test_get_by_tag_reads_only_matching_files(self, tmp_notes_dir, monkeypatch):
        store = NoteStore(tmp_notes_dir)
        for i in range(1, 21):
            store.create(_make_note(str(i), tags=["common"]))
        # Exactly one note has the rare tag.
        store.create(_make_note("99", tags=["rare"]))

        reads = _count_markdown_reads(monkeypatch)
        result = store.get_by_tag("rare")
        assert len(result) == 1
        assert result[0].title == "Note 99"
        assert reads["count"] <= 2, f"read {reads['count']} files for a 1-match tag"

    def test_get_by_path_returns_correct_note(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        store.create(_make_note("1"))
        target = _make_note("2")
        store.create(target)

        result = store.get_by_path(target.path)
        assert result is not None
        assert result.id == target.id

    def test_get_by_path_reads_only_one_file(self, tmp_notes_dir, monkeypatch):
        store = NoteStore(tmp_notes_dir)
        for i in range(1, 21):
            store.create(_make_note(str(i)))
        target = _make_note("99")
        store.create(target)

        reads = _count_markdown_reads(monkeypatch)
        result = store.get_by_path(target.path)
        assert result is not None
        assert reads["count"] <= 1, f"read {reads['count']} files for a single path lookup"

    def test_get_by_path_missing_returns_none(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        store.create(_make_note("1"))
        assert store.get_by_path("/notes/does-not-exist.md") is None


# --------------------------------------------------------------------------- #
# 5. Reindex: rebuild from disk when index is missing or out of sync.
# --------------------------------------------------------------------------- #


class TestReindex:
    def test_first_use_builds_index_from_existing_files(self, tmp_notes_dir):
        # Seed the directory using one store, then drop the index file to
        # simulate a fresh deploy where only the markdown files exist.
        seed = NoteStore(tmp_notes_dir)
        for i in range(1, 4):
            seed.create(_make_note(str(i)))
        index_path = seed.index_db_path
        Path(index_path).unlink()

        # A brand-new store over the same dir must self-populate the index.
        store = NoteStore(tmp_notes_dir)
        store.reindex()
        assert store.count() == 3
        assert len(_read_index(store)) == 3

    def test_lazy_reindex_on_count_when_index_empty(self, tmp_notes_dir):
        seed = NoteStore(tmp_notes_dir)
        for i in range(1, 4):
            seed.create(_make_note(str(i)))
        Path(seed.index_db_path).unlink()

        store = NoteStore(tmp_notes_dir)
        # No explicit reindex: the first count must detect the mismatch and repair.
        assert store.count() == 3

    def test_reindex_repairs_stale_extra_row(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        store.create(_make_note("1"))
        # Inject a bogus row that has no backing file on disk.
        conn = sqlite3.connect(store.index_db_path)
        conn.execute(
            "INSERT INTO note_index (id, path, title, tags, created, modified, mtime) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ghost", "/notes/ghost.md", "Ghost", "[]", "", "", 0.0),
        )
        conn.commit()
        conn.close()

        store.reindex()
        ids = {r["id"] for r in _read_index(store)}
        assert "ghost" not in ids
        assert store.count() == 1

    def test_reindex_recovers_missing_row(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        store.create(_make_note("1"))
        note2 = _make_note("2")
        store.create(note2)
        # Delete the index row for note2 but keep its file -> index undercount.
        conn = sqlite3.connect(store.index_db_path)
        conn.execute("DELETE FROM note_index WHERE id = ?", (note2.id,))
        conn.commit()
        conn.close()

        store.reindex()
        assert store.count() == 2
        ids = {r["id"] for r in _read_index(store)}
        assert note2.id in ids

    def test_reindex_is_idempotent(self, tmp_notes_dir):
        store = NoteStore(tmp_notes_dir)
        for i in range(1, 4):
            store.create(_make_note(str(i)))
        store.reindex()
        store.reindex()
        assert store.count() == 3


# --------------------------------------------------------------------------- #
# 6. Scale smoke test: ~1000 notes, list(limit=50) reads ~50 files, count fast.
# --------------------------------------------------------------------------- #


@pytest.mark.slow
class TestScale:
    def test_list_of_50_reads_about_50_files_at_1000_notes(self, tmp_notes_dir, monkeypatch):
        store = NoteStore(tmp_notes_dir)
        n = 1000
        for i in range(1, n + 1):
            store.create(
                _make_note(
                    str(i),
                    modified=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).replace(
                        second=i % 60
                    ),
                )
            )

        assert store.count() == n

        reads = _count_markdown_reads(monkeypatch)
        page = store.list(skip=0, limit=50)
        assert len(page) == 50
        # A 50-item page must not scan 1000 files.
        assert reads["count"] <= 55, f"read {reads['count']} files for a 50-item page out of {n}"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _read_index(store: NoteStore) -> list[dict]:
    conn = sqlite3.connect(store.index_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, path, title, tags, created, modified, mtime FROM note_index"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _count_markdown_reads(monkeypatch) -> dict:
    """Patch Path.read_text to count how many markdown files get read.

    Returns a mutable dict whose ``count`` field is incremented on each
    .md read. Install AFTER seeding so only the measured operation counts.
    """
    counter = {"count": 0}
    original = Path.read_text

    def _counting_read_text(self, *args, **kwargs):
        if str(self).endswith(".md"):
            counter["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _counting_read_text)
    return counter
