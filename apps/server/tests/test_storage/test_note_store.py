"""Tests for NoteStore class."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def tmp_notes_dir(tmp_path):
    """Create a temporary notes directory."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    return str(notes_dir)


@pytest.fixture
def sample_note():
    """Create a sample Note for testing."""
    from src.models.note import Note

    return Note(
        id="12345678-1234-5678-1234-567812345678",
        title="Test Note",
        content="# Hello\n\nThis is a test note.",
        frontmatter={"tags": ["test", "demo"], "aliases": ["alias1"]},
        links=["other-note-id"],
        path="/notes/test-note.md",
        created=datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc),
        modified=datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_note2():
    """Create a second sample Note for testing."""
    from src.models.note import Note

    return Note(
        id="87654321-4321-8765-4321-876543210987",
        title="Second Note",
        content="Content about Python programming.",
        frontmatter={"tags": ["python", "coding"]},
        links=[],
        path="/notes/second-note.md",
        created=datetime(2026, 6, 11, 10, 0, 0, tzinfo=timezone.utc),
        modified=datetime(2026, 6, 11, 10, 0, 0, tzinfo=timezone.utc),
    )


class TestNoteStoreInit:
    """Test NoteStore initialization."""

    def test_creates_directory_if_not_exists(self, tmp_path):
        from src.storage.note_store import NoteStore

        notes_dir = str(tmp_path / "new_notes")
        NoteStore(notes_dir)
        assert os.path.isdir(notes_dir)

    def test_init_with_existing_directory(self, tmp_notes_dir):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        assert store.notes_dir == tmp_notes_dir


class TestNoteStoreCreate:
    """Test note creation."""

    def test_create_saves_note_as_markdown(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        result = store.create(sample_note)
        assert result.id == sample_note.id
        assert result.title == sample_note.title

    def test_create_creates_file(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        files = list(Path(tmp_notes_dir).glob("*.md"))
        assert len(files) == 1

    def test_create_file_has_yaml_frontmatter(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        files = list(Path(tmp_notes_dir).glob("*.md"))
        content = files[0].read_text()
        assert content.startswith("---\n")
        assert "id:" in content
        assert "title:" in content

    def test_create_preserves_content(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        files = list(Path(tmp_notes_dir).glob("*.md"))
        content = files[0].read_text()
        assert "# Hello" in content
        assert "This is a test note." in content

    def test_create_returns_note_object(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        result = store.create(sample_note)
        assert result.title == "Test Note"
        assert result.content == sample_note.content


class TestNoteStoreGet:
    """Test note retrieval by ID."""

    def test_get_existing_note(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        result = store.get(sample_note.id)
        assert result is not None
        assert result.id == sample_note.id
        assert result.title == sample_note.title

    def test_get_nonexistent_note_returns_none(self, tmp_notes_dir):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        result = store.get("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_get_preserves_all_fields(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        result = store.get(sample_note.id)
        assert result.title == sample_note.title
        assert result.content == sample_note.content
        assert result.path == sample_note.path
        assert result.frontmatter == sample_note.frontmatter
        assert result.links == sample_note.links


class TestNoteStoreUpdate:
    """Test note updating."""

    def test_update_modifies_note(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        sample_note.title = "Updated Title"
        result = store.update(sample_note)
        assert result.title == "Updated Title"

    def test_update_reflected_in_file(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        sample_note.title = "Updated Title"
        store.update(sample_note)
        result = store.get(sample_note.id)
        assert result.title == "Updated Title"

    def test_update_changes_modified_timestamp(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        original_modified = sample_note.modified
        sample_note.title = "Updated"
        result = store.update(sample_note)
        assert result.modified >= original_modified

    def test_update_preserves_created_timestamp(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        original_created = sample_note.created
        sample_note.title = "Updated"
        result = store.update(sample_note)
        assert result.created == original_created


class TestNoteStoreDelete:
    """Test note deletion."""

    def test_delete_existing_note(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        result = store.delete(sample_note.id)
        assert result is True

    def test_delete_removes_file(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        store.delete(sample_note.id)
        files = list(Path(tmp_notes_dir).glob("*.md"))
        assert len(files) == 0

    def test_delete_nonexistent_returns_false(self, tmp_notes_dir):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        result = store.delete("00000000-0000-0000-0000-000000000000")
        assert result is False

    def test_deleted_note_not_gettable(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        store.delete(sample_note.id)
        result = store.get(sample_note.id)
        assert result is None


class TestNoteStoreListAll:
    """Test listing all notes."""

    def test_list_all_empty(self, tmp_notes_dir):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        result = store.list_all()
        assert result == []

    def test_list_all_returns_notes(self, tmp_notes_dir, sample_note, sample_note2):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        store.create(sample_note2)
        result = store.list_all()
        assert len(result) == 2

    def test_list_all_returns_note_objects(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        result = store.list_all()
        assert result[0].title == sample_note.title

    def test_list_all_skips_empty_file(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        (Path(tmp_notes_dir) / "03511673-fa09-4ac7-b78f-f8c77338c35a.md").write_text("")
        result = store.list_all()
        assert [n.id for n in result] == [sample_note.id]

    def test_list_all_skips_malformed_file(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        (Path(tmp_notes_dir) / "deadbeef-0000-0000-0000-000000000000.md").write_text(
            "no frontmatter here\njust body text"
        )
        result = store.list_all()
        assert [n.id for n in result] == [sample_note.id]


class TestNoteStoreSearch:
    """Test full-text search."""

    def test_search_finds_matching_notes(self, tmp_notes_dir, sample_note, sample_note2):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        store.create(sample_note2)
        result = store.search("Python")
        assert len(result) == 1
        assert result[0].title == "Second Note"

    def test_search_case_insensitive(self, tmp_notes_dir, sample_note2):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note2)
        result = store.search("python")
        assert len(result) == 1

    def test_search_no_match(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        result = store.search("nonexistent")
        assert result == []

    def test_search_empty_query(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        result = store.search("")
        assert len(result) == 1

    def test_search_in_title(self, tmp_notes_dir, sample_note, sample_note2):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        store.create(sample_note2)
        result = store.search("Test Note")
        assert len(result) >= 1


class TestNoteStoreGetByTag:
    """Test filtering by tag."""

    def test_get_by_tag_finds_notes(self, tmp_notes_dir, sample_note, sample_note2):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        store.create(sample_note2)
        result = store.get_by_tag("python")
        assert len(result) == 1
        assert result[0].title == "Second Note"

    def test_get_by_tag_no_match(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        result = store.get_by_tag("nonexistent")
        assert result == []

    def test_get_by_tag_multiple_matches(self, tmp_notes_dir, sample_note, sample_note2):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        store.create(sample_note2)
        result = store.get_by_tag("test")
        assert len(result) >= 1


class TestNoteStoreGetByPath:
    """Test getting note by file path."""

    def test_get_by_path_existing(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        result = store.get_by_path(sample_note.path)
        assert result is not None
        assert result.id == sample_note.id

    def test_get_by_path_nonexistent(self, tmp_notes_dir):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        result = store.get_by_path("/notes/nonexistent.md")
        assert result is None

    def test_get_by_path_matches_correct_note(self, tmp_notes_dir, sample_note, sample_note2):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        store.create(sample_note2)
        result = store.get_by_path(sample_note2.path)
        assert result.title == "Second Note"


class TestNoteStoreHiddenNotes:
    """Hidden notes are processed + stored but excluded from user-facing reads."""

    @staticmethod
    def _hidden(tmp_notes_dir):
        from src.models.note import Note
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        visible = Note(
            id="11111111-1111-1111-1111-111111111111",
            title="Visible",
            content="visible body about apples",
            frontmatter={"tags": ["fruit"]},
            path="/notes/visible.md",
        )
        hidden = Note(
            id="22222222-2222-2222-2222-222222222222",
            title="Hidden",
            content="hidden body about apples",
            frontmatter={"tags": ["fruit"]},
            path="/notes/hidden.md",
            source_type="hidden",
        )
        store.create(visible)
        store.create(hidden)
        return store, visible, hidden

    def test_list_excludes_hidden(self, tmp_notes_dir):
        store, visible, hidden = self._hidden(tmp_notes_dir)
        ids = {n.id for n in store.list()}
        assert visible.id in ids
        assert hidden.id not in ids

    def test_list_with_explicit_source_type_returns_hidden(self, tmp_notes_dir):
        store, _visible, hidden = self._hidden(tmp_notes_dir)
        ids = {n.id for n in store.list(source_type="hidden")}
        assert ids == {hidden.id}

    def test_count_excludes_hidden(self, tmp_notes_dir):
        store, _visible, _hidden = self._hidden(tmp_notes_dir)
        assert store.count() == 1

    def test_search_excludes_hidden(self, tmp_notes_dir):
        store, visible, hidden = self._hidden(tmp_notes_dir)
        ids = {n.id for n in store.search("apples")}
        assert visible.id in ids
        assert hidden.id not in ids

    def test_get_by_tag_excludes_hidden(self, tmp_notes_dir):
        store, visible, hidden = self._hidden(tmp_notes_dir)
        ids = {n.id for n in store.get_by_tag("fruit")}
        assert visible.id in ids
        assert hidden.id not in ids

    def test_get_by_id_returns_hidden(self, tmp_notes_dir):
        store, _visible, hidden = self._hidden(tmp_notes_dir)
        assert store.get(hidden.id) is not None


class TestNoteStoreAtomicWrite:
    """Writes must be crash-safe: never leave a partial/0-byte or stray temp file."""

    def test_create_leaves_no_temp_file(self, tmp_notes_dir, sample_note):
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        assert list(Path(tmp_notes_dir).glob("*.tmp")) == []
        assert list(Path(tmp_notes_dir).glob(".*.tmp")) == []

    def test_failed_write_keeps_old_file_and_no_temp(self, tmp_notes_dir, sample_note, monkeypatch):
        import src.storage.note_store as ns
        from src.storage.note_store import NoteStore

        store = NoteStore(tmp_notes_dir)
        store.create(sample_note)
        original = store.get(sample_note.id).content

        def _boom(*_args, **_kwargs):
            raise OSError("simulated crash during rename")

        monkeypatch.setattr(ns.os, "replace", _boom)
        sample_note.content = "THIS MUST NOT PERSIST"
        with pytest.raises(OSError):
            store.update(sample_note)

        assert store.get(sample_note.id).content == original
        assert list(Path(tmp_notes_dir).glob(".*.tmp")) == []
