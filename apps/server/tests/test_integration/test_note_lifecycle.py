"""Integration tests for Note lifecycle: CRUD + search + frontmatter.

Uses real file-based NoteStore in a temp directory.
"""

from __future__ import annotations

import tempfile

import pytest

from src.models.note import Note
from src.storage.note_store import NoteStore


@pytest.fixture
def note_store():
    """Create a NoteStore backed by a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield NoteStore(notes_dir=tmpdir)


@pytest.fixture
def sample_note():
    return Note(
        id="11111111-1111-1111-1111-111111111111",
        title="Python Guide",
        content="Python is a versatile programming language.",
        frontmatter={"tags": ["python", "coding"], "aliases": ["py"]},
        links=["22222222-2222-2222-2222-222222222222"],
        path="/notes/python-guide.md",
    )


@pytest.fixture
def second_note():
    return Note(
        id="22222222-2222-2222-2222-222222222222",
        title="Rust Guide",
        content="Rust is a systems programming language focused on safety.",
        frontmatter={"tags": ["rust", "coding"], "aliases": ["rs"]},
        links=[],
        path="/notes/rust-guide.md",
    )


@pytest.fixture
def third_note():
    return Note(
        id="33333333-3333-3333-3333-333333333333",
        title="Machine Learning",
        content="ML uses Python extensively for data science and neural networks.",
        frontmatter={"tags": ["ml", "python"], "aliases": []},
        links=[],
        path="/notes/ml.md",
    )


class TestCreateNoteAndRetrieve:
    def test_create_note_and_retrieve(self, note_store, sample_note):
        note_store.create(sample_note)
        result = note_store.get(sample_note.id)
        assert result is not None
        assert result.id == sample_note.id
        assert result.title == "Python Guide"
        assert result.content == "Python is a versatile programming language."
        assert result.path == "/notes/python-guide.md"
        assert result.links == ["22222222-2222-2222-2222-222222222222"]
        assert result.frontmatter == {"tags": ["python", "coding"], "aliases": ["py"]}


class TestUpdateNote:
    def test_update_note(self, note_store, sample_note):
        note_store.create(sample_note)
        original_modified = sample_note.modified

        sample_note.title = "Updated Python Guide"
        sample_note.content = "Updated content about Python."
        result = note_store.update(sample_note)

        assert result.title == "Updated Python Guide"
        assert result.modified >= original_modified

        # Verify persisted
        retrieved = note_store.get(sample_note.id)
        assert retrieved.title == "Updated Python Guide"
        assert retrieved.content == "Updated content about Python."


class TestDeleteNote:
    def test_delete_note(self, note_store, sample_note):
        note_store.create(sample_note)
        assert note_store.get(sample_note.id) is not None

        deleted = note_store.delete(sample_note.id)
        assert deleted is True
        assert note_store.get(sample_note.id) is None


class TestSearchNotes:
    def test_search_notes(self, note_store, sample_note, second_note, third_note):
        note_store.create(sample_note)
        note_store.create(second_note)
        note_store.create(third_note)

        results = note_store.search("Python")
        titles = [n.title for n in results]
        assert "Python Guide" in titles
        assert "Machine Learning" in titles
        assert "Rust Guide" not in titles

    def test_search_case_insensitive(self, note_store, second_note):
        note_store.create(second_note)
        results = note_store.search("rust")
        assert len(results) == 1
        assert results[0].title == "Rust Guide"

    def test_search_no_match(self, note_store, sample_note):
        note_store.create(sample_note)
        results = note_store.search("nonexistent")
        assert results == []


class TestNoteWithFrontmatter:
    def test_note_with_frontmatter(self, note_store, sample_note):
        note_store.create(sample_note)
        result = note_store.get(sample_note.id)

        assert result.frontmatter["tags"] == ["python", "coding"]
        assert result.frontmatter["aliases"] == ["py"]

    def test_note_without_frontmatter(self, note_store):
        note = Note(
            id="44444444-4444-4444-4444-444444444444",
            title="Plain Note",
            content="No frontmatter here.",
            path="/notes/plain.md",
        )
        note_store.create(note)
        result = note_store.get(note.id)
        # When no tags/aliases are set, they serialize as empty lists
        assert result.frontmatter.get("tags", []) == []
        assert result.frontmatter.get("aliases", []) == []
