"""Tests for Note model."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError


class TestNoteCreation:
    """Test basic Note creation with required and default fields."""

    def test_create_note_with_required_fields(self):
        from src.models.note import Note

        note = Note(
            title="My Note",
            content="Some content here",
            path="/notes/my-note.md",
        )
        assert note.title == "My Note"
        assert note.content == "Some content here"
        assert note.path == "/notes/my-note.md"

    def test_id_is_auto_generated_uuid4(self):
        from src.models.note import Note

        note = Note(title="Test", content="Body", path="/test.md")
        parsed = uuid.UUID(note.id)
        assert parsed.version == 4

    def test_frontmatter_defaults_to_empty_dict(self):
        from src.models.note import Note

        note = Note(title="Test", content="Body", path="/test.md")
        assert note.frontmatter == {}

    def test_links_defaults_to_empty_list(self):
        from src.models.note import Note

        note = Note(title="Test", content="Body", path="/test.md")
        assert note.links == []

    def test_created_and_modified_default_to_utcnow(self):
        from src.models.note import Note

        before = datetime.utcnow()
        note = Note(title="Test", content="Body", path="/test.md")
        after = datetime.utcnow()
        assert before <= note.created <= after
        assert before <= note.modified <= after


class TestNoteWithAllFields:
    """Test Note creation with all fields specified."""

    def test_create_note_with_all_fields(self, fixed_uuid, fixed_uuid2, fixed_datetime):
        from src.models.note import Note

        note = Note(
            id=fixed_uuid,
            title="Full Note",
            content="Full content",
            frontmatter={"tags": ["test"]},
            links=["other-note"],
            path="/notes/full.md",
            created=fixed_datetime,
            modified=fixed_datetime,
        )
        assert note.id == fixed_uuid
        assert note.frontmatter == {"tags": ["test"]}
        assert note.links == ["other-note"]
        assert note.created == fixed_datetime
        assert note.modified == fixed_datetime


class TestNoteValidation:
    """Test Note validation rules."""

    def test_empty_title_raises_error(self):
        from src.models.note import Note

        with pytest.raises(ValidationError):
            Note(title="", content="Body", path="/test.md")

    def test_empty_content_is_allowed(self):
        from src.models.note import Note

        note = Note(title="Title", content="", path="/test.md")
        assert note.content == ""

    def test_empty_path_raises_error(self):
        from src.models.note import Note

        with pytest.raises(ValidationError):
            Note(title="Title", content="Body", path="")

    def test_dict_is_serializable(self):
        from src.models.note import Note

        note = Note(
            title="Test",
            content="Body",
            frontmatter={"key": "value"},
            links=["link1"],
            path="/test.md",
        )
        data = note.model_dump()
        assert data["title"] == "Test"
        assert data["frontmatter"] == {"key": "value"}
        assert data["links"] == ["link1"]
