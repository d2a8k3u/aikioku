"""Tests for Import/Export API endpoints and file_import helpers."""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helper-level tests (no HTTP)
# ---------------------------------------------------------------------------


class TestParseMarkdownFile:
    """Test parse_markdown_file creates Note with correct fields."""

    def test_parses_basic_frontmatter_and_content(self):
        from src.storage.file_import import parse_markdown_file

        content = (
            "---\n"
            "title: My Test Note\n"
            "tags: [test, demo]\n"
            "---\n"
            "# Hello World\n\nThis is the body."
        )
        note = parse_markdown_file(content, "my-test-note.md")
        assert note.title == "My Test Note"
        assert note.content.strip() == "# Hello World\n\nThis is the body."
        assert note.path == "my-test-note.md"
        assert "test" in note.frontmatter.get("tags", [])

    def test_parses_title_from_filename_when_no_frontmatter(self):
        from src.storage.file_import import parse_markdown_file

        content = "Just some content without frontmatter."
        note = parse_markdown_file(content, "my-awesome-note.md")
        assert note.title == "My Awesome Note"
        assert note.content == content
        assert note.path == "my-awesome-note.md"

    def test_parses_title_from_h1_when_no_frontmatter_title(self):
        from src.storage.file_import import parse_markdown_file

        content = "# H1 Title\n\nBody text here."
        note = parse_markdown_file(content, "some-file.md")
        assert note.title == "H1 Title"

    def test_extracts_tags_from_frontmatter(self):
        from src.storage.file_import import parse_markdown_file

        content = (
            "---\n"
            "title: Tagged Note\n"
            "tags: [python, coding, ai]\n"
            "---\n"
            "Body."
        )
        note = parse_markdown_file(content, "tagged.md")
        assert note.frontmatter["tags"] == ["python", "coding", "ai"]

    def test_extracts_aliases_from_frontmatter(self):
        from src.storage.file_import import parse_markdown_file

        content = (
            "---\n"
            "title: Aliased Note\n"
            "aliases: [alias1, alias2]\n"
            "---\n"
            "Body."
        )
        note = parse_markdown_file(content, "aliased.md")
        assert note.frontmatter["aliases"] == ["alias1", "alias2"]

    def test_generates_uuid_id(self):
        from src.storage.file_import import parse_markdown_file

        content = "---\ntitle: ID Test\n---\nBody."
        note = parse_markdown_file(content, "id-test.md")
        # Should be a valid UUID
        uuid.UUID(note.id)

    def test_preserves_created_modified_from_frontmatter(self):
        from src.storage.file_import import parse_markdown_file

        content = (
            "---\n"
            "title: Dated Note\n"
            "created: 2026-01-15T10:30:00+00:00\n"
            "modified: 2026-02-20T14:00:00+00:00\n"
            "---\n"
            "Body."
        )
        note = parse_markdown_file(content, "dated.md")
        assert note.created.year == 2026
        assert note.created.month == 1
        assert note.modified.year == 2026
        assert note.modified.month == 2


class TestExtractWikilinks:
    """Test extract_wikilinks finds all [[links]]."""

    def test_extracts_single_wikilink(self):
        from src.storage.file_import import extract_wikilinks

        content = "See [[Other Note]] for details."
        result = extract_wikilinks(content)
        assert result == ["Other Note"]

    def test_extracts_multiple_wikilinks(self):
        from src.storage.file_import import extract_wikilinks

        content = "See [[Note A]] and [[Note B]] for details."
        result = extract_wikilinks(content)
        assert result == ["Note A", "Note B"]

    def test_extracts_wikilinks_with_aliases(self):
        from src.storage.file_import import extract_wikilinks

        content = "See [[Target Note|Display Text]] for details."
        result = extract_wikilinks(content)
        assert "Target Note|Display Text" in result

    def test_returns_empty_list_when_no_wikilinks(self):
        from src.storage.file_import import extract_wikilinks

        content = "This has no wikilinks at all."
        result = extract_wikilinks(content)
        assert result == []

    def test_extracts_wikilinks_from_multiline_content(self):
        from src.storage.file_import import extract_wikilinks

        content = (
            "# Title\n\n"
            "First reference [[Link One]].\n\n"
            "Second reference [[Link Two]]."
        )
        result = extract_wikilinks(content)
        assert result == ["Link One", "Link Two"]

    def test_deduplicates_wikilinks(self):
        from src.storage.file_import import extract_wikilinks

        content = "[[Same Link]] appears twice [[Same Link]]."
        result = extract_wikilinks(content)
        assert result == ["Same Link"]


class TestExtractTags:
    """Test extract_tags finds all #tags."""

    def test_extracts_single_tag(self):
        from src.storage.file_import import extract_tags

        content = "This is a #test tag."
        result = extract_tags(content)
        assert result == ["test"]

    def test_extracts_multiple_tags(self):
        from src.storage.file_import import extract_tags

        content = "Tags: #python #coding #ai"
        result = extract_tags(content)
        assert result == ["python", "coding", "ai"]

    def test_returns_empty_list_when_no_tags(self):
        from src.storage.file_import import extract_tags

        content = "No tags here."
        result = extract_tags(content)
        assert result == []

    def test_extracts_tags_from_headers(self):
        from src.storage.file_import import extract_tags

        content = "# Title with #tag in it"
        result = extract_tags(content)
        assert "tag" in result

    def test_deduplicates_tags(self):
        from src.storage.file_import import extract_tags

        content = "#python is great. I love #python."
        result = extract_tags(content)
        assert result == ["python"]

    def test_ignores_tags_in_code_blocks(self):
        from src.storage.file_import import extract_tags

        content = "```\n#not-a-tag\n```\n#real-tag"
        result = extract_tags(content)
        assert "real-tag" in result
        # Code block content should be excluded
        assert "not-a-tag" not in result


# ---------------------------------------------------------------------------
# API endpoint tests (HTTP via TestClient, mocked store)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_notes_dir(tmp_path):
    """Create a temporary notes directory."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    return str(notes_dir)


@pytest.fixture
def client(tmp_notes_dir):
    """Create a FastAPI TestClient with mocked note store."""
    from src.api import import_export

    mock_store = MagicMock()
    mock_store.create = MagicMock(side_effect=lambda n: n)
    mock_store.get = MagicMock(return_value=None)
    mock_store.list_all = MagicMock(return_value=[])

    with patch.object(import_export, "_note_store", mock_store):
        from src.main import app
        yield TestClient(app), mock_store


class TestImportMarkdownEndpoint:
    """Test POST /api/import/markdown creates note."""

    def test_import_single_markdown_file(self, client):
        cli, mock_store = client
        mock_store.create = MagicMock(side_effect=lambda n: n)

        file_content = (
            "---\n"
            "title: Imported Note\n"
            "tags: [imported, test]\n"
            "---\n"
            "# Imported\n\nThis note was imported."
        )
        response = cli.post(
            "/api/import/markdown",
            files={"file": ("imported.md", io.BytesIO(file_content.encode()), "text/markdown")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Imported Note"
        assert data["path"] == "imported.md"
        mock_store.create.assert_called_once()

    def test_import_without_frontmatter(self, client):
        cli, mock_store = client
        mock_store.create = MagicMock(side_effect=lambda n: n)

        file_content = "# Simple Note\n\nJust content."
        response = cli.post(
            "/api/import/markdown",
            files={"file": ("simple.md", io.BytesIO(file_content.encode()), "text/markdown")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Simple Note"

    def test_import_missing_file_returns_error(self, client):
        cli, mock_store = client

        response = cli.post("/api/import/markdown")
        assert response.status_code == 422


class TestImportAuthConsistency:
    """Import routes carry the same require_auth dependency as POST /api/notes/.

    In the LOCAL-TRUST default (auth_required=False) this is a no-op: anonymous
    access is still allowed, so importing with no token must keep working.
    """

    def test_import_markdown_works_without_token_when_auth_not_required(self, client):
        cli, mock_store = client
        mock_store.create = MagicMock(side_effect=lambda n: n)

        file_content = "# No Token Import\n\nStill allowed in local-trust mode."
        response = cli.post(
            "/api/import/markdown",
            files={"file": ("no_token.md", io.BytesIO(file_content.encode()), "text/markdown")},
        )
        assert response.status_code == 200

    def test_import_markdown_route_declares_require_auth_dependency(self):
        from src.api import import_export
        from src.auth import require_auth

        route = next(
            r
            for r in import_export.router.routes
            if getattr(r, "path", None) == "/api/import/markdown"
        )
        dep_calls = [d.call for d in route.dependant.dependencies]
        assert require_auth in dep_calls


class TestImportBulkEndpoint:
    """Test POST /api/import/bulk accepts multiple markdown files."""

    def test_import_multiple_files(self, client):
        cli, mock_store = client
        mock_store.create = MagicMock(side_effect=lambda n: n)

        file1 = ("note1.md", io.BytesIO(b"---\ntitle: Note 1\n---\nContent 1"), "text/markdown")
        file2 = ("note2.md", io.BytesIO(b"---\ntitle: Note 2\n---\nContent 2"), "text/markdown")

        response = cli.post(
            "/api/import/bulk",
            files=[("files", file1), ("files", file2)],
        )
        assert response.status_code == 200
        data = response.json()
        assert data["imported_count"] == 2
        assert mock_store.create.call_count == 2

    def test_import_bulk_empty_returns_zero(self, client):
        cli, mock_store = client

        response = cli.post(
            "/api/import/bulk",
            files=[],
        )
        # Should handle gracefully
        assert response.status_code in (200, 422)


class TestImportObsidianEndpoint:
    """Test POST /api/import/obsidian accepts Obsidian vault zip."""

    def test_import_obsidian_vault(self, client):
        cli, mock_store = client
        mock_store.create = MagicMock(side_effect=lambda n: n)

        # Create a minimal zip in memory
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "note1.md",
                "---\ntitle: Obsidian Note\n---\n# Hello from Obsidian",
            )
            zf.writestr(
                "subdir/note2.md",
                "---\ntitle: Nested Note\n---\nNested content.",
            )
        buf.seek(0)

        response = cli.post(
            "/api/import/obsidian",
            files={"file": ("vault.zip", buf, "application/zip")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["imported_count"] == 2

    def test_import_obsidian_non_zip_returns_error(self, client):
        cli, mock_store = client

        response = cli.post(
            "/api/import/obsidian",
            files={"file": ("not-a-zip.md", io.BytesIO(b"not a zip"), "text/markdown")},
        )
        assert response.status_code == 400


class TestExportNoteEndpoint:
    """Test GET /api/export/note/{note_id} returns markdown."""

    def test_export_single_note(self, client):
        cli, mock_store = client

        note_id = str(uuid.uuid4())
        note_data = MagicMock()
        note_data.id = note_id
        note_data.title = "Export Test"
        note_data.content = "# Exported\n\nThis is the exported content."
        note_data.frontmatter = {"tags": ["export", "test"]}
        note_data.links = []
        note_data.path = "/notes/export-test.md"
        note_data.created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        note_data.modified = datetime(2026, 6, 1, tzinfo=timezone.utc)
        note_data.source_type = "note"
        mock_store.get = MagicMock(return_value=note_data)

        response = cli.get(f"/api/export/note/{note_id}")
        assert response.status_code == 200
        assert "text/markdown" in response.headers.get("content-type", "")
        body = response.content.decode()
        assert "Export Test" in body
        assert "Exported" in body

    def test_export_nonexistent_note_returns_404(self, client):
        cli, mock_store = client
        mock_store.get = MagicMock(return_value=None)

        response = cli.get(f"/api/export/note/{uuid.uuid4()}")
        assert response.status_code == 404


class TestExportAllEndpoint:
    """Test GET /api/export/all returns zip of all notes."""

    def test_export_all_notes(self, client):
        cli, mock_store = client

        notes = []
        for i in range(3):
            note = MagicMock()
            note.id = str(uuid.uuid4())
            note.title = f"Note {i}"
            note.content = f"Content {i}"
            note.frontmatter = {"tags": [f"tag{i}"]}
            note.links = []
            note.path = f"/notes/note-{i}.md"
            note.created = datetime(2026, 1, 1, tzinfo=timezone.utc)
            note.modified = datetime(2026, 6, 1, tzinfo=timezone.utc)
            note.source_type = "note"
            notes.append(note)
        mock_store.list_all = MagicMock(return_value=notes)

        response = cli.get("/api/export/all")
        assert response.status_code == 200
        assert "application/zip" in response.headers.get("content-type", "")

        # Verify it's a valid zip with 3 files
        import zipfile

        buf = io.BytesIO(response.content)
        with zipfile.ZipFile(buf, "r") as zf:
            names = zf.namelist()
            assert len(names) == 3

    def test_export_all_empty_returns_empty_zip(self, client):
        cli, mock_store = client
        mock_store.list_all = MagicMock(return_value=[])

        response = cli.get("/api/export/all")
        assert response.status_code == 200

        import zipfile

        buf = io.BytesIO(response.content)
        with zipfile.ZipFile(buf, "r") as zf:
            assert len(zf.namelist()) == 0
