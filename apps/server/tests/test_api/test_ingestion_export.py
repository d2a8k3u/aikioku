"""Tests for Ingestion & Export production features."""

from __future__ import annotations

import builtins
import io
import os
import zipfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    """Create a FastAPI TestClient with a fresh NoteStore in a temp directory."""
    from src.main import app
    from src.storage.note_store import NoteStore
    from src.api import import_export
    from src.config import settings

    notes_dir = str(tmp_path / "notes")
    store = NoteStore(notes_dir)
    # Wire store into both app.state and import_export module
    app.state.note_store = store
    import_export._note_store = store
    # Patch settings so any runtime reads use temp dir
    original_notes_dir = settings.notes_dir
    settings.notes_dir = notes_dir
    # Initialize git sync
    from src.storage.git_sync import GitSync

    gs = GitSync(notes_dir)
    app.state.git_sync = gs
    yield TestClient(app), store, notes_dir
    # Restore
    settings.notes_dir = original_notes_dir
    import_export._note_store = None
    for attr in ("note_store", "git_sync"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


# ---------------------------------------------------------------------------
# Ingestion parsers (unit tests)
# ---------------------------------------------------------------------------


class TestPdfParser:
    def test_import_error_when_pymupdf_missing(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fitz":
                raise ImportError("No module named 'fitz'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", fake_import):
            from src.ingestion.pdf_parser import parse_pdf

            with pytest.raises(ImportError):
                parse_pdf(b"fake pdf", "test.pdf")


class TestDocxParser:
    def test_import_error_when_docx_missing(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "docx":
                raise ImportError("No module named 'docx'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", fake_import):
            from src.ingestion.docx_parser import parse_docx

            with pytest.raises(ImportError):
                parse_docx(b"fake docx", "test.docx")


class TestAudioParser:
    def test_parse_audio_returns_fallback_note(self):
        """parse_audio no longer uses OpenAI Whisper — returns fallback note."""
        from src.ingestion.audio_parser import parse_audio

        note = parse_audio(b"fake audio", "test.mp3")
        assert note.title == "Test"
        assert "Transcription unavailable" in note.content


class TestImageParser:
    def test_import_error_when_pil_missing(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("PIL", "pytesseract"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", fake_import):
            from src.ingestion.image_parser import parse_image

            with pytest.raises(ImportError):
                parse_image(b"fake image", "test.png")


class TestWebParser:
    def test_import_error_when_requests_missing(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("requests", "readability"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", fake_import):
            from src.ingestion.web_parser import parse_web_clip

            with pytest.raises(ImportError):
                parse_web_clip("https://example.com")


class TestEmailParser:
    def test_parse_email_simple(self):
        from src.ingestion.email_parser import parse_email

        raw = (
            b"Subject: Hello World\r\n"
            b"Date: Mon, 01 Jan 2024 12:00:00 +0000\r\n"
            b"\r\n"
            b"This is the body."
        )
        note = parse_email(raw, "test.eml")
        assert note.title == "Hello World"
        assert "This is the body." in note.content
        assert note.created.year == 2024

    def test_parse_email_multipart(self):
        from src.ingestion.email_parser import parse_email

        raw = (
            b"Subject: Multi\r\n"
            b"MIME-Version: 1.0\r\n"
            b"Content-Type: multipart/alternative; boundary=bound\r\n"
            b"\r\n"
            b"--bound\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\n"
            b"Plain text\r\n"
            b"--bound\r\n"
            b"Content-Type: text/html\r\n"
            b"\r\n"
            b"<html><body>HTML</body></html>\r\n"
            b"--bound--\r\n"
        )
        note = parse_email(raw, "multi.eml")
        assert note.title == "Multi"
        assert "Plain text" in note.content
        assert "HTML" in note.content

    def test_parse_email_bad_input_graceful(self):
        from src.ingestion.email_parser import parse_email

        # Garbage bytes should still produce a Note (default title)
        note = parse_email(b"not an email", "bad.eml")
        assert isinstance(note.title, str)
        assert note.content == "not an email"


# ---------------------------------------------------------------------------
# API import endpoints
# ---------------------------------------------------------------------------


class TestImportPdf:
    def test_import_pdf_missing_dependency(self, client):
        cli, store, _ = client
        with patch("src.ingestion.pdf_parser.parse_pdf", side_effect=ImportError("No PyMuPDF")):
            response = cli.post(
                "/api/import/pdf",
                files={"file": ("test.pdf", io.BytesIO(b"fake"), "application/pdf")},
            )
        assert response.status_code == 503

    def test_import_pdf_bad_file(self, client):
        cli, store, _ = client
        with patch("src.ingestion.pdf_parser.parse_pdf", side_effect=ValueError("bad pdf")):
            response = cli.post(
                "/api/import/pdf",
                files={"file": ("test.pdf", io.BytesIO(b"fake"), "application/pdf")},
            )
        assert response.status_code == 400


class TestImportDocx:
    def test_import_docx_missing_dependency(self, client):
        cli, store, _ = client
        with patch(
            "src.ingestion.docx_parser.parse_docx", side_effect=ImportError("No python-docx")
        ):
            response = cli.post(
                "/api/import/docx",
                files={
                    "file": (
                        "test.docx",
                        io.BytesIO(b"fake"),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )
        assert response.status_code == 503


class TestImportAudio:
    def test_import_audio_missing_dependency(self, client):
        cli, store, _ = client
        with patch("src.ingestion.audio_parser.parse_audio", side_effect=ImportError("No whisper")):
            response = cli.post(
                "/api/import/audio",
                files={"file": ("test.mp3", io.BytesIO(b"fake"), "audio/mpeg")},
            )
        assert response.status_code == 503


class TestImportImage:
    def test_import_image_missing_dependency(self, client):
        cli, store, _ = client
        with patch(
            "src.ingestion.image_parser.parse_image", side_effect=ImportError("No tesseract")
        ):
            response = cli.post(
                "/api/import/image",
                files={"file": ("test.png", io.BytesIO(b"fake"), "image/png")},
            )
        assert response.status_code == 503


class TestImportWeb:
    def test_import_web_missing_dependency(self, client):
        cli, store, _ = client
        with patch(
            "src.ingestion.web_parser.parse_web_clip", side_effect=ImportError("No requests")
        ):
            response = cli.post("/api/import/web?url=https%3A%2F%2Fexample.com")
        assert response.status_code == 503

    def test_import_web_bad_url(self, client):
        cli, store, _ = client
        with patch("src.ingestion.web_parser.parse_web_clip", side_effect=RuntimeError("timeout")):
            response = cli.post("/api/import/web?url=https%3A%2F%2Fexample.com")
        assert response.status_code == 400


class TestImportEmail:
    def test_import_email_success(self, client):
        cli, store, _ = client
        raw = b"Subject: Hello\r\n\r\nBody text"
        response = cli.post(
            "/api/import/email",
            files={"file": ("test.eml", io.BytesIO(raw), "message/rfc822")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Hello"
        assert "Body text" in data["content"]

    def test_import_email_bad_file(self, client):
        cli, store, _ = client
        response = cli.post(
            "/api/import/email",
            files={"file": ("bad.eml", io.BytesIO(b"not an email"), "message/rfc822")},
        )
        # Bad email bytes still parse gracefully
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Logseq / Notion / Roam import
# ---------------------------------------------------------------------------


class TestImportLogseq:
    def test_import_logseq_zip(self, client):
        cli, store, _ = client
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("pages/note1.md", "---\ntitle: Logseq Note\n---\nContent")
        buf.seek(0)
        response = cli.post(
            "/api/import/logseq",
            files={"file": ("logseq.zip", buf, "application/zip")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["imported_count"] == 1

    def test_import_logseq_non_zip(self, client):
        cli, store, _ = client
        response = cli.post(
            "/api/import/logseq",
            files={"file": ("not.zip", io.BytesIO(b"nope"), "text/plain")},
        )
        assert response.status_code == 400


class TestImportNotion:
    def test_import_notion_zip(self, client):
        cli, store, _ = client
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("Page 1.md", "---\ntitle: Notion Page\n---\nContent")
            zf.writestr("Database.csv", "Name,Status\nTask 1,Done\nTask 2,Open\n")
        buf.seek(0)
        response = cli.post(
            "/api/import/notion",
            files={"file": ("notion.zip", buf, "application/zip")},
        )
        assert response.status_code == 200
        data = response.json()
        # 1 markdown + 2 CSV rows
        assert data["imported_count"] == 3


class TestImportRoam:
    def test_import_roam_json(self, client):
        cli, store, _ = client
        payload = [
            {
                "title": "Roam Page",
                "children": [
                    {"string": "Block 1", "children": [{"string": "Nested"}]},
                    {"string": "Block 2"},
                ],
            }
        ]
        import json as _json

        response = cli.post(
            "/api/import/roam",
            files={
                "file": ("roam.json", io.BytesIO(_json.dumps(payload).encode()), "application/json")
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["imported_count"] == 1

    def test_import_roam_zip(self, client):
        cli, store, _ = client
        import json as _json

        payload = [{"title": "Zipped Roam", "children": [{"string": "Hello"}]}]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("roam.json", _json.dumps(payload))
        buf.seek(0)
        response = cli.post(
            "/api/import/roam",
            files={"file": ("roam.zip", buf, "application/zip")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["imported_count"] == 1

    def test_import_roam_bad_input(self, client):
        cli, store, _ = client
        response = cli.post(
            "/api/import/roam",
            files={"file": ("bad.txt", io.BytesIO(b"not json"), "text/plain")},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------


class TestExportAnki:
    def test_export_anki_empty(self, client):
        cli, store, _ = client
        response = cli.get("/api/export/anki")
        assert response.status_code == 200
        data = response.json()
        assert "decks" in data
        assert "notes" in data
        assert data["notes"] == []

    def test_export_anki_with_notes(self, client):
        cli, store, _ = client
        from src.models.note import Note

        store.create(
            Note(title="Note 1", content="Body 1", path="n1.md", frontmatter={"tags": ["a", "b"]})
        )
        store.create(Note(title="Note 2", content="Body 2", path="n2.md"))
        response = cli.get("/api/export/anki")
        data = response.json()
        assert len(data["notes"]) == 2
        tags = {tuple(n["tags"]) for n in data["notes"]}
        assert tags == {("a", "b"), ()}


class TestExportBibtex:
    def test_export_bibtex_empty(self, client):
        cli, store, _ = client
        response = cli.get("/api/export/bibtex")
        assert response.status_code == 200
        assert "@misc" not in response.text

    def test_export_bibtex_with_notes(self, client):
        cli, store, _ = client
        from src.models.note import Note

        store.create(
            Note(
                title="Bib Note",
                content="Abstract here",
                path="n.md",
                frontmatter={"tags": ["tag1"]},
            )
        )
        response = cli.get("/api/export/bibtex")
        assert response.status_code == 200
        assert "@misc" in response.text
        assert "Bib Note" in response.text
        assert "tag1" in response.text


class TestExportJsonExpanded:
    def test_export_json_includes_cognitive_signals(self, client):
        cli, store, _ = client
        response = cli.get("/api/export/json")
        assert response.status_code == 200
        data = response.json()
        assert "cognitive_signals" in data
        assert "settings" in data
        assert "plugin_state" in data


# ---------------------------------------------------------------------------
# Git sync API
# ---------------------------------------------------------------------------


class TestGitSyncApi:
    def test_git_commit(self, client):
        cli, store, tmpdir = client
        note_path = os.path.join(tmpdir, "note1.md")
        with open(note_path, "w") as f:
            f.write("Hello\n")
        response = cli.post("/api/sync/git/commit?message=Add+note1")
        assert response.status_code == 200
        data = response.json()
        assert data["committed"] is True

    def test_git_history(self, client):
        cli, store, tmpdir = client
        response = cli.get("/api/sync/git/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_git_diff(self, client):
        cli, store, tmpdir = client
        note_path = os.path.join(tmpdir, "note.md")
        with open(note_path, "w") as f:
            f.write("v1\n")
        cli.post("/api/sync/git/commit?message=add+note")
        response = cli.get("/api/sync/git/diff/note")
        assert response.status_code == 200
        data = response.json()
        assert "diff" in data


# ---------------------------------------------------------------------------
# Note versioning
# ---------------------------------------------------------------------------


class TestNoteVersioning:
    def test_note_history(self, client):
        cli, store, tmpdir = client
        from src.models.note import Note

        note = Note(title="Versioned", content="v1", path="v.md")
        created = store.create(note)
        # Simulate a modification and commit
        created.content = "v2"
        store.update(created)
        gs = cli.app.state.git_sync
        gs.commit("update note")
        response = cli.get(f"/api/notes/{created.id}/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_note_diff(self, client):
        cli, store, tmpdir = client
        from src.models.note import Note

        note = Note(title="Versioned", content="v1", path="v.md")
        created = store.create(note)
        created.content = "v2"
        store.update(created)
        gs = cli.app.state.git_sync
        gs.commit("update note")
        response = cli.get(f"/api/notes/{created.id}/diff")
        assert response.status_code == 200
        data = response.json()
        assert "diff" in data

    def test_note_diff_custom_commits(self, client):
        cli, store, tmpdir = client
        from src.models.note import Note

        note = Note(title="Versioned", content="v1", path="v.md")
        created = store.create(note)
        gs = cli.app.state.git_sync
        gs.commit("first")
        created.content = "v2"
        store.update(created)
        gs.commit("second")
        history = gs.get_note_history(created.id, limit=2)
        if len(history) >= 2:
            a, b = history[1]["hash"], history[0]["hash"]
            response = cli.get(f"/api/notes/{created.id}/diff?commit_a={a}&commit_b={b}")
            assert response.status_code == 200
            data = response.json()
            assert "diff" in data

    def test_note_history_nonexistent(self, client):
        cli, store, _ = client
        response = cli.get("/api/notes/00000000-0000-0000-0000-000000000000/history")
        # Git history for a nonexistent file returns empty list (HTTP 200)
        assert response.status_code == 200
        data = response.json()
        assert data == []


# ---------------------------------------------------------------------------
# Entity extraction / embedding wiring
# ---------------------------------------------------------------------------


class TestImportTriggersExtraction:
    def test_markdown_import_triggers_entities_and_embeddings(self, client):
        cli, store, _ = client
        with (
            patch("src.api.import_export._extract_and_store_entities") as mock_extract,
            patch("src.api.import_export._store_note_embeddings") as mock_embed,
        ):
            response = cli.post(
                "/api/import/markdown",
                files={"file": ("test.md", io.BytesIO(b"# Hello\n\nBody"), "text/markdown")},
            )
            assert response.status_code == 200
            mock_extract.assert_called_once()
            mock_embed.assert_called_once()

    def test_obsidian_import_triggers_entities_and_embeddings(self, client):
        cli, store, _ = client
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("n.md", "# Hello")
        buf.seek(0)
        with (
            patch("src.api.import_export._extract_and_store_entities") as mock_extract,
            patch("src.api.import_export._store_note_embeddings") as mock_embed,
        ):
            response = cli.post(
                "/api/import/obsidian",
                files={"file": ("vault.zip", buf, "application/zip")},
            )
            assert response.status_code == 200
            assert mock_extract.call_count == 1
            assert mock_embed.call_count == 1

    def test_bulk_import_triggers_entities_and_embeddings(self, client):
        cli, store, _ = client
        with (
            patch("src.api.import_export._extract_and_store_entities") as mock_extract,
            patch("src.api.import_export._store_note_embeddings") as mock_embed,
        ):
            response = cli.post(
                "/api/import/bulk",
                files=[
                    ("files", ("a.md", io.BytesIO(b"# A"), "text/markdown")),
                    ("files", ("b.md", io.BytesIO(b"# B"), "text/markdown")),
                ],
            )
            assert response.status_code == 200
            assert mock_extract.call_count == 2
            assert mock_embed.call_count == 2
