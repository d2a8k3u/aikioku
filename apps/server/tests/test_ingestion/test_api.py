"""Tests for ingestion API endpoints."""

from __future__ import annotations

import io
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
    app.state.note_store = store
    import_export._note_store = store
    original_notes_dir = settings.notes_dir
    settings.notes_dir = notes_dir
    yield TestClient(app), store, notes_dir
    settings.notes_dir = original_notes_dir
    import_export._note_store = None
    for attr in ("note_store",):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


class TestImportPdfEndpoint:
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


class TestImportDocxEndpoint:
    def test_import_docx_missing_dependency(self, client):
        cli, store, _ = client
        with patch("src.ingestion.docx_parser.parse_docx", side_effect=ImportError("No python-docx")):
            response = cli.post(
                "/api/import/docx",
                files={"file": ("test.docx", io.BytesIO(b"fake"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )
        assert response.status_code == 503

    def test_import_docx_bad_file(self, client):
        cli, store, _ = client
        with patch("src.ingestion.docx_parser.parse_docx", side_effect=ValueError("bad docx")):
            response = cli.post(
                "/api/import/docx",
                files={"file": ("test.docx", io.BytesIO(b"fake"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )
        assert response.status_code == 400


class TestImportAudioEndpoint:
    def test_import_audio_missing_dependency(self, client):
        cli, store, _ = client
        with patch("src.ingestion.audio_parser.parse_audio", side_effect=ImportError("No whisper")):
            response = cli.post(
                "/api/import/audio",
                files={"file": ("test.mp3", io.BytesIO(b"fake"), "audio/mpeg")},
            )
        assert response.status_code == 503

    def test_import_audio_transcription_fails(self, client):
        cli, store, _ = client
        with patch("src.ingestion.audio_parser.parse_audio", side_effect=RuntimeError("transcription failed")):
            response = cli.post(
                "/api/import/audio",
                files={"file": ("test.mp3", io.BytesIO(b"fake"), "audio/mpeg")},
            )
        assert response.status_code == 400


class TestImportImageEndpoint:
    def test_import_image_missing_dependency(self, client):
        cli, store, _ = client
        with patch("src.ingestion.image_parser.parse_image", side_effect=ImportError("No tesseract")):
            response = cli.post(
                "/api/import/image",
                files={"file": ("test.png", io.BytesIO(b"fake"), "image/png")},
            )
        assert response.status_code == 503

    def test_import_image_ocr_fails(self, client):
        cli, store, _ = client
        with patch("src.ingestion.image_parser.parse_image", side_effect=RuntimeError("OCR failed")):
            response = cli.post(
                "/api/import/image",
                files={"file": ("test.png", io.BytesIO(b"fake"), "image/png")},
            )
        assert response.status_code == 400


class TestImportWebEndpoint:
    def test_import_web_missing_dependency(self, client):
        cli, store, _ = client
        with patch("src.ingestion.web_parser.parse_web_clip", side_effect=ImportError("No requests")):
            response = cli.post("/api/import/web?url=https%3A%2F%2Fexample.com")
        assert response.status_code == 503

    def test_import_web_bad_url(self, client):
        cli, store, _ = client
        with patch("src.ingestion.web_parser.parse_web_clip", side_effect=RuntimeError("timeout")):
            response = cli.post("/api/import/web?url=https%3A%2F%2Fexample.com")
        assert response.status_code == 400


class TestImportEmailEndpoint:
    def test_import_email_success(self, client):
        cli, store, _ = client
        raw = (
            b"Subject: Hello\r\n"
            b"\r\n"
            b"Body text"
        )
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
