"""Tests for GET /api/export/json endpoint."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI TestClient with mocked note store."""
    from src.main import app

    yield TestClient(app)


@pytest.fixture
def tmp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def client_with_graph(tmp_db_path):
    """Create a FastAPI TestClient with a real KnowledgeGraph in app state."""
    from src.knowledge.graph import KnowledgeGraph
    from src.main import app

    kg = KnowledgeGraph(db_path=tmp_db_path)
    app.state.knowledge_graph = kg
    yield TestClient(app)


class TestExportJson:
    """Test GET /api/export/json."""

    def test_returns_json_structure(self, client):
        with patch("src.api.import_export.get_note_store") as mock_store_fn:
            mock_store = MagicMock()
            mock_store.list_all = MagicMock(return_value=[])
            mock_store_fn.return_value = mock_store

            response = client.get("/api/export/json")

        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("application/json")

        data = response.json()
        assert "notes" in data
        assert "entities" in data
        assert "relations" in data
        assert "memories" in data
        assert "cards" in data
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_notes_included(self, client):
        from datetime import datetime

        note = MagicMock()
        note.id = "note-1"
        note.title = "My Note"
        note.content = "Body"
        note.frontmatter = {}
        note.links = []
        note.path = "/path.md"
        note.created = datetime(2026, 6, 1, 12, 0, 0)
        note.modified = datetime(2026, 6, 2, 12, 0, 0)

        with patch("src.api.import_export.get_note_store") as mock_store_fn:
            mock_store = MagicMock()
            mock_store.list_all = MagicMock(return_value=[note])
            mock_store_fn.return_value = mock_store

            response = client.get("/api/export/json")

        data = response.json()
        assert len(data["notes"]) == 1
        assert data["notes"][0]["id"] == "note-1"
        assert data["notes"][0]["title"] == "My Note"
        assert data["notes"][0]["content"] == "Body"
        assert data["notes"][0]["created"] == "2026-06-01T12:00:00"

    def test_empty_export(self, client):
        with patch("src.api.import_export.get_note_store") as mock_store_fn:
            mock_store = MagicMock()
            mock_store.list_all = MagicMock(return_value=[])
            mock_store_fn.return_value = mock_store

            response = client.get("/api/export/json")

        data = response.json()
        assert data["notes"] == []
        assert data["entities"] == []
        assert data["relations"] == []
        assert data["memories"] == []
        assert data["cards"] == []
        assert data["version"] == "0.1.0"

    def test_entity_count_zero(self, client_with_graph):
        response = client_with_graph.get("/api/export/json")
        assert response.status_code == 200
        data = response.json()
        assert data["entities"] == []
        assert data["relations"] == []
