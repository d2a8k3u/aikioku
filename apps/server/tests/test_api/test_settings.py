"""Tests for Settings API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI TestClient."""
    from src.main import app

    yield TestClient(app)


class TestGetSettings:
    """Test GET /api/settings."""

    def test_returns_current_config(self, client):
        """GET /api/settings should return current settings values."""
        response = client.get("/api/settings")

        assert response.status_code == 200
        data = response.json()
        assert "llm_provider" in data
        assert "embedding_model" in data
        assert "auto_extract" in data
        assert "auto_consolidation" in data
        assert isinstance(data["llm_provider"], str)
        assert isinstance(data["embedding_model"], str)
        assert isinstance(data["auto_extract"], bool)
        assert isinstance(data["auto_consolidation"], bool)


class TestUpdateSettings:
    """Test PUT /api/settings."""

    def test_updates_and_returns_new_config(self, client):
        """PUT /api/settings should update settings and return new values."""
        update_payload = {
            "llm_provider": "ollama_remote",
            "embedding_model": "mxbai-embed-large",
            "auto_extract": False,
            "auto_consolidation": True,
        }

        response = client.put("/api/settings", json=update_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["llm_provider"] == "ollama_remote"
        assert data["embedding_model"] == "mxbai-embed-large"
        assert data["auto_extract"] is False
        assert data["auto_consolidation"] is True

    def test_partial_update(self, client):
        """PUT /api/settings with partial fields should update only those fields."""
        update_payload = {
            "llm_provider": "openrouter",
        }

        response = client.put("/api/settings", json=update_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["llm_provider"] == "openrouter"
