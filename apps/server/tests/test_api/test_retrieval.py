"""Tests for Retrieval API endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.main import app
    yield TestClient(app)


class TestHybridRetrieval:
    """Test POST /api/retrieval/hybrid."""

    def test_returns_fused_results(self, client):
        mock_results = [
            {"note_id": "note-1", "score": 0.95, "source": "fusion", "snippet": "Snippet A", "metadata": {"title": "Note 1"}},
            {"note_id": "note-2", "score": 0.88, "source": "fusion", "snippet": "Snippet B", "metadata": {"title": "Note 2"}},
        ]

        with patch("src.api.retrieval._hybrid_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results
            response = client.post(
                "/api/retrieval/hybrid",
                json={"query": "machine learning", "limit": 5},
            )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["note_id"] == "note-1"
        assert data[0]["score"] == 0.95
        assert data[0]["source"] == "fusion"
        assert data[0]["snippet"] == "Snippet A"
        mock_search.assert_awaited_once()
        assert mock_search.call_args[0][1] == "machine learning"
        assert mock_search.call_args[0][2] == 5

    def test_uses_default_limit(self, client):
        mock_results = []

        with patch("src.api.retrieval._hybrid_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results
            response = client.post(
                "/api/retrieval/hybrid",
                json={"query": "neural networks"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data == []
        mock_search.assert_awaited_once()
        assert mock_search.call_args[0][1] == "neural networks"
        assert mock_search.call_args[0][2] == 20

    def test_missing_query_returns_422(self, client):
        response = client.post(
            "/api/retrieval/hybrid",
            json={"limit": 10},
        )
        assert response.status_code == 422

    def test_empty_query_returns_422(self, client):
        response = client.post(
            "/api/retrieval/hybrid",
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_results_are_serializable(self, client):
        mock_results = [
            {"note_id": "note-1", "score": 0.5, "source": "fusion", "snippet": "", "metadata": {}},
        ]

        with patch("src.api.retrieval._hybrid_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = mock_results
            response = client.post(
                "/api/retrieval/hybrid",
                json={"query": "test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data[0]["note_id"], str)
        assert isinstance(data[0]["score"], float)
        assert isinstance(data[0]["source"], str)
