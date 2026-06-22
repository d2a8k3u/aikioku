"""Tests for Stats API endpoint."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.main import app
    yield TestClient(app)


class TestGetStats:
    def test_returns_all_counts_and_version(self, client):
        with (
            patch("src.api.stats._get_note_count", new_callable=AsyncMock, return_value=5) as mock_notes,
            patch("src.api.stats._get_entity_count", new_callable=AsyncMock, return_value=3) as mock_entities,
            patch("src.api.stats._get_relation_count", new_callable=AsyncMock, return_value=2) as mock_relations,
            patch("src.api.stats._get_memory_count", new_callable=AsyncMock, return_value=10) as mock_memories,
            patch("src.api.stats._get_card_count", return_value=4) as mock_cards,
        ):
            response = client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["notes"] == 5
        assert data["entities"] == 3
        assert data["relations"] == 2
        assert data["memories"] == 10
        assert data["cards"] == 4
        assert data["version"] == "0.1.0"
        mock_notes.assert_awaited_once()
        mock_entities.assert_awaited_once()
        mock_relations.assert_awaited_once()
        mock_memories.assert_awaited_once()
        mock_cards.assert_called_once()

    def test_zero_counts_when_empty(self, client):
        with (
            patch("src.api.stats._get_note_count", new_callable=AsyncMock, return_value=0),
            patch("src.api.stats._get_entity_count", new_callable=AsyncMock, return_value=0),
            patch("src.api.stats._get_relation_count", new_callable=AsyncMock, return_value=0),
            patch("src.api.stats._get_memory_count", new_callable=AsyncMock, return_value=0),
            patch("src.api.stats._get_card_count", return_value=0),
        ):
            response = client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["notes"] == 0
        assert data["entities"] == 0
        assert data["relations"] == 0
        assert data["memories"] == 0
        assert data["cards"] == 0
        assert data["version"] == "0.1.0"
