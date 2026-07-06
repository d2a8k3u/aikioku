"""Tests for settings hot-reload: cache invalidation and spawn_background usage."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def configured_client():
    """Create a TestClient with app.state.configured=True and build_runtime mocked."""
    from src.main import app

    app.state.configured = True
    with patch("src.main.build_runtime"):
        yield TestClient(app)


class TestModelSwitchInvalidatesCache:
    """Verify cache_invalidate is called when a provider-affecting setting changes."""

    def test_model_switch_invalidates_cache(self, configured_client):
        with patch(
            "src.cache.semantic_cache.cache_invalidate", new_callable=AsyncMock
        ) as mock_invalidate:
            response = configured_client.put(
                "/api/settings", json={"llm_provider": "ollama_remote"}
            )

            assert response.status_code == 200
            mock_invalidate.assert_called_once()


class TestCacheInvalidateUsesSpawnBackground:
    """Verify spawn_background is used (not raw asyncio.ensure_future)."""

    def test_cache_invalidate_uses_spawn_background(self, configured_client):
        with patch("src.api.settings.spawn_background") as mock_spawn:
            response = configured_client.put("/api/settings", json={"llm_model": "llama3"})

            assert response.status_code == 200
            mock_spawn.assert_called_once()


class TestModelSwitchNewRequestsSucceed:
    """Verify the settings API returns 200 after a model switch."""

    def test_model_switch_new_requests_succeed(self, configured_client):
        response = configured_client.put("/api/settings", json={"llm_provider": "openrouter"})

        assert response.status_code == 200
        data = response.json()
        assert data["llm_provider"] == "openrouter"
