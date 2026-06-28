"""Model-listing and setup-test endpoints must build normalized URLs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api import settings as settings_api
from src.api import setup as setup_api
from src.llm.model_list import list_ollama_models, list_openrouter_models


def _ok(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    return resp


class TestModelListUrls:
    @pytest.mark.asyncio
    async def test_ollama_tags_url_normalized(self):
        with patch(
            "httpx.AsyncClient.get", new=AsyncMock(return_value=_ok({"models": []}))
        ) as mock_get:
            await list_ollama_models("https://api.ollama.com/v1")

        args, kwargs = mock_get.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "https://api.ollama.com/api/tags"

    @pytest.mark.asyncio
    async def test_openrouter_models_url_normalized(self):
        with patch(
            "httpx.AsyncClient.get", new=AsyncMock(return_value=_ok({"data": []}))
        ) as mock_get:
            await list_openrouter_models("https://openrouter.ai")

        args, kwargs = mock_get.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "https://openrouter.ai/api/v1/models"


class TestSetupTestUrls:
    @pytest.mark.asyncio
    async def test_ollama_branch_normalizes(self):
        payload = setup_api.TestPayload(
            llm_provider="ollama_remote", ollama_base_url="https://api.ollama.com/api/v1"
        )
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=_ok({}))) as mock_get:
            await setup_api.test_connection(payload)

        args, kwargs = mock_get.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "https://api.ollama.com/api/tags"

    @pytest.mark.asyncio
    async def test_openrouter_branch_normalizes(self):
        payload = setup_api.TestPayload(
            llm_provider="openrouter", openrouter_base_url="https://openrouter.ai"
        )
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=_ok({}))) as mock_get:
            await setup_api.test_connection(payload)

        args, kwargs = mock_get.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "https://openrouter.ai/api/v1/models"

    @pytest.mark.asyncio
    async def test_empty_ollama_base_short_circuits(self):
        payload = setup_api.TestPayload(llm_provider="ollama", ollama_base_url="")
        result = await setup_api.test_connection(payload)
        assert result["ok"] is False


class TestEmbeddingModelsListing:
    @pytest.mark.asyncio
    async def test_openai_curated_filtered_by_q(self):
        res = await settings_api.list_embedding_models(provider="openai", q="large", _user=None)
        ids = [m["id"] for m in res["models"]]
        assert ids == ["text-embedding-3-large"]
        assert res["error"] is None

    @pytest.mark.asyncio
    async def test_ollama_uses_embedding_base_and_tags(self):
        tags = _ok({"models": [{"name": "nomic-embed-text", "details": {"family": "nomic-bert"}}]})
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=tags)) as mock_get:
            res = await settings_api.list_embedding_models(
                provider="ollama",
                base_url="http://host:11434/api",
                api_key=None,
                q=None,
                _user=None,
            )
        args, kwargs = mock_get.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "http://host:11434/api/tags"
        assert any(m["id"] == "nomic-embed-text" for m in res["models"])

    @pytest.mark.asyncio
    async def test_huggingface_live_search(self):
        hits = _ok([{"id": "sentence-transformers/all-MiniLM-L6-v2"}])
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=hits)) as mock_get:
            res = await settings_api.list_embedding_models(
                provider="huggingface", q="minilm", _user=None
            )
        args, kwargs = mock_get.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "https://huggingface.co/api/models"
        assert kwargs["params"]["search"] == "minilm"
        assert res["models"][0]["provider"] == "huggingface"
