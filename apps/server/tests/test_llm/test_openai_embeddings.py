"""Tests for OpenAIEmbeddingProvider + factory.build_embedding_provider routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm.ollama_remote import EmbeddingUnavailableError, OllamaRemoteProvider
from src.llm.openai_embeddings import OpenAIEmbeddingProvider


class TestOpenAIEmbed:
    @pytest.mark.asyncio
    async def test_embed_posts_to_v1_embeddings_with_normalized_base(self):
        # Messy base (already has /v1) must still resolve to one /v1/embeddings.
        provider = OpenAIEmbeddingProvider(
            base_url="https://api.openai.com/v1", api_key="sk-x", model="text-embedding-3-small"
        )
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=resp)) as mock_post:
            out = await provider.embed("hi")

        assert out == [0.1, 0.2, 0.3]
        args, kwargs = mock_post.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "https://api.openai.com/v1/embeddings"
        assert kwargs["json"] == {"model": "text-embedding-3-small", "input": "hi"}
        assert kwargs["headers"]["Authorization"] == "Bearer sk-x"

    @pytest.mark.asyncio
    async def test_strict_raises_on_failure(self):
        provider = OpenAIEmbeddingProvider(api_key="sk-x", strict_embeddings=True)
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.ConnectError("down"))):
            with pytest.raises(EmbeddingUnavailableError):
                await provider.embed("x")

    @pytest.mark.asyncio
    async def test_non_strict_returns_deterministic_fallback(self):
        provider = OpenAIEmbeddingProvider(strict_embeddings=False, embedding_fallback_dim=1024)
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.ConnectError("down"))):
            out = await provider.embed("x")
        assert len(out) == 1024


class TestBuildEmbeddingProviderRouting:
    def test_openai_provider_selected(self):
        from src.llm import factory

        with (
            patch("src.runtime_config.embedding_provider", return_value="openai"),
            patch("src.runtime_config.openai_api_key", return_value="sk"),
            patch(
                "src.runtime_config.openai_embedding_model", return_value="text-embedding-3-large"
            ),
            patch("src.runtime_config.embedding_strict", return_value=True),
            patch("src.runtime_config.embedding_dimension", return_value=3072),
        ):
            emb = factory.build_embedding_provider()
        assert isinstance(emb, OpenAIEmbeddingProvider)
        assert emb.model == "text-embedding-3-large"

    def test_ollama_provider_uses_ollama_embedding_model(self):
        from src.llm import factory

        with (
            patch("src.runtime_config.embedding_provider", return_value="ollama"),
            patch("src.runtime_config.ollama_embedding_model", return_value="mxbai-embed-large"),
            patch("src.runtime_config.ollama_embedding_base_url", return_value="http://host:11434"),
            patch("src.runtime_config.embedding_strict", return_value=False),
            patch("src.runtime_config.embedding_dimension", return_value=1024),
        ):
            emb = factory.build_embedding_provider()
        assert isinstance(emb, OllamaRemoteProvider)
        assert emb.embedding_model == "mxbai-embed-large"
        assert emb.embedding_provider == "ollama"

    def test_ollama_remote_passes_embedding_api_key(self):
        from src.llm import factory

        with (
            patch("src.runtime_config.embedding_provider", return_value="ollama_remote"),
            patch("src.runtime_config.ollama_embedding_model", return_value="mxbai-embed-large"),
            patch(
                "src.runtime_config.ollama_embedding_base_url",
                return_value="https://api.ollama.com",
            ),
            patch("src.runtime_config.ollama_api_key", return_value="ollama-secret"),
            patch("src.runtime_config.embedding_strict", return_value=False),
            patch("src.runtime_config.embedding_dimension", return_value=1024),
        ):
            emb = factory.build_embedding_provider()
        assert isinstance(emb, OllamaRemoteProvider)
        assert emb.embedding_provider == "ollama"  # native /api/embed routing
        assert emb.embedding_api_key == "ollama-secret"

    def test_hf_provider_uses_hf_embedding_model(self):
        from src.llm import factory

        with (
            patch("src.runtime_config.embedding_provider", return_value="huggingface"),
            patch("src.runtime_config.hf_embedding_model", return_value="BAAI/bge-m3"),
            patch("src.runtime_config.hf_api_key", return_value="hf-x"),
            patch("src.runtime_config.ollama_embedding_base_url", return_value="http://host:11434"),
            patch("src.runtime_config.embedding_strict", return_value=False),
            patch("src.runtime_config.embedding_dimension", return_value=1024),
        ):
            emb = factory.build_embedding_provider()
        assert isinstance(emb, OllamaRemoteProvider)
        assert emb.embedding_model == "BAAI/bge-m3"
        assert emb.embedding_provider == "huggingface"
