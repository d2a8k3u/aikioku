"""Tests for aclose() on LLMProvider and all implementations."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.base import LLMProvider
from src.llm.ollama import OllamaProvider
from src.llm.ollama_remote import OllamaRemoteProvider
from src.llm.openai_embeddings import OpenAIEmbeddingProvider
from src.llm.openrouter import OpenRouterProvider
from src.llm.router import LLMRouter


class _ConcreteProvider(LLMProvider):
    """Minimal concrete provider for testing the base class."""

    async def complete(self, prompt: str, system: str = "", **kwargs):
        return ""

    async def stream(self, prompt: str, system: str = "", **kwargs):
        yield ""

    async def embed(self, text: str):
        return []

    def is_available(self) -> bool:
        return True


class TestBaseAclose:
    @pytest.mark.asyncio
    async def test_base_aclose_is_noop(self):
        provider = _ConcreteProvider()
        result = await provider.aclose()
        assert result is None


class TestOllamaRemoteAclose:
    @pytest.mark.asyncio
    async def test_ollama_remote_aclose_closes_both_clients(self):
        provider = OllamaRemoteProvider()
        provider._client = AsyncMock()
        provider._embed_client = AsyncMock()

        await provider.aclose()

        provider._client.aclose.assert_awaited_once()
        provider._embed_client.aclose.assert_awaited_once()


class TestOllamaAclose:
    @pytest.mark.asyncio
    async def test_ollama_aclose_closes_client(self):
        provider = OllamaProvider()
        provider._client = AsyncMock()

        await provider.aclose()

        provider._client.aclose.assert_awaited_once()


class TestOpenRouterAclose:
    @pytest.mark.asyncio
    async def test_openrouter_aclose_closes_client_and_embedder(self):
        provider = OpenRouterProvider()
        provider._client = AsyncMock()
        provider._embedder = AsyncMock()

        await provider.aclose()

        provider._client.aclose.assert_awaited_once()
        provider._embedder.aclose.assert_awaited_once()


class TestOpenAIEmbeddingsAclose:
    @pytest.mark.asyncio
    async def test_openai_embeddings_aclose_closes_client(self):
        provider = OpenAIEmbeddingProvider()
        provider._client = AsyncMock()

        await provider.aclose()

        provider._client.aclose.assert_awaited_once()


class TestRouterAclose:
    @pytest.mark.asyncio
    async def test_router_aclose_closes_all_providers(self):
        p1 = AsyncMock()
        p2 = AsyncMock()
        p3 = AsyncMock()
        router = LLMRouter(providers=[p1, p2, p3])

        await router.aclose()

        p1.aclose.assert_awaited_once()
        p2.aclose.assert_awaited_once()
        p3.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_router_aclose_handles_failure(self, caplog):
        p1 = AsyncMock()
        p2 = AsyncMock()
        p2.aclose.side_effect = RuntimeError("boom")
        p3 = AsyncMock()
        router = LLMRouter(providers=[p1, p2, p3])

        with caplog.at_level(logging.WARNING):
            await router.aclose()

        p1.aclose.assert_awaited_once()
        p2.aclose.assert_awaited_once()
        p3.aclose.assert_awaited_once()
        assert any("boom" in record.message for record in caplog.records)
