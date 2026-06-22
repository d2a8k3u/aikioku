"""Mock LLM provider tests — avoid needing a real Ollama server."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm.ollama import OllamaProvider
from src.llm.ollama_remote import OllamaRemoteProvider
from src.llm.openrouter import OpenRouterProvider


class TestOllamaProviderMocked:
    """Test OllamaProvider with mocked httpx.AsyncClient.post."""

    @pytest.mark.asyncio
    async def test_complete_returns_response_text(self):
        provider = OllamaProvider(base_url="http://fake", model="m")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "hello back"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)) as mock_post:
            result = await provider.complete("hi", system="be nice")
            assert result == "hello back"
            mock_post.assert_awaited_once()
            args, kwargs = mock_post.call_args
            assert kwargs["json"]["model"] == "m"
            assert kwargs["json"]["prompt"] == "hi"
            assert kwargs["json"]["system"] == "be nice"
            assert kwargs["json"]["stream"] is False

    @pytest.mark.asyncio
    async def test_complete_returns_empty_when_no_response_key(self):
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)):
            result = await provider.complete("x")
            assert result == ""

    @pytest.mark.asyncio
    async def test_complete_raises_on_http_error(self):
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "fail", request=MagicMock(), response=MagicMock(status_code=500)
        )

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)):
            with pytest.raises(httpx.HTTPStatusError):
                await provider.complete("x")

    @pytest.mark.asyncio
    async def test_embed_returns_embedding_list(self):
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)):
            result = await provider.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_returns_empty_when_no_embedding_key(self):
        provider = OllamaProvider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)):
            result = await provider.embed("test")
            assert result == []

    @pytest.mark.asyncio
    async def test_stream_yields_text_chunks(self):
        provider = OllamaProvider()
        raw_lines = [
            json.dumps({"response": "chunk1"}),
            json.dumps({"response": "chunk2"}),
            json.dumps({"done": True}),
        ]
        # Build an async iterator that yields lines
        async def _aiter_lines():
            for line in raw_lines:
                yield line

        fake_resp = MagicMock()
        fake_resp.aiter_lines = _aiter_lines
        fake_resp.__aenter__ = AsyncMock(return_value=fake_resp)
        fake_resp.__aexit__ = AsyncMock(return_value=False)

        with patch.object(provider._client, "stream", return_value=fake_resp):
            chunks = [chunk async for chunk in provider.stream("hi")]
            assert chunks == ["chunk1", "chunk2"]

    def test_is_available_true(self):
        provider = OllamaProvider(base_url="http://localhost:99999", model="m")
        with patch("urllib.request.urlopen", return_value=MagicMock()):
            assert provider.is_available() is True

    def test_is_available_false_on_exception(self):
        provider = OllamaProvider(base_url="http://localhost:99999", model="m")
        with patch("urllib.request.urlopen", side_effect=Exception("nope")):
            assert provider.is_available() is False


class TestOllamaRemoteProviderMocked:
    """Test OllamaRemoteProvider with mocked httpx.AsyncClient.post."""

    @pytest.mark.asyncio
    async def test_complete_uses_auth_header(self):
        provider = OllamaRemoteProvider(base_url="http://remote", model="r", api_key="secret")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "remote hello"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)) as mock_post:
            result = await provider.complete("hi")
            assert result == "remote hello"
            _, kwargs = mock_post.call_args
            assert kwargs["json"]["model"] == "r"

    @pytest.mark.asyncio
    async def test_complete_without_api_key(self):
        provider = OllamaRemoteProvider(base_url="http://remote", model="r")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)):
            result = await provider.complete("x")
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_embed_remote(self):
        # embed() uses a dedicated embedding endpoint via a fresh httpx client
        # (no cloud bearer), so patch at the class level rather than _client.
        provider = OllamaRemoteProvider(
            api_key="k",
            embedding_base_url="http://host.docker.internal:11434",
            embedding_model="mxbai-embed-large",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.9]]}

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            result = await provider.embed("x")
            assert result == [0.9]

    @pytest.mark.asyncio
    async def test_stream_remote(self):
        provider = OllamaRemoteProvider(api_key="k")
        raw_lines = [
            json.dumps({"response": "A"}),
            json.dumps({"response": "B"}),
        ]
        async def _aiter_lines():
            for line in raw_lines:
                yield line

        fake_resp = MagicMock()
        fake_resp.aiter_lines = _aiter_lines
        fake_resp.__aenter__ = AsyncMock(return_value=fake_resp)
        fake_resp.__aexit__ = AsyncMock(return_value=False)

        with patch.object(provider._client, "stream", return_value=fake_resp):
            chunks = [c async for c in provider.stream("q")]
            assert chunks == ["A", "B"]

    def test_is_available_true_with_api_key(self):
        provider = OllamaRemoteProvider(base_url="http://r", api_key="k")
        with patch("urllib.request.urlopen", return_value=MagicMock()):
            assert provider.is_available() is True

    def test_is_available_false_on_exception(self):
        provider = OllamaRemoteProvider(base_url="http://r")
        with patch("urllib.request.urlopen", side_effect=Exception("down")):
            assert provider.is_available() is False


class TestBaseUrlNormalizationResolvesEndpoints:
    """A messy user-entered base URL must resolve to the canonical endpoint."""

    @pytest.mark.asyncio
    async def test_ollama_complete_strips_v1_and_appends_native_path(self):
        provider = OllamaProvider(base_url="http://fake/v1", model="m")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)) as mock_post:
            await provider.complete("hi")

        args, kwargs = mock_post.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "http://fake/api/generate"

    @pytest.mark.asyncio
    async def test_ollama_remote_complete_strips_trailing_api(self):
        provider = OllamaRemoteProvider(base_url="https://api.ollama.com/api", model="r", api_key="k")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)) as mock_post:
            await provider.complete("hi")

        args, kwargs = mock_post.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "https://api.ollama.com/api/generate"

    @pytest.mark.asyncio
    async def test_openrouter_complete_adds_api_v1_when_missing(self):
        provider = OpenRouterProvider(base_url="https://openrouter.ai", api_key="k")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(provider._client, "post", new=AsyncMock(return_value=mock_resp)) as mock_post:
            await provider.complete("hi")

        args, kwargs = mock_post.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "https://openrouter.ai/api/v1/chat/completions"


class TestOllamaProviderPatchAtClassLevel:
    """Alternative patching strategy — patch httpx.AsyncClient.post globally."""

    @pytest.mark.asyncio
    async def test_complete_patching_async_client_class(self):
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_post.return_value = MagicMock(
                json=lambda: {"response": "class-patched"},
                raise_for_status=MagicMock(),
            )
            provider = OllamaProvider()
            result = await provider.complete("hi")
            assert result == "class-patched"
            assert mock_post.await_count == 1

    @pytest.mark.asyncio
    async def test_embed_patching_async_client_class(self):
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_post.return_value = MagicMock(
                json=lambda: {"embedding": [1.0, 2.0]},
                raise_for_status=MagicMock(),
            )
            provider = OllamaProvider()
            result = await provider.embed("hi")
            assert result == [1.0, 2.0]
