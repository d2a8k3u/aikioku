"""Tests for the real semantic embedding path.

Embeddings must come from the host Ollama ``mxbai-embed-large`` model (1024-dim)
via a DEDICATED embedding base url, separate from the cloud chat ``base_url``.
The deterministic-hash fallback must become a surfaced/hard failure in strict
mode instead of a silent steady state.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm import ollama_remote
from src.llm.ollama_remote import (
    EmbeddingUnavailableError,
    OllamaRemoteProvider,
)


class TestEmbedUsesDedicatedEndpoint:
    """embed() must hit the embedding base url with the embedding model."""

    @pytest.mark.asyncio
    async def test_embed_posts_to_embedding_base_url_with_model(self):
        provider = OllamaRemoteProvider(
            base_url="https://api.ollama.com",
            api_key="cloud-secret",
            embedding_base_url="http://host.docker.internal:11434",
            embedding_model="mxbai-embed-large",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:
            result = await provider.embed("hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_awaited_once()
        args, kwargs = mock_post.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "http://host.docker.internal:11434/api/embed"
        assert kwargs["json"]["model"] == "mxbai-embed-large"
        assert kwargs["json"]["input"] == "hello world"

    @pytest.mark.asyncio
    async def test_embed_normalizes_messy_embedding_base(self):
        """A base with an accidental /api or /v1 still resolves to /api/embed once."""
        provider = OllamaRemoteProvider(
            base_url="https://api.ollama.com",
            embedding_base_url="http://host.docker.internal:11434/api",
            embedding_model="mxbai-embed-large",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.1]]}

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:
            await provider.embed("hi")

        args, kwargs = mock_post.call_args
        url = args[0] if args else kwargs["url"]
        assert url == "http://host.docker.internal:11434/api/embed"

    @pytest.mark.asyncio
    async def test_embed_parses_legacy_embedding_shape(self):
        provider = OllamaRemoteProvider(
            embedding_base_url="http://host.docker.internal:11434",
            embedding_model="mxbai-embed-large",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": [0.5, 0.6]}

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            result = await provider.embed("x")

        assert result == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_sends_bearer_when_embedding_api_key_set(self):
        """A remote Ollama embedder authenticates with its own embedding_api_key."""
        provider = OllamaRemoteProvider(
            embedding_base_url="https://api.ollama.com",
            embedding_model="mxbai-embed-large",
            embedding_api_key="emb-secret",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[1.0]]}

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:
            await provider.embed("hi")

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer emb-secret"

    @pytest.mark.asyncio
    async def test_embed_does_not_send_cloud_bearer_to_host(self):
        """Host Ollama needs no Authorization; the cloud bearer must not leak."""
        provider = OllamaRemoteProvider(
            base_url="https://api.ollama.com",
            api_key="cloud-secret",
            embedding_base_url="http://host.docker.internal:11434",
            embedding_model="mxbai-embed-large",
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[1.0]]}

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)) as mock_post:
            await provider.embed("hi")

        _, kwargs = mock_post.call_args
        headers = kwargs.get("headers")
        # Either no headers passed, or headers without a cloud bearer.
        if headers:
            assert "Authorization" not in headers or "cloud-secret" not in headers.get(
                "Authorization", ""
            )


class TestStrictModeRaises:
    """strict_embeddings=True must raise instead of returning a hash vector."""

    @pytest.mark.asyncio
    async def test_strict_raises_on_total_failure(self):
        provider = OllamaRemoteProvider(
            embedding_base_url="http://host.docker.internal:11434",
            embedding_model="mxbai-embed-large",
            strict_embeddings=True,
            hf_api_key="",  # no HF fallback
        )
        with patch(
            "httpx.AsyncClient.post",
            new=AsyncMock(side_effect=httpx.ConnectError("down")),
        ):
            with pytest.raises(EmbeddingUnavailableError) as excinfo:
                await provider.embed("hello world")

        # Message should name the model and the embedding base url.
        msg = str(excinfo.value)
        assert "mxbai-embed-large" in msg
        assert "host.docker.internal" in msg

    @pytest.mark.asyncio
    async def test_strict_raises_on_non_200(self):
        provider = OllamaRemoteProvider(
            embedding_base_url="http://host.docker.internal:11434",
            embedding_model="mxbai-embed-large",
            strict_embeddings=True,
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"error": "unauthorized"}

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            with pytest.raises(EmbeddingUnavailableError):
                await provider.embed("x")

    @pytest.mark.asyncio
    async def test_strict_skips_hf_fallback(self):
        """In strict mode the HF fallback (wrong dimension) must NOT be attempted."""
        provider = OllamaRemoteProvider(
            embedding_base_url="http://host.docker.internal:11434",
            embedding_model="mxbai-embed-large",
            strict_embeddings=True,
            hf_api_key="hf-token-set",  # would normally trigger HF fallback
        )
        with (
            patch.object(provider, "_hf_embed", new=AsyncMock(return_value=[0.0] * 384)) as mock_hf,
            patch(
                "httpx.AsyncClient.post",
                new=AsyncMock(side_effect=httpx.ConnectError("down")),
            ),
        ):
            with pytest.raises(EmbeddingUnavailableError):
                await provider.embed("x")

        mock_hf.assert_not_awaited()


class TestNonStrictDeterministicFallback:
    """strict_embeddings=False must return a deterministic vector and count it."""

    @pytest.mark.asyncio
    async def test_non_strict_returns_fallback_and_increments_counter(self):
        provider = OllamaRemoteProvider(
            embedding_base_url="http://host.docker.internal:11434",
            embedding_model="mxbai-embed-large",
            strict_embeddings=False,
            embedding_fallback_dim=1024,
            hf_api_key="",
        )
        before = ollama_remote.DETERMINISTIC_FALLBACK_COUNT

        with patch(
            "httpx.AsyncClient.post",
            new=AsyncMock(side_effect=httpx.ConnectError("down")),
        ):
            result = await provider.embed("hello world")

        assert len(result) == 1024
        assert all(isinstance(v, float) for v in result)
        assert ollama_remote.DETERMINISTIC_FALLBACK_COUNT == before + 1

    @pytest.mark.asyncio
    async def test_non_strict_fallback_is_deterministic(self):
        provider = OllamaRemoteProvider(
            embedding_base_url="http://host.docker.internal:11434",
            strict_embeddings=False,
            embedding_fallback_dim=1024,
        )
        with patch(
            "httpx.AsyncClient.post",
            new=AsyncMock(side_effect=httpx.ConnectError("down")),
        ):
            a = await provider.embed("same text")
            b = await provider.embed("same text")

        assert a == b


class TestEmbeddingBaseUrlFallback:
    """When embedding_base_url is empty, it falls back to base_url."""

    def test_empty_embedding_base_url_falls_back_to_base_url(self):
        provider = OllamaRemoteProvider(base_url="http://chat-host:11434")
        assert provider.embedding_base_url == "http://chat-host:11434"

    def test_explicit_embedding_base_url_is_used(self):
        provider = OllamaRemoteProvider(
            base_url="https://api.ollama.com",
            embedding_base_url="http://host.docker.internal:11434",
        )
        assert provider.embedding_base_url == "http://host.docker.internal:11434"


class TestConfigDefaults:
    """Config defaults."""

    def test_embedding_defaults(self):
        from src.config import Settings

        s = Settings()
        assert s.embedding_dimension == 1024
        assert s.ollama_embedding_model == "mxbai-embed-large"
        assert s.ollama_embedding_base_url == "http://host.docker.internal:11434"
        assert s.embedding_strict is True


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@pytest.mark.integration
class TestRealHostEmbeddings:
    """Host-gated integration test against the REAL mxbai-embed-large model.

    Skips gracefully if the host Ollama / model is unreachable so it never
    fails CI without the host.
    """

    @pytest.mark.asyncio
    async def test_semantic_similarity_ordering(self):

        base = "http://host.docker.internal:11434"
        model = "mxbai-embed-large"

        # Probe reachability + model availability; skip if not usable.
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{base}/api/embed",
                    json={"model": model, "input": "probe"},
                    headers={},
                )
            if resp.status_code != 200:
                pytest.skip(
                    f"Host embedder {model} at {base} returned {resp.status_code}; skipping."
                )
            probe = resp.json().get("embeddings", [[]])[0]
            if not probe:
                pytest.skip("Host embedder returned empty vector; skipping.")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Host embedder {model} at {base} unreachable: {exc}")

        provider = OllamaRemoteProvider(
            base_url="https://api.ollama.com",
            embedding_base_url=base,
            embedding_model=model,
            strict_embeddings=True,
        )

        v_python = await provider.embed("the python programming language")
        v_coding = await provider.embed("coding in python")
        v_cake = await provider.embed("a recipe for chocolate cake")

        assert len(v_python) == 1024

        sim_related = _cosine(v_python, v_coding)
        sim_unrelated = _cosine(v_python, v_cake)

        assert sim_related > sim_unrelated, (
            f"Expected related pair more similar: related={sim_related:.4f} "
            f"unrelated={sim_unrelated:.4f}"
        )
