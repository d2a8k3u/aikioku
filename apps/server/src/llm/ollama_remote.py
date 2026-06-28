"""Ollama remote LLM provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, AsyncIterator, cast

import httpx

from src.llm.base import LLMProvider
from src.llm.urls import (
    OLLAMA_EMBED,
    OLLAMA_GENERATE,
    OLLAMA_TAGS,
    join,
    normalize_base,
)

logger = logging.getLogger(__name__)

# Number of times embed() fell back to the deterministic (non-semantic) vector.
# Incremented only in non-strict mode so degraded operation is observable.
DETERMINISTIC_FALLBACK_COUNT: int = 0


class EmbeddingUnavailableError(RuntimeError):
    """Raised when no real embedding could be produced and strict mode is on."""


class OllamaRemoteProvider(LLMProvider):
    """Remote LLM provider via Ollama Cloud API.

    Uses a separate ``embedding_model`` for ``/api/embed`` calls because chat
    models (e.g. ``kimi-k2.6:cloud``) do not expose that endpoint.  Falls back
    to HuggingFace Inference API (free tier) when an API key is configured,
    and finally to a deterministic hash-based embedding so downstream code keeps
    working.
    """

    def __init__(
        self,
        base_url: str = "https://api.ollama.com",
        model: str = "kimi-k2.6:cloud",
        api_key: str = "",
        embedding_model: str = "mxbai-embed-large",
        embedding_base_url: str = "",
        embedding_provider: str = "ollama",
        embedding_api_key: str = "",
        hf_api_key: str = "",
        embedding_fallback_dim: int = 1024,
        strict_embeddings: bool = False,
    ) -> None:
        self.base_url = normalize_base(base_url, dialect="ollama")
        self.model = model
        self.api_key = api_key
        self.embedding_model = embedding_model
        # Dedicated embedding endpoint (host Ollama). Falls back to chat base_url.
        self.embedding_base_url = normalize_base(embedding_base_url or base_url, dialect="ollama")
        self.embedding_provider = embedding_provider
        # Bearer for a REMOTE Ollama embedding endpoint. Separate from the chat
        # ``api_key`` so a local/host embedder never receives the cloud chat bearer.
        self.embedding_api_key = embedding_api_key
        self.strict_embeddings = strict_embeddings
        self._hf_api_key = hf_api_key
        self._embedding_fallback_dim = embedding_fallback_dim
        self._client = httpx.AsyncClient(
            timeout=120.0,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )
        # Dedicated embed client for connection reuse (embedding calls are
        # frequent and previously created a new AsyncClient per call).
        self._embed_client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {embedding_api_key}"} if embedding_api_key else {},
        )

    async def complete(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            **kwargs,
        }
        resp = await self._client.post(
            join(self.base_url, OLLAMA_GENERATE, dialect="ollama"), json=payload
        )
        resp.raise_for_status()
        return cast(str, resp.json().get("response", ""))

    async def stream(self, prompt: str, system: str = "", **kwargs: Any) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            **kwargs,
        }
        async with self._client.stream(
            "POST", join(self.base_url, OLLAMA_GENERATE, dialect="ollama"), json=payload
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    async def embed(self, text: str) -> list[float]:
        """Generate a semantic embedding vector.

        Priority order depends on ``embedding_provider``:

        ``"ollama"`` (default):
          1. Host Ollama ``/api/embed`` endpoint
          2. HuggingFace Inference API fallback (if HF_API_KEY set, non-strict)
          3. Deterministic hash fallback (or raise in strict mode)

        ``"huggingface"``:
          1. HuggingFace Inference API
          2. Deterministic hash fallback (or raise in strict mode)
        """
        global DETERMINISTIC_FALLBACK_COUNT

        if self.embedding_provider == "huggingface":
            # 1. HuggingFace Inference API
            if self._hf_api_key:
                try:
                    vec = await self._hf_embed(text)
                    if vec:
                        logger.debug("HF embed success (%d dims)", len(vec))
                        return vec
                except Exception as exc:
                    logger.debug("HF embed failed: %s", exc)
            # 2. Final fallback
            if self.strict_embeddings:
                raise EmbeddingUnavailableError(
                    f"No real embedding available via HuggingFace for model "
                    f"'{self.embedding_model}'. Refusing to return a "
                    f"deterministic hash vector in strict mode."
                )
            logger.error(
                "Embedding unavailable via HuggingFace for %s, using "
                "DETERMINISTIC (non-semantic) fallback — embeddings are degraded.",
                self.embedding_model,
            )
            DETERMINISTIC_FALLBACK_COUNT += 1
            return _deterministic_embedding(text, self._embedding_fallback_dim)

        # Default: "ollama" — current behavior
        # 1. Host Ollama embedding model (no cloud bearer).
        embed_url = join(self.embedding_base_url, OLLAMA_EMBED, dialect="ollama")
        payload = {"model": self.embedding_model, "input": text}
        headers = (
            {"Authorization": f"Bearer {self.embedding_api_key}"} if self.embedding_api_key else {}
        )
        try:
            resp = await self._embed_client.post(
                embed_url,
                json=payload,
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) > 0:
                    vec = list(embeddings[0])
                    logger.debug(
                        "Ollama embed success via %s (%d dims)",
                        self.embedding_model,
                        len(vec),
                    )
                    return vec
                # Old response shape fallback
                vec = data.get("embedding", [])
                if vec:
                    return list(vec)
        except Exception as exc:
            logger.debug(
                "Ollama embed failed for %s at %s: %s",
                self.embedding_model,
                self.embedding_base_url,
                exc,
            )

        # 2. HuggingFace Inference API fallback.
        # Skipped in strict mode: the HF model (all-MiniLM, 384-dim) does not match
        # the configured embedding dimension, so silently substituting it would
        # corrupt the vector store. In strict mode we prefer to fail loud below.
        if self._hf_api_key and not self.strict_embeddings:
            try:
                vec = await self._hf_embed(text)
                if vec:
                    logger.debug("HF embed fallback success (%d dims)", len(vec))
                    return vec
            except Exception as exc:
                logger.debug("HF embed fallback failed: %s", exc)

        # 3. Final fallback
        if self.strict_embeddings:
            raise EmbeddingUnavailableError(
                f"No real embedding available for model '{self.embedding_model}' "
                f"at '{embed_url}' (HF fallback "
                f"{'tried' if self._hf_api_key else 'disabled'}). Refusing to "
                f"return a deterministic hash vector in strict mode."
            )
        logger.error(
            "Embedding unavailable for %s at %s, using DETERMINISTIC (non-semantic) "
            "fallback — embeddings are degraded.",
            self.embedding_model,
            self.embedding_base_url,
        )
        DETERMINISTIC_FALLBACK_COUNT += 1
        return _deterministic_embedding(text, self._embedding_fallback_dim)

    async def _hf_embed(self, text: str, model: str = "") -> list[float]:
        """Call HuggingFace Inference API (feature-extraction) for embeddings.

        Uses the configured ``embedding_model`` unless an explicit model is given.
        """
        model = model or self.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        headers = {
            "Authorization": f"Bearer {self._hf_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": text}
        # HF decommissioned api-inference.huggingface.co; the hf-inference provider
        # now serves feature-extraction via the router host.
        url = (
            f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
        )
        # HF serverless intermittently times out or returns 429/5xx under bursty
        # concurrent load (e.g. backfill). Retry transient failures with backoff so
        # a single hiccup doesn't permanently drop a memory's vector.
        retryable_status = {429, 500, 502, 503, 504}
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._embed_client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                # HF returns a list of vectors (one per sentence) or a single vector
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list):
                        return list(data[0])  # list of vectors → first vector
                    return list(data)  # single vector
                raise ValueError(f"Unexpected HF response: {data}")
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in retryable_status:
                    raise
                last_exc = exc
            except httpx.TransportError as exc:
                last_exc = exc
            if attempt < 2:
                await asyncio.sleep(0.5 * 2**attempt)
        assert last_exc is not None
        raise last_exc

    def is_available(self) -> bool:
        try:
            import urllib.request

            req = urllib.request.Request(
                join(self.base_url, OLLAMA_TAGS, dialect="ollama"),
                headers={
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                    "User-Agent": "python-httpx/0.28.1",
                },
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False


def _deterministic_embedding(text: str, dim: int = 768) -> list[float]:
    """Create a deterministic pseudo-embedding from text.

    Uses SHA-256 to seed a simple LCG.  This is NOT a semantic embedding,
    but it provides consistent vectors when no real model is available.
    """
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes[:8], "big")
    vec: list[float] = []
    a = 1664525
    c = 1013904223
    m = 2**32
    for _ in range(dim):
        seed = (a * seed + c) % m
        vec.append((seed / m) * 2.0 - 1.0)
    return vec
