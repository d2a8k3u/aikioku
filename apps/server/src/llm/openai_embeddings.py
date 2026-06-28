"""OpenAI embeddings provider.

A dedicated embedding-only provider (chat methods are unused): ``embed()`` calls
OpenAI's ``POST /v1/embeddings``. Strict/fallback behavior mirrors
``OllamaRemoteProvider`` so the rest of the system treats every embedder the same.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

import httpx

from src.llm.base import LLMProvider
from src.llm.ollama_remote import (
    EmbeddingUnavailableError,
    _deterministic_embedding,
)
from src.llm.urls import OPENAI_EMBEDDINGS, OPENAI_MODELS, join, normalize_base

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(LLMProvider):
    """Embeddings via the OpenAI API (``/v1/embeddings``)."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com",
        api_key: str = "",
        model: str = "text-embedding-3-small",
        embedding_fallback_dim: int = 1024,
        strict_embeddings: bool = False,
    ) -> None:
        self.base_url = normalize_base(base_url, dialect="openai")
        self.api_key = api_key
        self.model = model
        self.strict_embeddings = strict_embeddings
        self._embedding_fallback_dim = embedding_fallback_dim
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    async def embed(self, text: str) -> list[float]:
        embed_url = join(self.base_url, OPENAI_EMBEDDINGS, dialect="openai")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "input": text}
        try:
            resp = await self._client.post(embed_url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    vec = data[0].get("embedding")
                    if vec:
                        return list(vec)
        except Exception as exc:
            logger.debug("OpenAI embed failed for %s at %s: %s", self.model, embed_url, exc)

        if self.strict_embeddings:
            raise EmbeddingUnavailableError(
                f"No real embedding available for OpenAI model '{self.model}' at "
                f"'{embed_url}'. Refusing to return a deterministic hash vector in "
                f"strict mode."
            )
        logger.error(
            "Embedding unavailable for OpenAI model %s at %s, using DETERMINISTIC "
            "(non-semantic) fallback — embeddings are degraded.",
            self.model,
            embed_url,
        )
        return _deterministic_embedding(text, self._embedding_fallback_dim)

    async def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        raise NotImplementedError("OpenAIEmbeddingProvider does not serve completions")

    async def stream(self, prompt: str, system: str = "", **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError("OpenAIEmbeddingProvider does not serve completions")
        yield ""  # pragma: no cover — makes this an async generator

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import urllib.request
            req = urllib.request.Request(
                join(self.base_url, OPENAI_MODELS, dialect="openai"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "python-httpx/0.28.1",
                },
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False
