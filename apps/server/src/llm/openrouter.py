"""OpenRouter LLM provider (OpenAI-compatible chat API).

OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible
``/chat/completions`` endpoint that proxies many hosted models
(``openrouter/owl-alpha``, ``anthropic/claude-3.5-sonnet``, ``openai/gpt-4o``, …) behind one API key.

It has **no** embeddings endpoint, so ``embed()`` delegates to an internal
``OllamaRemoteProvider`` configured with the host-Ollama / HuggingFace embedding
settings — the same strategy the cloud-Ollama provider uses. This keeps the
vector store on a stable embedding model regardless of which chat model serves
completions.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from src.llm.base import LLMProvider
from src.llm.ollama_remote import OllamaRemoteProvider
from src.llm.prompt_cache import build_messages_with_cache
from src.llm.urls import OPENROUTER_CHAT, OPENROUTER_MODELS, join, normalize_base


class OpenRouterProvider(LLMProvider):
    """Chat/completion via OpenRouter; embeddings via host Ollama / HF."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "openrouter/owl-alpha",
        base_url: str = "https://openrouter.ai/api/v1",
        embedding_base_url: str = "",
        embedding_model: str = "mxbai-embed-large",
        embedding_provider: str = "ollama",
        hf_api_key: str = "",
        embedding_fallback_dim: int = 1024,
        strict_embeddings: bool = False,
    ) -> None:
        self.base_url = normalize_base(base_url, dialect="openrouter")
        self.model = model
        self.api_key = api_key
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # Optional OpenRouter attribution headers (ignored if unrecognised).
        headers["HTTP-Referer"] = "http://localhost"
        headers["X-Title"] = "Aikioku"
        self._client = httpx.AsyncClient(timeout=120.0, headers=headers)
        # Embeddings are not served by OpenRouter; reuse the Ollama/HF path.
        self._embedder = OllamaRemoteProvider(
            base_url=embedding_base_url or "http://localhost:11434",
            api_key="",
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            hf_api_key=hf_api_key,
            embedding_fallback_dim=embedding_fallback_dim,
            strict_embeddings=strict_embeddings,
        )

    @staticmethod
    def _messages(prompt: str, system: str) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    async def complete(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        payload = {
            "model": self.model,
            "messages": build_messages_with_cache(system, prompt),
            "stream": False,
            **kwargs,
        }
        resp = await self._client.post(
            join(self.base_url, OPENROUTER_CHAT, dialect="openrouter"), json=payload
        )
        resp.raise_for_status()
        body = resp.json()
        # OpenRouter (OpenAI-compatible) can report failures inside an HTTP-200
        # body ({"error": {...}} with no usable choices). Raise so LLMRouter
        # records a failure and fails over instead of returning a fake "".
        if isinstance(body, dict) and body.get("error"):
            raise httpx.HTTPStatusError(
                f"OpenRouter error: {body['error']}",
                request=resp.request,
                response=resp,
            )
        choices = body.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    async def stream(self, prompt: str, system: str = "", **kwargs: Any) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": build_messages_with_cache(system, prompt),
            "stream": True,
            **kwargs,
        }
        async with self._client.stream(
            "POST", join(self.base_url, OPENROUTER_CHAT, dialect="openrouter"), json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                # Mid-stream errors arrive inside the HTTP-200 SSE channel as a
                # top-level {"error": {...}} field. Raise so LLMRouter.stream
                # fails over to the fallback provider instead of ending empty.
                if isinstance(chunk, dict) and chunk.get("error"):
                    raise RuntimeError(f"OpenRouter stream error: {chunk['error']}")
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if content:
                    yield content

    async def embed(self, text: str) -> list[float]:
        return await self._embedder.embed(text)

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import urllib.request

            req = urllib.request.Request(
                join(self.base_url, OPENROUTER_MODELS, dialect="openrouter"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "python-httpx/0.28.1",
                },
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False
