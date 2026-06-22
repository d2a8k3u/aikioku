"""Ollama LLM provider (local)."""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from src.llm.base import LLMProvider
from src.llm.urls import (
    OLLAMA_EMBEDDINGS,
    OLLAMA_GENERATE,
    OLLAMA_TAGS,
    join,
    normalize_base,
)


class OllamaProvider(LLMProvider):
    """Local LLM provider via Ollama."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3") -> None:
        self.base_url = normalize_base(base_url, dialect="ollama")
        self.model = model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            **kwargs,
        }
        resp = await self._client.post(join(self.base_url, OLLAMA_GENERATE, dialect="ollama"), json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")

    async def stream(self, prompt: str, system: str = "", **kwargs) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            **kwargs,
        }
        async with self._client.stream("POST", join(self.base_url, OLLAMA_GENERATE, dialect="ollama"), json=payload) as resp:
            async for line in resp.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    async def embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        resp = await self._client.post(join(self.base_url, OLLAMA_EMBEDDINGS, dialect="ollama"), json=payload)
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    def is_available(self) -> bool:
        try:
            import urllib.request
            urllib.request.urlopen(join(self.base_url, OLLAMA_TAGS, dialect="ollama"), timeout=5)
            return True
        except Exception:
            return False
