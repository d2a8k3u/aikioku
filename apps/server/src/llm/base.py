"""Base LLM provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        """Generate a completion. Returns the response text."""
        ...

    @abstractmethod
    async def stream(self, prompt: str, system: str = "", **kwargs) -> AsyncIterator[str]:
        """Generate a streaming completion. Yields text chunks."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        ...
