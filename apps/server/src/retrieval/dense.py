"""DenseRetriever: embedding-based retrieval using an EmbeddingStore and LLMProvider."""

from __future__ import annotations

import asyncio

from src.knowledge.embeddings import EmbeddingStore
from src.llm.base import LLMProvider
from src.retrieval.search_result import SearchResult


class DenseRetriever:
    """Retrieves notes by embedding the query and searching a vector store."""

    def __init__(
        self, embedding_store: EmbeddingStore, llm_provider: LLMProvider
    ) -> None:
        """Create a DenseRetriever.

        Args:
            embedding_store: The vector store to search.
            llm_provider: The LLM provider used to embed the query.
        """
        self._store = embedding_store
        self._llm = llm_provider

    async def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search for notes similar to the query.

        Args:
            query: The search query text.
            limit: Maximum number of results to return.

        Returns:
            A list of SearchResult objects ordered by descending relevance.
        """
        try:
            embedding = await asyncio.wait_for(self._llm.embed(query), timeout=5.0)
        except asyncio.TimeoutError:
            return []
        raw_results = self._store.search(embedding, limit=limit)

        return [
            SearchResult(
                note_id=r["note_id"],
                score=r["score"],
                source="dense",
                snippet=r.get("text", ""),
            )
            for r in raw_results
        ]
