"""Tests for DenseRetriever."""

from __future__ import annotations

import pytest

from src.retrieval.dense import DenseRetriever
from src.retrieval.search_result import SearchResult


class MockEmbeddingStore:
    """Mock EmbeddingStore that returns configurable search results."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results or []
        self.last_query_embedding: list[float] | None = None
        self.last_limit: int | None = None

    def search(self, query_embedding: list[float], limit: int = 20) -> list[dict]:
        self.last_query_embedding = query_embedding
        self.last_limit = limit
        return self._results[:limit]


class MockLLMProvider:
    """Mock LLMProvider that returns a fixed embedding."""

    def __init__(self, embedding: list[float] | None = None) -> None:
        self._embedding = embedding or [0.1, 0.2, 0.3]
        self.last_text: str | None = None

    async def embed(self, text: str) -> list[float]:
        self.last_text = text
        return self._embedding

    async def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        return ""

    async def stream(self, prompt: str, system: str = "", **kwargs):
        yield ""

    def is_available(self) -> bool:
        return True


@pytest.fixture
def mock_store() -> MockEmbeddingStore:
    return MockEmbeddingStore(
        results=[
            {"note_id": "note-1", "score": 0.95, "distance": 0.05},
            {"note_id": "note-2", "score": 0.87, "distance": 0.13},
            {"note_id": "note-3", "score": 0.72, "distance": 0.28},
        ]
    )


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider(embedding=[0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_search_returns_results_with_correct_fields(
    mock_store: MockEmbeddingStore, mock_llm: MockLLMProvider
) -> None:
    retriever = DenseRetriever(embedding_store=mock_store, llm_provider=mock_llm)
    results = await retriever.search("test query")

    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)

    assert results[0].note_id == "note-1"
    assert results[0].score == 0.95
    assert results[0].source == "dense"

    assert results[1].note_id == "note-2"
    assert results[1].score == 0.87
    assert results[1].source == "dense"

    assert results[2].note_id == "note-3"
    assert results[2].score == 0.72
    assert results[2].source == "dense"


@pytest.mark.asyncio
async def test_search_passes_query_to_embed(
    mock_store: MockEmbeddingStore, mock_llm: MockLLMProvider
) -> None:
    retriever = DenseRetriever(embedding_store=mock_store, llm_provider=mock_llm)
    await retriever.search("my specific query")

    assert mock_llm.last_text == "my specific query"


@pytest.mark.asyncio
async def test_search_passes_embedding_to_store(
    mock_store: MockEmbeddingStore, mock_llm: MockLLMProvider
) -> None:
    retriever = DenseRetriever(embedding_store=mock_store, llm_provider=mock_llm)
    await retriever.search("query")

    assert mock_store.last_query_embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_search_respects_limit(
    mock_store: MockEmbeddingStore, mock_llm: MockLLMProvider
) -> None:
    retriever = DenseRetriever(embedding_store=mock_store, llm_provider=mock_llm)
    results = await retriever.search("query", limit=2)

    assert len(results) == 2
    assert mock_store.last_limit == 2
    assert results[0].note_id == "note-1"
    assert results[1].note_id == "note-2"


@pytest.mark.asyncio
async def test_search_default_limit_is_20(
    mock_store: MockEmbeddingStore, mock_llm: MockLLMProvider
) -> None:
    retriever = DenseRetriever(embedding_store=mock_store, llm_provider=mock_llm)
    await retriever.search("query")

    assert mock_store.last_limit == 20


@pytest.mark.asyncio
async def test_search_returns_empty_for_empty_store(
    mock_llm: MockLLMProvider,
) -> None:
    empty_store = MockEmbeddingStore(results=[])
    retriever = DenseRetriever(embedding_store=empty_store, llm_provider=mock_llm)
    results = await retriever.search("query")

    assert results == []


@pytest.mark.asyncio
async def test_search_carries_chunk_text_as_snippet(
    mock_llm: MockLLMProvider,
) -> None:
    """The matched chunk text from the store must flow into SearchResult.snippet."""
    store = MockEmbeddingStore(results=[{"note_id": "x", "text": "real chunk", "score": 0.9}])
    retriever = DenseRetriever(embedding_store=store, llm_provider=mock_llm)

    results = await retriever.search("query")

    assert len(results) == 1
    assert results[0].note_id == "x"
    assert results[0].snippet == "real chunk"
