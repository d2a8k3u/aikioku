"""Tests for sub-window chunking (BOD 9).

Covers:
- sub_window_chunk() function
- store_note_embeddings() with sub-window pipeline
- EmbeddingStore.add_document() and max-similarity search
- DenseRetriever with sub-window results
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.pipeline import sub_window_chunk, store_note_embeddings
from src.llm.base import LLMProvider
from src.models.note import Note
from src.retrieval.dense import DenseRetriever
from src.retrieval.search_result import SearchResult


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_chroma_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "chroma")


@pytest.fixture
def embedding_store(tmp_chroma_path):
    return EmbeddingStore(db_path=tmp_chroma_path)


@pytest.fixture
def mock_llm():
    """LLM provider that returns distinct embeddings per call."""
    provider = AsyncMock(spec=LLMProvider)
    provider.is_available = MagicMock(return_value=True)

    # Track call count so each sub-window gets a unique embedding.
    _call_count = [0]

    async def _embed_side_effect(text: str) -> list[float]:
        _call_count[0] += 1
        base = float(_call_count[0]) / 100.0
        return [base, 0.5, 0.5]

    provider.embed.side_effect = _embed_side_effect
    return provider


@pytest.fixture
def short_note():
    """Note with < 256 words — should produce 1 sub-window."""
    return Note(
        id="note-short",
        title="Short Note",
        content="This is a short note with just a few words.",
        path="/notes/short.md",
    )


@pytest.fixture
def long_note():
    """Note with > 256 words — should produce multiple overlapping sub-windows."""
    # ~300 words: 300 / 128 stride ≈ 2 windows (256 + 44 leftover → 2 windows)
    content = " ".join(f"word{i}" for i in range(300))
    return Note(
        id="note-long",
        title="Long Note",
        content=content,
        path="/notes/long.md",
    )


# --------------------------------------------------------------------------- #
# Tests: sub_window_chunk()
# --------------------------------------------------------------------------- #


class TestSubWindowChunk:
    """Unit tests for the sub_window_chunk function."""

    def test_empty_text_returns_empty(self):
        result = sub_window_chunk("")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        result = sub_window_chunk("   \n  \t  ")
        assert result == []

    def test_short_text_single_window(self):
        """Text with < 256 words → single window covering all words."""
        text = "hello world " * 10  # 20 words
        result = sub_window_chunk(text)
        assert len(result) == 1
        assert result[0]["start_word"] == 0
        assert result[0]["end_word"] == 20
        assert "hello" in result[0]["text"]

    def test_exactly_window_size_single_window(self):
        """Text with exactly 256 words → single window."""
        words = [f"w{i}" for i in range(256)]
        text = " ".join(words)
        result = sub_window_chunk(text)
        assert len(result) == 1
        assert result[0]["start_word"] == 0
        assert result[0]["end_word"] == 256

    def test_long_text_multiple_windows(self):
        """Text with 300 words → 2 windows (256 + 44 leftover)."""
        words = [f"w{i}" for i in range(300)]
        text = " ".join(words)
        result = sub_window_chunk(text)
        # 300 words, window=256, stride=128:
        # Window 0: words 0-255 (256 words)
        # Window 1: words 128-299 (172 words, last window)
        assert len(result) == 2
        assert result[0]["start_word"] == 0
        assert result[0]["end_word"] == 256
        assert result[1]["start_word"] == 128
        assert result[1]["end_word"] == 300

    def test_overlap_is_50_percent(self):
        """Window 0 and Window 1 should overlap by 128 words (50%)."""
        words = [f"w{i}" for i in range(400)]
        text = " ".join(words)
        result = sub_window_chunk(text)
        # 400 words, window=256, stride=128:
        # Window 0: 0-255
        # Window 1: 128-383
        # Window 2: 256-399
        assert len(result) == 3
        # Overlap between window 0 and 1: words 128-255 (128 words)
        assert result[0]["end_word"] - result[1]["start_word"] == 128
        # Overlap between window 1 and 2: words 256-383 (128 words)
        assert result[1]["end_word"] - result[2]["start_word"] == 128

    def test_very_long_text_many_windows(self):
        """1000 words → ~7 windows."""
        words = [f"w{i}" for i in range(1000)]
        text = " ".join(words)
        result = sub_window_chunk(text)
        # 1000 words, window=256, stride=128:
        # Starts: 0, 128, 256, 384, 512, 640, 768 → 7 windows
        # Last window: 768-999 (232 words)
        assert len(result) == 7
        assert result[-1]["start_word"] == 768
        assert result[-1]["end_word"] == 1000

    def test_window_text_is_correct_substring(self):
        """Each window's text should be the exact words in that range."""
        words = [f"w{i}" for i in range(300)]
        text = " ".join(words)
        result = sub_window_chunk(text)
        # Window 0: words 0-255
        expected_0 = " ".join(words[0:256])
        assert result[0]["text"] == expected_0
        # Window 1: words 128-299
        expected_1 = " ".join(words[128:300])
        assert result[1]["text"] == expected_1


# --------------------------------------------------------------------------- #
# Tests: store_note_embeddings() with sub-windows
# --------------------------------------------------------------------------- #


class TestStoreNoteEmbeddingsSubWindow:
    """Integration tests for the sub-window embedding pipeline."""

    @pytest.mark.asyncio
    async def test_short_note_stores_one_sub_window(self, embedding_store, mock_llm, short_note):
        await store_note_embeddings(short_note, mock_llm, embedding_store)
        # Short note (< 256 words) → 1 semantic chunk → 1 sub-window
        assert embedding_store.count() == 1

    @pytest.mark.asyncio
    async def test_long_note_stores_multiple_sub_windows(
        self, embedding_store, mock_llm, long_note
    ):
        await store_note_embeddings(long_note, mock_llm, embedding_store)
        # Long note (300 words) → 1 semantic chunk → 2 sub-windows
        assert embedding_store.count() >= 2

    @pytest.mark.asyncio
    async def test_sub_windows_have_distinct_embeddings(self, embedding_store, mock_llm, long_note):
        """Each sub-window gets its own embedding call."""
        await store_note_embeddings(long_note, mock_llm, embedding_store)
        # embed() should be called once per sub-window
        assert mock_llm.embed.call_count >= 2

    @pytest.mark.asyncio
    async def test_reembed_is_idempotent(self, embedding_store, mock_llm, short_note):
        """Calling store_note_embeddings twice should not double the count."""
        await store_note_embeddings(short_note, mock_llm, embedding_store)
        count1 = embedding_store.count()

        await store_note_embeddings(short_note, mock_llm, embedding_store)
        count2 = embedding_store.count()

        # Should be the same (old embeddings deleted before re-embed)
        assert count2 == count1

    @pytest.mark.asyncio
    async def test_empty_note_skips_embedding(self, embedding_store, mock_llm):
        empty_note = Note(
            id="note-empty",
            title="Empty",
            content="",
            path="/notes/empty.md",
        )
        await store_note_embeddings(empty_note, mock_llm, embedding_store)
        assert embedding_store.count() == 0
        mock_llm.embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_only_note_skips_embedding(self, embedding_store, mock_llm):
        ws_note = Note(
            id="note-ws",
            title="Whitespace",
            content="   \n  \t  ",
            path="/notes/ws.md",
        )
        await store_note_embeddings(ws_note, mock_llm, embedding_store)
        assert embedding_store.count() == 0


# --------------------------------------------------------------------------- #
# Tests: EmbeddingStore.add_document() and max-similarity search
# --------------------------------------------------------------------------- #


class TestAddDocument:
    """Tests for the new add_document() method."""

    def test_add_document_stores_single_entry(self, embedding_store):
        embedding_store.add_document(
            doc_id="note-1#c0#sw0",
            text="sub-window text",
            embedding=[0.1, 0.2, 0.3],
            metadata={"note_id": "note-1", "chunk_index": 0, "sub_window_index": 0},
        )
        assert embedding_store.count() == 1

    def test_add_document_preserves_metadata(self, embedding_store):
        embedding_store.add_document(
            doc_id="note-x#c1#sw2",
            text="some text",
            embedding=[0.5, 0.5, 0.5],
            metadata={
                "note_id": "note-x",
                "chunk_index": 1,
                "sub_window_index": 2,
                "window_start": 128,
                "window_end": 256,
            },
        )
        # Search should find it
        results = embedding_store.search([0.5, 0.5, 0.5])
        assert len(results) == 1
        assert results[0]["note_id"] == "note-x"
        assert results[0]["text"] == "some text"


class TestMaxSimilaritySearch:
    """Tests for max-similarity grouping across sub-windows."""

    def test_max_similarity_picks_best_sub_window(self, embedding_store):
        """When a note has multiple sub-windows, the best match wins."""
        # Sub-window 0: far from query
        embedding_store.add_document(
            doc_id="n1#c0#sw0",
            text="irrelevant text",
            embedding=[0.0, 1.0, 0.0],  # orthogonal to query
            metadata={"note_id": "n1"},
        )
        # Sub-window 1: close to query
        embedding_store.add_document(
            doc_id="n1#c0#sw1",
            text="highly relevant text",
            embedding=[0.99, 0.01, 0.0],  # nearly identical to query
            metadata={"note_id": "n1"},
        )

        results = embedding_store.search([1.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0]["note_id"] == "n1"
        # Should return the best sub-window text
        assert results[0]["text"] == "highly relevant text"
        assert results[0]["score"] > 0.9

    def test_max_similarity_across_multiple_notes(self, embedding_store):
        """Each note gets its max score; results ordered by that max."""
        # Note A: best sub-window score ~0.99
        embedding_store.add_document("a#c0#sw0", "A bad", [0.0, 1.0, 0.0], {"note_id": "a"})
        embedding_store.add_document("a#c0#sw1", "A good", [0.99, 0.01, 0.0], {"note_id": "a"})
        # Note B: best sub-window score ~0.95
        embedding_store.add_document("b#c0#sw0", "B ok", [0.95, 0.05, 0.0], {"note_id": "b"})
        # Note C: best sub-window score ~0.50
        embedding_store.add_document("c#c0#sw0", "C far", [0.5, 0.5, 0.0], {"note_id": "c"})

        results = embedding_store.search([1.0, 0.0, 0.0], limit=3)
        assert len(results) == 3
        assert results[0]["note_id"] == "a"
        assert results[1]["note_id"] == "b"
        assert results[2]["note_id"] == "c"
        assert results[0]["score"] > results[1]["score"] > results[2]["score"]

    def test_search_respects_limit_with_sub_windows(self, embedding_store):
        """Limit applies to number of notes returned, not sub-windows."""
        for i in range(10):
            # Each note gets 2 sub-windows
            embedding_store.add_document(
                f"n{i}#c0#sw0",
                f"note {i} window 0",
                [float(i) / 20.0, 0.5, 0.5],
                {"note_id": f"n{i}"},
            )
            embedding_store.add_document(
                f"n{i}#c0#sw1",
                f"note {i} window 1",
                [float(i) / 20.0, 0.5, 0.5],
                {"note_id": f"n{i}"},
            )

        results = embedding_store.search([0.5, 0.5, 0.5], limit=5)
        assert len(results) == 5

    def test_search_empty_with_sub_windows(self, embedding_store):
        results = embedding_store.search([0.1, 0.2, 0.3])
        assert results == []


# --------------------------------------------------------------------------- #
# Tests: DenseRetriever with sub-window results
# --------------------------------------------------------------------------- #


class MockEmbeddingStore:
    """Mock store returning sub-window-style results."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self._results = results or []

    def search(self, query_embedding: list[float], limit: int = 20) -> list[dict]:
        return self._results[:limit]


class MockLLM:
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def complete(self, prompt: str, system: str = "", **kwargs) -> str:
        return ""

    async def stream(self, prompt: str, system: str = "", **kwargs):
        yield ""

    def is_available(self) -> bool:
        return True


class TestDenseRetrieverSubWindow:
    """DenseRetriever works transparently with sub-window search results."""

    @pytest.mark.asyncio
    async def test_dense_retriever_handles_sub_window_results(self):
        """The store now returns max-similarity results; DenseRetriever
        should map them to SearchResult objects unchanged."""
        store = MockEmbeddingStore(
            results=[
                {"note_id": "n1", "text": "best sub-window of n1", "score": 0.95},
                {"note_id": "n2", "text": "best sub-window of n2", "score": 0.87},
            ]
        )
        llm = MockLLM()
        retriever = DenseRetriever(embedding_store=store, llm_provider=llm)

        results = await retriever.search("query")
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].note_id == "n1"
        assert results[0].score == 0.95
        assert results[0].snippet == "best sub-window of n1"
        assert results[0].source == "dense"
        assert results[1].note_id == "n2"
        assert results[1].score == 0.87
        assert results[1].snippet == "best sub-window of n2"
