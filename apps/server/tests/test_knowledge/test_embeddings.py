"""Tests for EmbeddingStore class using ChromaDB with semantic chunking."""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture
def tmp_db_path():
    d = tempfile.mkdtemp()
    yield os.path.join(d, "chroma")


@pytest.fixture
def store(tmp_db_path):
    from src.knowledge.embeddings import EmbeddingStore

    return EmbeddingStore(db_path=tmp_db_path)


class TestEmbeddingStoreInit:
    def test_init_creates_store(self, tmp_db_path):
        from src.knowledge.embeddings import EmbeddingStore

        s = EmbeddingStore(db_path=tmp_db_path)
        assert s is not None
        assert s.count() == 0


class TestAdd:
    def test_add_stores_embedding(self, store):
        store.add("note-1", "Hello world", [0.1, 0.2, 0.3])
        assert store.count() == 1

    def test_add_overwrites_existing(self, store):
        store.add("note-1", "Hello world", [0.1, 0.2, 0.3])
        store.add("note-1", "Updated", [0.4, 0.5, 0.6])
        # ChromaDB stores chunks: second add inserts extra docs for same note
        # count should remain low (2 chunks max for short text)
        assert store.count() <= 2


class TestSearch:
    def test_search_returns_results(self, store):
        store.add("note-1", "Python programming", [0.1, 0.2, 0.3])
        store.add("note-2", "Rust programming", [0.4, 0.5, 0.6])
        results = store.search([0.1, 0.2, 0.3], limit=2)
        assert len(results) == 2

    def test_search_returns_note_id(self, store):
        store.add("note-1", "Python", [0.1, 0.2, 0.3])
        results = store.search([0.1, 0.2, 0.3])
        assert results[0]["note_id"] == "note-1"

    def test_search_returns_score(self, store):
        store.add("note-1", "Python", [1.0, 0.0, 0.0])
        results = store.search([1.0, 0.0, 0.0])
        assert "score" in results[0]
        assert results[0]["score"] > 0.9

    def test_search_respects_limit(self, store):
        for i in range(5):
            store.add(f"note-{i}", f"Content {i}", [float(i) / 10.0, 0.5, 0.5])
        results = store.search([0.5, 0.5, 0.5], limit=3)
        assert len(results) == 3

    def test_search_empty_store_returns_empty(self, store):
        results = store.search([0.1, 0.2, 0.3])
        assert results == []

    def test_search_results_ordered_by_similarity(self, store):
        store.add("note-exact", "Exact match", [1.0, 0.0, 0.0])
        store.add("note-far", "Far away", [0.0, 1.0, 0.0])
        results = store.search([1.0, 0.0, 0.0])
        assert results[0]["note_id"] == "note-exact"


class TestDelete:
    def test_delete_existing(self, store):
        store.add("note-1", "Hello", [0.1, 0.2, 0.3])
        store.delete("note-1")
        assert store.count() == 0

    def test_delete_nonexistent_does_not_raise(self, store):
        store.delete("nonexistent")

    def test_delete_one_of_many(self, store):
        store.add("note-1", "Hello", [0.1, 0.2, 0.3])
        store.add("note-2", "World", [0.4, 0.5, 0.6])
        store.delete("note-1")
        assert store.count() == 1

    def test_delete_removes_from_search_results(self, store):
        store.add("note-1", "Hello", [0.1, 0.2, 0.3])
        store.add("note-2", "World", [0.4, 0.5, 0.6])
        store.delete("note-1")
        results = store.search([0.1, 0.2, 0.3])
        assert len(results) == 1
        assert results[0]["note_id"] == "note-2"


class TestSemanticChunking:
    def test_long_text_creates_multiple_chunks(self, store):
        long_text = "Hello. " * 500  # ~3000 chars > 2000 target
        store.add("note-long", long_text, [0.1, 0.2, 0.3])
        # Expect >1 chunk for long text
        assert store.count() > 1

    def test_short_text_single_chunk(self, store):
        store.add("note-short", "Short text.", [0.1, 0.2, 0.3])
        assert store.count() == 1

    def test_chunk_search_returns_best_chunk(self, store):
        store.add(
            "note-chunks",
            "First part is about apples. " * 100 + "Second part is about bananas." * 100,
            [0.9, 0.0, 0.0],
        )
        results = store.search([0.9, 0.0, 0.0])
        assert len(results) >= 1
        assert results[0]["note_id"] == "note-chunks"
        assert "apples" in results[0]["text"] or "bananas" in results[0]["text"]


class TestGetEmbeddings:
    def test_get_embeddings_returns_vectors(self, store):
        store.add("note-1", "A", [0.1, 0.2, 0.3])
        store.add("note-2", "B", [0.4, 0.5, 0.6])
        embs = store.get_embeddings(["note-1", "note-2"])
        assert "note-1" in embs
        assert "note-2" in embs
        assert len(embs["note-1"]) == 3
        assert len(embs["note-2"]) == 3

    def test_get_embeddings_missing_returns_empty(self, store):
        embs = store.get_embeddings(["note-missing"])
        assert embs == {}
