"""Integration tests for Retrieval pipeline.

Uses real ChromaDB, real file-based notes, and real Kuzu KG in temp dirs.
LLMProvider is mocked for dense retrieval embedding generation.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.graph import KnowledgeGraph
from src.models.entity import Entity, EntityType
from src.models.note import Note
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import HybridFusion
from src.retrieval.graph_retrieval import GraphRetriever
from src.retrieval.sparse import SparseRetriever
from src.storage.note_store import NoteStore


def _make_embedding(dim: int = 384, seed: int = 0) -> list[float]:
    """Create a deterministic pseudo-embedding vector."""
    import random
    rng = random.Random(seed)
    vec = [rng.uniform(-1, 1) for _ in range(dim)]
    magnitude = sum(x * x for x in vec) ** 0.5
    return [x / magnitude for x in vec]


@pytest.fixture
def temp_dirs():
    """Create temporary directories for all stores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        yield {
            "notes": str(base / "notes"),
            "chroma": str(base / "chroma"),
            "kuzu": str(base / "kuzu.db"),
        }


@pytest.fixture
def note_store(temp_dirs):
    return NoteStore(notes_dir=temp_dirs["notes"])


@pytest.fixture
def embedding_store(temp_dirs):
    return EmbeddingStore(db_path=temp_dirs["chroma"])


@pytest.fixture
def knowledge_graph(temp_dirs):
    return KnowledgeGraph(db_path=temp_dirs["kuzu"])


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock()
    provider.embed.return_value = _make_embedding(seed=42)
    return provider


@pytest.fixture
def populated_note_store(note_store):
    """Create multiple notes in the store."""
    notes = [
        Note(
            id="n-001",
            title="Python Programming",
            content="Python is a high-level programming language. It supports multiple paradigms.",
            frontmatter={"tags": ["python", "coding"]},
            path="/notes/python.md",
        ),
        Note(
            id="n-002",
            title="Rust Systems",
            content="Rust is a systems programming language focused on memory safety and concurrency.",
            frontmatter={"tags": ["rust", "coding"]},
            path="/notes/rust.md",
        ),
        Note(
            id="n-003",
            title="Machine Learning",
            content="Machine learning uses Python for data science, neural networks, and deep learning.",
            frontmatter={"tags": ["ml", "python"]},
            path="/notes/ml.md",
        ),
    ]
    for note in notes:
        note_store.create(note)
    return note_store, notes


@pytest.fixture
def populated_embedding_store(embedding_store, populated_note_store):
    """Add embeddings for all notes."""
    store, notes = populated_note_store
    for i, note in enumerate(notes):
        embedding = _make_embedding(seed=i + 1)
        embedding_store.add(note_id=note.id, text=note.content, embedding=embedding)
    return embedding_store, notes


@pytest.fixture
def populated_kg(knowledge_graph, populated_note_store):
    """Add entities linked to notes."""
    store, notes = populated_note_store

    entities = [
        Entity(
            id="e-001",
            name="Python",
            type=EntityType.Concept,
            confidence=0.95,
            source_note_ids=["n-001", "n-003"],
        ),
        Entity(
            id="e-002",
            name="Rust",
            type=EntityType.Concept,
            confidence=0.9,
            source_note_ids=["n-002"],
        ),
        Entity(
            id="e-003",
            name="Machine Learning",
            type=EntityType.Concept,
            confidence=0.85,
            source_note_ids=["n-003"],
        ),
    ]
    for entity in entities:
        knowledge_graph.create_entity(entity)

    return knowledge_graph, entities


# --------------------------------------------------------------------------- #
# Dense retrieval
# --------------------------------------------------------------------------- #

class TestDenseRetrieval:
    @pytest.mark.asyncio
    async def test_dense_retrieval(self, populated_embedding_store, mock_llm_provider):
        """Dense retriever should return semantically similar results."""
        emb_store, notes = populated_embedding_store

        dense = DenseRetriever(embedding_store=emb_store, llm_provider=mock_llm_provider)
        results = await dense.search("programming languages", limit=5)

        assert len(results) > 0
        for r in results:
            assert r.source == "dense"

        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @pytest.mark.asyncio
    async def test_dense_retrieval_empty_store(self, temp_dirs, mock_llm_provider):
        """Dense retriever on empty store should return empty list."""
        emb_store = EmbeddingStore(db_path=temp_dirs["chroma"] + "_empty")
        dense = DenseRetriever(embedding_store=emb_store, llm_provider=mock_llm_provider)
        results = await dense.search("anything", limit=5)
        assert results == []


# --------------------------------------------------------------------------- #
# Sparse retrieval
# --------------------------------------------------------------------------- #

class TestSparseRetrieval:
    def test_sparse_retrieval(self, temp_dirs, populated_note_store):
        """Sparse retriever should find keyword matches."""
        store, notes = populated_note_store
        sparse = SparseRetriever(notes_dir=temp_dirs["notes"])

        results = sparse.search("Python", limit=10)
        assert len(results) >= 2  # Python Programming + Machine Learning

        for r in results:
            assert r.source == "sparse"
            assert r.score > 0

    def test_sparse_retrieval_keyword_specificity(self, temp_dirs, populated_note_store):
        """Sparse retriever should differentiate between specific keywords."""
        store, notes = populated_note_store
        sparse = SparseRetriever(notes_dir=temp_dirs["notes"])

        results = sparse.search("Rust", limit=10)
        assert len(results) >= 1
        for r in results:
            assert "rust" in r.snippet.lower() or "rust" in r.note_id.lower()

    def test_sparse_retrieval_no_match(self, temp_dirs, populated_note_store):
        """Sparse retriever with no matching query returns empty."""
        store, notes = populated_note_store
        sparse = SparseRetriever(notes_dir=temp_dirs["notes"])

        results = sparse.search("xyznonexistent", limit=10)
        assert results == []


# --------------------------------------------------------------------------- #
# Hybrid fusion
# --------------------------------------------------------------------------- #

class TestHybridFusion:
    @pytest.mark.asyncio
    async def test_hybrid_fusion(self, temp_dirs, populated_note_store, populated_embedding_store, populated_kg, mock_llm_provider):
        """Hybrid fusion should return fused results from all 3 retrievers."""
        store, notes = populated_note_store
        emb_store, _ = populated_embedding_store
        kg, _ = populated_kg

        dense = DenseRetriever(embedding_store=emb_store, llm_provider=mock_llm_provider)
        sparse = SparseRetriever(notes_dir=temp_dirs["notes"])
        graph = GraphRetriever(graph=kg)

        fusion = HybridFusion(dense=dense, sparse=sparse, graph=graph)

        results = await fusion.search("Python programming", limit=10)

        # Should have results from at least 2 sources (sparse + graph at minimum)
        assert len(results) > 0

        # Verify all results have valid note_ids and fused source
        for r in results:
            assert r.note_id is not None
            assert r.score > 0
            assert r.source == "fusion"

    @pytest.mark.asyncio
    async def test_hybrid_fusion_returns_fused_results(self, temp_dirs, populated_note_store, populated_embedding_store, populated_kg, mock_llm_provider):
        """Fused results should be deduplicated and sorted by score."""
        store, notes = populated_note_store
        emb_store, _ = populated_embedding_store
        kg, _ = populated_kg

        dense = DenseRetriever(embedding_store=emb_store, llm_provider=mock_llm_provider)
        sparse = SparseRetriever(notes_dir=temp_dirs["notes"])
        graph = GraphRetriever(graph=kg)

        fusion = HybridFusion(dense=dense, sparse=sparse, graph=graph)
        results = await fusion.search("Python", limit=5)

        # All results should have source='fusion'
        for r in results:
            assert r.source == "fusion"

        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

        # No duplicate note_ids
        note_ids = [r.note_id for r in results]
        assert len(note_ids) == len(set(note_ids))


# --------------------------------------------------------------------------- #
# Graph retrieval
# --------------------------------------------------------------------------- #

class TestGraphRetrieval:
    def test_graph_retrieval(self, populated_kg):
        """Graph retriever should find notes via entity names."""
        kg, entities = populated_kg
        retriever = GraphRetriever(graph=kg)

        results = retriever.search("Python", limit=10)
        assert len(results) >= 1

        # Should return notes n-001 and n-003 (both linked to Python entity)
        note_ids = {r.note_id for r in results}
        assert "n-001" in note_ids or "n-003" in note_ids

        for r in results:
            assert r.source == "graph"
            assert r.score > 0

    def test_graph_retrieval_unique_note_ids(self, populated_kg):
        """Graph retriever should not return duplicate note_ids."""
        kg, entities = populated_kg
        retriever = GraphRetriever(graph=kg)

        # Search for something that matches entity with multiple source_note_ids
        results = retriever.search("Python", limit=10)
        note_ids = [r.note_id for r in results]
        assert len(note_ids) == len(set(note_ids))

    def test_graph_retrieval_no_match(self, populated_kg):
        """Graph retriever with no matching entity returns empty."""
        kg, entities = populated_kg
        retriever = GraphRetriever(graph=kg)

        results = retriever.search("zzzznonexistent", limit=10)
        assert results == []
