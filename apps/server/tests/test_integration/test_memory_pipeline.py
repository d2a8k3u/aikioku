"""Integration tests for Memory Pipeline.

Uses real Kuzu DB in tempdir, real EventBus, and real MemoryExtractor +
MemoryConsolidator. LLMProvider is mocked.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events import EventBus
from src.knowledge.graph import KnowledgeGraph
from src.llm.base import LLMProvider
from src.memory.consolidation import MemoryConsolidator
from src.memory.extraction import MemoryExtractor
from src.models.memory import Memory, MemoryTier
from src.models.note import Note


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_dirs():
    """Create temporary directories for Kuzu DB and EventBus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        yield {
            "kuzu": str(base / "kuzu.db"),
            "events": str(base / "events.db"),
        }


@pytest.fixture
def knowledge_graph(tmp_dirs):
    return KnowledgeGraph(db_path=tmp_dirs["kuzu"])


@pytest.fixture
def event_bus(tmp_dirs):
    return EventBus(db_path=tmp_dirs["events"])


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock(spec=LLMProvider)
    provider.complete = AsyncMock(return_value="[]")
    provider.embed = AsyncMock(return_value=[0.0] * 384)
    provider.is_available = MagicMock(return_value=True)
    return provider


@pytest.fixture
def sample_note():
    return Note(
        id="note-001",
        title="Python Facts",
        content=(
            "Python is a high-level programming language. "
            "Python was created by Guido van Rossum. "
            "Python supports multiple paradigms including OOP and functional programming."
        ),
        path="/notes/python-facts.md",
    )


@pytest.fixture
def extractor(mock_llm_provider, event_bus):
    return MemoryExtractor(llm_provider=mock_llm_provider, event_bus=event_bus)


@pytest.fixture
def consolidator(knowledge_graph, event_bus):
    return MemoryConsolidator(graph=knowledge_graph, event_bus=event_bus)


def _make_memory(
    subject: str = "Alice",
    predicate: str = "knows",
    object: str = "Bob",
    confidence: float = 0.9,
    source: str = "test-source",
    created: datetime | None = None,
    vitality_score: float = 0.8,
    tier: MemoryTier = MemoryTier.hot,
) -> Memory:
    if created is None:
        created = datetime.now(timezone.utc)
    return Memory(
        subject=subject,
        predicate=predicate,
        object=object,
        confidence=confidence,
        source=source,
        created=created,
        modified=created,
        vitality_score=vitality_score,
        tier=tier,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestExtractMemoriesFromNote:
    """Create note with factual content, run MemoryExtractor, verify memories extracted."""

    @pytest.mark.asyncio
    async def test_extract_memories_from_note(self, extractor, mock_llm_provider, sample_note):
        mock_response = json.dumps(
            [
                {
                    "subject": "Python",
                    "predicate": "is_a",
                    "object": "high-level programming language",
                    "confidence": 0.95,
                },
                {
                    "subject": "Python",
                    "predicate": "created_by",
                    "object": "Guido van Rossum",
                    "confidence": 0.90,
                },
                {
                    "subject": "Python",
                    "predicate": "supports",
                    "object": "OOP",
                    "confidence": 0.85,
                },
            ]
        )
        mock_llm_provider.complete.return_value = mock_response

        memories = await extractor.extract_from_note(sample_note)

        assert len(memories) == 3
        assert all(isinstance(m, Memory) for m in memories)

        assert memories[0].subject == "Python"
        assert memories[0].predicate == "is_a"
        assert memories[0].object == "high-level programming language"
        assert memories[0].confidence == 0.95
        assert memories[0].source == sample_note.id

        assert memories[1].subject == "Python"
        assert memories[1].predicate == "created_by"
        assert memories[1].object == "Guido van Rossum"
        assert memories[1].confidence == 0.90
        assert memories[1].source == sample_note.id

        assert memories[2].subject == "Python"
        assert memories[2].predicate == "supports"
        assert memories[2].object == "OOP"
        assert memories[2].confidence == 0.85
        assert memories[2].source == sample_note.id

    @pytest.mark.asyncio
    async def test_extract_sets_source_to_note_id(self, extractor, mock_llm_provider, sample_note):
        mock_llm_provider.complete.return_value = json.dumps(
            [
                {"subject": "A", "predicate": "rel", "object": "B", "confidence": 0.5},
            ]
        )

        memories = await extractor.extract_from_note(sample_note)

        assert len(memories) == 1
        assert memories[0].source == "note-001"

    @pytest.mark.asyncio
    async def test_extract_empty_response(self, extractor, mock_llm_provider, sample_note):
        mock_llm_provider.complete.return_value = "[]"

        memories = await extractor.extract_from_note(sample_note)

        assert memories == []


class TestConsolidationDedup:
    """Create duplicate memories, run stage_deduplicate, verify deduped."""

    @pytest.mark.asyncio
    async def test_consolidation_dedup(self, consolidator):
        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.9)
        m2 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.7)
        m3 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.5)

        clusters = [[m1, m2, m3]]
        result = await consolidator.stage_deduplicate(clusters)

        assert len(result) == 1
        assert len(result[0]) == 1

    @pytest.mark.asyncio
    async def test_consolidation_keeps_first_duplicate(self, consolidator):
        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.9)
        m2 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.7)

        clusters = [[m1, m2]]
        result = await consolidator.stage_deduplicate(clusters)

        assert len(result[0]) == 1
        assert result[0][0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_no_duplicates_intact(self, consolidator):
        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob")
        m2 = _make_memory(subject="Alice", predicate="likes", object="Bob")

        clusters = [[m1, m2]]
        result = await consolidator.stage_deduplicate(clusters)

        assert len(result[0]) == 2

    @pytest.mark.asyncio
    async def test_multiple_clusters(self, consolidator):
        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.9)
        m2 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.5)
        m3 = _make_memory(subject="Charlie", predicate="likes", object="Python", confidence=0.8)
        m4 = _make_memory(subject="Charlie", predicate="likes", object="Python", confidence=0.6)

        clusters = [[m1, m2], [m3, m4]]
        result = await consolidator.stage_deduplicate(clusters)

        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1


class TestConsolidationMerge:
    """Create complementary memories, run stage_merge, verify merged."""

    @pytest.mark.asyncio
    async def test_consolidation_merge(self, consolidator):
        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.5)
        m2 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.9)
        m3 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.7)

        clusters = [[m1, m2, m3]]
        result = await consolidator.stage_merge(clusters)

        assert len(result) == 1
        assert result[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_merge_keeps_unique_across_clusters(self, consolidator):
        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.6)
        m2 = _make_memory(subject="Charlie", predicate="likes", object="Python", confidence=0.8)

        clusters = [[m1], [m2]]
        result = await consolidator.stage_merge(clusters)

        assert len(result) == 2
        subjects = {m.subject for m in result}
        assert "Alice" in subjects
        assert "Charlie" in subjects

    @pytest.mark.asyncio
    async def test_merge_same_triple_from_different_clusters(self, consolidator):
        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.3)
        m2 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.8)

        clusters = [[m1], [m2]]
        result = await consolidator.stage_merge(clusters)

        assert len(result) == 1
        assert result[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_merge_different_predicates_not_merged(self, consolidator):
        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.9)
        m2 = _make_memory(subject="Alice", predicate="likes", object="Bob", confidence=0.5)

        clusters = [[m1, m2]]
        result = await consolidator.stage_merge(clusters)

        assert len(result) == 2


class TestMemoryTiers:
    """Create memories with different ages/confidence, run stage_forgetting, verify tier assignment."""

    @pytest.mark.asyncio
    async def test_memory_tiers(self, consolidator):
        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=400)
        recent_date = now - timedelta(days=10)

        m_hot = _make_memory(
            subject="Alice",
            predicate="knows",
            object="Bob",
            confidence=0.9,
            created=recent_date,
            vitality_score=0.9,
            tier=MemoryTier.hot,
        )
        m_warm = _make_memory(
            subject="Charlie",
            predicate="likes",
            object="Python",
            confidence=0.6,
            created=recent_date,
            vitality_score=0.3,
            tier=MemoryTier.hot,
        )
        m_cold_old = _make_memory(
            subject="Dave",
            predicate="works_at",
            object="Acme",
            confidence=0.5,
            created=old_date,
            vitality_score=0.5,
            tier=MemoryTier.hot,
        )
        m_cold_low_conf = _make_memory(
            subject="Eve",
            predicate="studied",
            object="Math",
            confidence=0.2,
            created=recent_date,
            vitality_score=0.5,
            tier=MemoryTier.hot,
        )

        result = await consolidator.stage_forgetting([m_hot, m_warm, m_cold_old, m_cold_low_conf])

        assert len(result) == 4

        hot_mem = next(m for m in result if m.subject == "Alice")
        assert hot_mem.tier == MemoryTier.hot

        warm_mem = next(m for m in result if m.subject == "Charlie")
        assert warm_mem.tier == MemoryTier.warm

        cold_old_mem = next(m for m in result if m.subject == "Dave")
        assert cold_old_mem.tier == MemoryTier.cold

        cold_conf_mem = next(m for m in result if m.subject == "Eve")
        assert cold_conf_mem.tier == MemoryTier.cold

    @pytest.mark.asyncio
    async def test_old_memory_becomes_cold(self, consolidator):
        old_date = datetime.now(timezone.utc) - timedelta(days=500)
        m = _make_memory(confidence=0.5, created=old_date, vitality_score=0.5)

        result = await consolidator.stage_forgetting([m])

        assert result[0].tier == MemoryTier.cold

    @pytest.mark.asyncio
    async def test_low_confidence_becomes_cold(self, consolidator):
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)
        m = _make_memory(confidence=0.1, created=recent_date, vitality_score=0.5)

        result = await consolidator.stage_forgetting([m])

        assert result[0].tier == MemoryTier.cold

    @pytest.mark.asyncio
    async def test_high_vitality_stays_hot(self, consolidator):
        recent_date = datetime.now(timezone.utc) - timedelta(days=5)
        m = _make_memory(confidence=0.8, created=recent_date, vitality_score=0.9)

        result = await consolidator.stage_forgetting([m])

        assert result[0].tier == MemoryTier.hot

    @pytest.mark.asyncio
    async def test_boundary_vitality_warm(self, consolidator):
        recent_date = datetime.now(timezone.utc) - timedelta(days=5)
        m = _make_memory(confidence=0.5, created=recent_date, vitality_score=0.3)

        result = await consolidator.stage_forgetting([m])

        assert result[0].tier == MemoryTier.warm


class TestMemoryCreateSearchRoundTrip:
    """create-from-text → semantic search through a real EmbeddingStore."""

    @pytest.mark.asyncio
    async def test_create_then_semantic_search(self, tmp_path, mock_llm_provider):
        from src.api import memory as memory_api
        from src.knowledge.embeddings import EmbeddingStore

        memory_api._extractor = None
        memory_api._ensure_table()

        mock_llm_provider.complete = AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "subject": "Alice",
                        "predicate": "works_at",
                        "object": "Acme",
                        "confidence": 0.9,
                    },
                ]
            )
        )
        # One deterministic vector for every text → exact cosine match on search.
        embed_provider = AsyncMock()
        embed_provider.embed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        store = EmbeddingStore(
            str(tmp_path / "chroma_mem"), collection_name="memories", dimension=8
        )

        created = await memory_api._create_memory_from_text(
            "Alice works at Acme",
            source="user",
            llm=mock_llm_provider,
            provider=embed_provider,
            store=store,
        )
        assert len(created) == 1

        results = await memory_api._search_memories_semantic(
            "where does alice work", 5, embed_provider, store
        )
        assert len(results) == 1
        assert results[0]["subject"] == "Alice"
        assert results[0]["object"] == "Acme"
        assert results[0]["score"] > 0


class TestMemoryGraphBridge:
    """Consolidation reconciles the knowledge graph with memories (real Kuzu)."""

    @pytest.fixture
    def consolidator_with_llm(self, knowledge_graph, event_bus, mock_llm_provider):
        return MemoryConsolidator(
            graph=knowledge_graph, event_bus=event_bus, llm_provider=mock_llm_provider
        )

    async def test_consolidate_syncs_survivor_to_graph(
        self, consolidator_with_llm, knowledge_graph
    ):
        mem = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.9)
        await consolidator_with_llm.run([mem])
        assert knowledge_graph.count_entities() == 2
        assert knowledge_graph.count_relations() == 1
        rel = knowledge_graph.get_all_relations()[0]
        assert rel.properties["predicate"] == "knows"
        assert rel.properties["source_memory_ids"] == [mem.id]

    async def test_consolidate_dedup_trims_relation_provenance(
        self, consolidator_with_llm, knowledge_graph, mock_llm_provider
    ):
        from src.memory.graph_sync import sync_memory_to_graph

        a = _make_memory(subject="Alice", predicate="knows", object="Bob")
        b = _make_memory(subject="Alice", predicate="knows", object="Bob")  # dup, different id
        # Simulate both already synced at creation time.
        await sync_memory_to_graph(a, knowledge_graph, mock_llm_provider)
        await sync_memory_to_graph(b, knowledge_graph, mock_llm_provider)
        rel0 = knowledge_graph.get_all_relations()[0]
        assert set(rel0.properties["source_memory_ids"]) == {a.id, b.id}

        await consolidator_with_llm.run([a, b])
        # Dedup keeps one survivor; the removed dup's provenance is trimmed, not duped.
        assert knowledge_graph.count_relations() == 1
        rel = knowledge_graph.get_all_relations()[0]
        assert rel.properties["source_memory_ids"] == [a.id]

    async def test_cold_memory_removes_relation_keeps_shared_entity(
        self, consolidator_with_llm, knowledge_graph, mock_llm_provider
    ):
        from src.memory.graph_sync import sync_memory_to_graph
        from src.models.entity import Entity, EntityType

        knowledge_graph.create_entity(
            Entity(name="Alice", type=EntityType.Person, source_note_ids=["note-1"])
        )
        # Low confidence -> stage_forgetting marks it cold.
        mem = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.2)
        await sync_memory_to_graph(mem, knowledge_graph, mock_llm_provider)
        assert knowledge_graph.count_relations() == 1

        await consolidator_with_llm.run([mem])
        assert knowledge_graph.count_relations() == 0
        names = {e.name for e in knowledge_graph.find_entities(limit=50)}
        assert "Alice" in names  # kept: still sourced by a note
        assert "Bob" not in names  # deleted: memory-only orphan

    async def test_no_llm_skips_entity_sync(self, consolidator, knowledge_graph):
        # The `consolidator` fixture has no llm_provider -> typing impossible -> skip.
        await consolidator.run([_make_memory()])
        assert knowledge_graph.count_entities() == 0
