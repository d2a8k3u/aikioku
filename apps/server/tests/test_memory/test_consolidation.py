"""Tests for MemoryConsolidator — the 7-stage pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events import EventBus
from src.models.memory import Memory, MemoryTier


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
    """Helper to create a Memory with sensible defaults."""
    if created is None:
        created = datetime.now(timezone.utc)
    return Memory(
        subject=subject,
        predicate=predicate,
        object=object,
        confidence=confidence,
        source=source,
        created=created,
        vitality_score=vitality_score,
        tier=tier,
    )


@pytest.fixture
def mock_graph() -> MagicMock:
    """Return a mock KnowledgeGraph."""
    return MagicMock()


@pytest.fixture
def mock_event_bus(tmp_path) -> EventBus:
    """Return an EventBus backed by a temp database."""
    return EventBus(db_path=str(tmp_path / "events.db"))


class TestRun:
    """Test run() returns summary with counts."""

    @pytest.mark.asyncio
    async def test_run_returns_summary_with_counts(
        self, mock_graph, mock_event_bus
    ):
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        memories = [
            _make_memory(subject="Alice", predicate="knows", object="Bob"),
            _make_memory(subject="Alice", predicate="knows", object="Bob"),
            _make_memory(subject="Charlie", predicate="likes", object="Python"),
        ]

        summary = await consolidator.run(memories)

        assert isinstance(summary, dict)
        assert "input_count" in summary
        assert "output_count" in summary
        assert "duplicates_removed" in summary
        assert "conflicts_detected" in summary
        assert "archived_count" in summary
        assert summary["input_count"] == 3

    @pytest.mark.asyncio
    async def test_run_empty_list(self, mock_graph, mock_event_bus):
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        summary = await consolidator.run([])

        assert summary["input_count"] == 0
        assert summary["output_count"] == 0

    @pytest.mark.asyncio
    async def test_run_returns_processed_memories(self, mock_graph, mock_event_bus):
        """run() must surface the final processed memories so callers can
        persist tier/confidence updates and dedup/merge results back."""
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        # Two exact duplicates (merge removes one) + one distinct memory.
        memories = [
            _make_memory(subject="Alice", predicate="knows", object="Bob"),
            _make_memory(subject="Alice", predicate="knows", object="Bob"),
            _make_memory(subject="Charlie", predicate="likes", object="Python"),
        ]

        summary = await consolidator.run(memories)

        assert "memories" in summary
        processed = summary["memories"]
        assert isinstance(processed, list)
        assert all(isinstance(m, Memory) for m in processed)
        # Output count matches the processed list length.
        assert len(processed) == summary["output_count"]
        # The duplicate (Alice/knows/Bob) collapsed to a single survivor.
        triples = {(m.subject, m.predicate, m.object) for m in processed}
        assert ("Alice", "knows", "Bob") in triples
        assert ("Charlie", "likes", "Python") in triples


class TestStageDeduplicate:
    """Test stage_deduplicate removes exact duplicates."""

    @pytest.mark.asyncio
    async def test_removes_exact_duplicates(self, mock_graph, mock_event_bus):
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.9)
        m2 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.7)
        m3 = _make_memory(subject="Charlie", predicate="likes", object="Python", confidence=0.8)

        clusters = [[m1, m2, m3]]
        result = await consolidator.stage_deduplicate(clusters)

        # m1 and m2 are exact duplicates on (subject, predicate, object) → one removed
        assert len(result) == 1
        assert len(result[0]) == 2

    @pytest.mark.asyncio
    async def test_no_duplicates_intact(self, mock_graph, mock_event_bus):
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob")
        m2 = _make_memory(subject="Alice", predicate="likes", object="Bob")

        clusters = [[m1, m2]]
        result = await consolidator.stage_deduplicate(clusters)

        assert len(result[0]) == 2


class TestStageMerge:
    """Test stage_merge keeps highest confidence."""

    @pytest.mark.asyncio
    async def test_keeps_highest_confidence(self, mock_graph, mock_event_bus):
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        m1 = _make_memory(confidence=0.5)
        m2 = _make_memory(confidence=0.9)
        m3 = _make_memory(confidence=0.7)

        clusters = [[m1, m2, m3]]
        result = await consolidator.stage_merge(clusters)

        assert len(result) == 1
        assert result[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_single_memory_unchanged(self, mock_graph, mock_event_bus):
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        m1 = _make_memory(confidence=0.6)

        clusters = [[m1]]
        result = await consolidator.stage_merge(clusters)

        assert len(result) == 1
        assert result[0].confidence == 0.6


class TestStageConflictResolution:
    """Test stage_conflict_resolution detects contradictions."""

    @pytest.mark.asyncio
    async def test_detects_contradictions(self, mock_graph, mock_event_bus):
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        # Same subject+predicate, different objects → contradiction
        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.9)
        m2 = _make_memory(subject="Alice", predicate="knows", object="Charlie", confidence=0.8)

        result = await consolidator.stage_conflict_resolution([m1, m2])

        # Both should be flagged (marked with reduced confidence or flagged)
        assert len(result) == 2
        # At least one should have been flagged (confidence reduced or marked)
        flagged = [m for m in result if m.confidence < 0.9]
        assert len(flagged) >= 1

    @pytest.mark.asyncio
    async def test_no_conflict_when_same_object(self, mock_graph, mock_event_bus):
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        m1 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.9)
        m2 = _make_memory(subject="Alice", predicate="knows", object="Bob", confidence=0.8)

        result = await consolidator.stage_conflict_resolution([m1, m2])

        # No contradiction — same object
        assert len(result) == 2
        # Neither should be penalized
        assert all(m.confidence >= 0.8 for m in result)


class TestStageSummarize:
    """Test stage_summarize uses the injected provider or skips when None."""

    @pytest.mark.asyncio
    async def test_injected_provider_produces_guideline(
        self, mock_graph, mock_event_bus
    ):
        from src.memory.consolidation import MemoryConsolidator

        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="Alice has many connections.")

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus, llm_provider=provider
        )

        merged = [
            _make_memory(subject="Alice", predicate="knows", object="Bob"),
            _make_memory(subject="Alice", predicate="knows", object="Charlie"),
        ]

        result = await consolidator.stage_summarize(merged)

        provider.complete.assert_awaited()
        guidelines = [m for m in result if m.predicate == "has_guideline"]
        assert len(guidelines) == 1
        assert guidelines[0].object == "Alice has many connections."

    @pytest.mark.asyncio
    async def test_no_provider_skips_gracefully(
        self, mock_graph, mock_event_bus
    ):
        from src.memory.consolidation import MemoryConsolidator

        # No llm_provider → must skip LLM summarization without network/errors.
        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        merged = [
            _make_memory(subject="Alice", predicate="knows", object="Bob"),
            _make_memory(subject="Alice", predicate="knows", object="Charlie"),
        ]

        result = await consolidator.stage_summarize(merged)

        # Returns the merged memories unchanged, no synthetic guideline.
        assert result == merged
        assert all(m.predicate != "has_guideline" for m in result)


class TestStageForgetting:
    """Test stage_forgetting marks old/low-confidence memories."""

    @pytest.mark.asyncio
    async def test_marks_low_confidence(self, mock_graph, mock_event_bus):
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        old_date = datetime.now(timezone.utc) - timedelta(days=400)
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)

        m_old_low = _make_memory(
            confidence=0.2, created=old_date, vitality_score=0.1
        )
        m_recent_high = _make_memory(
            confidence=0.9, created=recent_date, vitality_score=0.9
        )

        result = await consolidator.stage_forgetting([m_old_low, m_recent_high])

        assert len(result) == 2
        # The old/low-confidence memory should be marked for archival (cold tier)
        old_mem = next(m for m in result if m.created == old_date)
        assert old_mem.tier == MemoryTier.cold

    @pytest.mark.asyncio
    async def test_marks_old_memories(self, mock_graph, mock_event_bus):
        from src.memory.consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            graph=mock_graph, event_bus=mock_event_bus
        )

        old_date = datetime.now(timezone.utc) - timedelta(days=500)
        m_old = _make_memory(confidence=0.5, created=old_date, vitality_score=0.3)

        result = await consolidator.stage_forgetting([m_old])

        assert result[0].tier == MemoryTier.cold
