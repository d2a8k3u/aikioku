"""Tests for memory -> knowledge-graph sync (src/memory/graph_sync.py)."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.knowledge.graph import KnowledgeGraph
from src.llm.base import LLMProvider
from src.memory.graph_sync import (
    map_predicate,
    remove_memory_from_graph,
    sync_memory_to_graph,
    type_names,
)
from src.models.entity import Entity, EntityType
from src.models.memory import Memory
from src.models.relation import RelationType

_TYPING = '[{"name": "Alice", "type": "Person"}, {"name": "Acme Corp", "type": "Organization"}]'


@pytest.fixture
def graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield KnowledgeGraph(db_path=os.path.join(tmpdir, "kg.db"))


@pytest.fixture
def mock_llm():
    provider = AsyncMock(spec=LLMProvider)
    provider.complete = AsyncMock(return_value=_TYPING)
    provider.is_available = MagicMock(return_value=True)
    return provider


def _memory(subject="Alice", predicate="works at", object="Acme Corp", confidence=0.9):
    return Memory(
        subject=subject,
        predicate=predicate,
        object=object,
        confidence=confidence,
        source="user",
    )


class TestMapPredicate:
    def test_exact_enum_match(self):
        assert map_predicate("created") == RelationType.created
        assert map_predicate("related_to") == RelationType.related_to

    def test_keyword_synonyms(self):
        assert map_predicate("works at") == RelationType.works_at
        assert map_predicate("founded") == RelationType.created
        assert map_predicate("born in") == RelationType.located_in
        assert map_predicate("part of") == RelationType.part_of

    def test_unknown_falls_back_to_related_to(self):
        assert map_predicate("xyzzy") == RelationType.related_to
        assert map_predicate("") == RelationType.related_to
        assert map_predicate("   ") == RelationType.related_to


class TestTypeNames:
    async def test_one_llm_call_returns_types(self, mock_llm):
        result = await type_names(["Alice", "Acme Corp"], mock_llm)
        assert result == {"Alice": EntityType.Person, "Acme Corp": EntityType.Organization}
        assert mock_llm.complete.await_count == 1

    async def test_missing_name_absent_from_result(self, mock_llm):
        result = await type_names(["Alice", "Bob"], mock_llm)
        assert result.get("Alice") == EntityType.Person
        assert "Bob" not in result  # caller defaults missing names to Concept

    async def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.complete = AsyncMock(side_effect=RuntimeError("boom"))
        assert await type_names(["Alice"], mock_llm) == {}


class TestSyncMemoryToGraph:
    async def test_creates_entities_and_relation(self, graph, mock_llm):
        mem = _memory()
        result = await sync_memory_to_graph(mem, graph, mock_llm)
        assert result is not None
        subj, obj, rel = result
        assert subj.name == "Alice" and subj.type == EntityType.Person
        assert obj.name == "Acme Corp" and obj.type == EntityType.Organization
        assert graph.count_entities() == 2
        assert graph.count_relations() == 1
        assert rel.type == RelationType.works_at
        assert rel.properties["predicate"] == "works at"
        assert rel.properties["source_memory_ids"] == [mem.id]
        assert rel.properties["derived_from"] == "memory"
        assert mem.id in subj.properties["source_memory_ids"]
        assert mem.id in obj.properties["source_memory_ids"]

    async def test_idempotent(self, graph, mock_llm):
        mem = _memory()
        await sync_memory_to_graph(mem, graph, mock_llm)
        await sync_memory_to_graph(mem, graph, mock_llm)
        assert graph.count_entities() == 2
        assert graph.count_relations() == 1
        rel = graph.get_all_relations()[0]
        assert rel.properties["source_memory_ids"] == [mem.id]  # deduped

    async def test_blank_triple_returns_none(self, graph, mock_llm):
        assert await sync_memory_to_graph(_memory(subject="  "), graph, mock_llm) is None

    async def test_reuses_existing_note_entity(self, graph, mock_llm):
        existing = Entity(
            name="Alice",
            type=EntityType.Person,
            confidence=0.8,
            source_note_ids=["note-1"],
        )
        graph.create_entity(existing)
        mem = _memory()
        subj, _obj, _rel = await sync_memory_to_graph(mem, graph, mock_llm)
        assert subj.id == existing.id  # resolved to the note entity, not duplicated
        assert graph.count_entities() == 2  # existing Alice + new Acme Corp
        reloaded = graph.get_entity(existing.id)
        assert reloaded.source_note_ids == ["note-1"]  # note provenance intact
        assert mem.id in reloaded.properties["source_memory_ids"]


class TestRemoveMemoryFromGraph:
    async def test_removes_relation_and_orphan_entities(self, graph, mock_llm):
        mem = _memory()
        await sync_memory_to_graph(mem, graph, mock_llm)
        assert graph.count_entities() == 2 and graph.count_relations() == 1
        remove_memory_from_graph(mem, graph)
        assert graph.count_relations() == 0
        assert graph.count_entities() == 0  # both entities sourced only by this memory

    async def test_keeps_shared_entity(self, graph, mock_llm):
        # Alice is also a note entity; Acme Corp is memory-only.
        graph.create_entity(
            Entity(name="Alice", type=EntityType.Person, source_note_ids=["note-1"])
        )
        mem = _memory()
        await sync_memory_to_graph(mem, graph, mock_llm)
        remove_memory_from_graph(mem, graph)
        assert graph.count_relations() == 0
        # Alice kept (has a note id), Acme Corp deleted (memory-only).
        names = {e.name for e in graph.find_entities(limit=50)}
        assert names == {"Alice"}

    async def test_keeps_relation_with_other_memory_source(self, graph, mock_llm):
        mem_a = _memory()
        mem_b = _memory()  # same triple, different id
        await sync_memory_to_graph(mem_a, graph, mock_llm)
        await sync_memory_to_graph(mem_b, graph, mock_llm)
        assert graph.count_relations() == 1
        remove_memory_from_graph(mem_a, graph)
        # Relation still sourced by mem_b → kept (with trimmed provenance).
        assert graph.count_relations() == 1
        rel = graph.get_all_relations()[0]
        assert rel.properties["source_memory_ids"] == [mem_b.id]
