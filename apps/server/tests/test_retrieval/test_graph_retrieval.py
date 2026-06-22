"""Tests for GraphRetriever class using Kuzu embedded database."""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture
def tmp_db_path():
    """Provide a temporary file path for Kuzu DB, cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def graph(tmp_db_path):
    """Provide a fresh KnowledgeGraph instance with sample entities."""
    from src.knowledge.graph import KnowledgeGraph
    from src.models.entity import Entity, EntityType

    kg = KnowledgeGraph(db_path=tmp_db_path)

    # Insert sample entities with source_note_ids
    entities = [
        Entity(
            name="Alice",
            type=EntityType.Person,
            confidence=0.95,
            source_note_ids=["note-1", "note-2"],
        ),
        Entity(
            name="Bob",
            type=EntityType.Person,
            confidence=0.88,
            source_note_ids=["note-3"],
        ),
        Entity(
            name="Acme Corp",
            type=EntityType.Organization,
            confidence=0.90,
            source_note_ids=["note-1", "note-4"],
        ),
        Entity(
            name="Project X",
            type=EntityType.Project,
            confidence=0.75,
            source_note_ids=["note-5"],
        ),
        Entity(
            name="New York City",
            type=EntityType.Place,
            confidence=0.80,
            source_note_ids=["note-2", "note-6"],
        ),
    ]
    for e in entities:
        kg.create_entity(e)
    return kg


@pytest.fixture
def retriever(graph):
    """Provide a GraphRetriever instance."""
    from src.retrieval.graph_retrieval import GraphRetriever

    return GraphRetriever(graph=graph)


class TestGraphRetrieverSearch:
    """Test GraphRetriever.search method."""

    def test_search_finds_entities_matching_query(self, retriever):
        """Search should return results for entities whose name contains the query."""
        results = retriever.search("Alice")
        assert len(results) >= 1
        note_ids = [r.note_id for r in results]
        assert "note-1" in note_ids or "note-2" in note_ids

    def test_search_returns_results_with_source_graph(self, retriever):
        """All results should have source='graph'."""
        results = retriever.search("Alice")
        assert len(results) >= 1
        for r in results:
            assert r.source == "graph"

    def test_search_respects_limit(self, graph):
        """Search should not return more than `limit` results."""
        from src.retrieval.graph_retrieval import GraphRetriever

        retriever = GraphRetriever(graph=graph)
        results = retriever.search("a", limit=2)
        assert len(results) <= 2

    def test_search_no_match_returns_empty(self, retriever):
        """Search with no matching query should return empty list."""
        results = retriever.search("zzzznonexistent")
        assert results == []

    def test_search_unique_note_ids(self, retriever):
        """Multiple entities referencing the same note should not duplicate results."""
        results = retriever.search("note-1")
        note_ids = [r.note_id for r in results]
        # Each note_id should be unique in results
        assert len(note_ids) == len(set(note_ids))


class TestGraphRetrieverEntityToResults:
    """Test GraphRetriever._entity_to_results helper."""

    def test_entity_to_results_produces_search_results(self, retriever):
        """_entity_to_results should convert entities to SearchResult objects."""
        from src.models.entity import Entity, EntityType

        entities = [
            Entity(
                name="Test",
                type=EntityType.Concept,
                source_note_ids=["note-a", "note-b"],
            ),
        ]
        results = retriever._entity_to_results(entities)
        assert len(results) == 2
        note_ids = {r.note_id for r in results}
        assert note_ids == {"note-a", "note-b"}

    def test_entity_to_results_sets_source_graph(self, retriever):
        """Results from _entity_to_results should have source='graph'."""
        from src.models.entity import Entity, EntityType

        entities = [Entity(name="X", type=EntityType.Concept, source_note_ids=["n1"])]
        results = retriever._entity_to_results(entities)
        assert all(r.source == "graph" for r in results)

    def test_entity_to_results_empty_list(self, retriever):
        """_entity_to_results with empty list returns empty list."""
        results = retriever._entity_to_results([])
        assert results == []
