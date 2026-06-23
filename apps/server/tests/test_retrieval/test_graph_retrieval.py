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


class TestQueryTokenization:
    """Tests for _tokenize_query helper and natural-language query handling."""

    def test_tokenize_query_filters_stop_words(self):
        """_tokenize_query removes stop words and returns meaningful tokens."""
        from src.retrieval.graph_retrieval import _tokenize_query

        tokens = _tokenize_query("what is the FEP")
        assert tokens == ["FEP"]

    def test_tokenize_query_handles_punctuation(self):
        """_tokenize_query strips punctuation from tokens."""
        from src.retrieval.graph_retrieval import _tokenize_query

        tokens = _tokenize_query("What is FEP?")
        assert tokens == ["FEP"]

    def test_tokenize_query_preserves_case_for_matching(self):
        """_tokenize_query preserves original case (Kuzu CONTAINS is case-sensitive)."""
        from src.retrieval.graph_retrieval import _tokenize_query

        tokens = _tokenize_query("Tell me about Active interface")
        assert tokens == ["Active", "interface"]

    def test_tokenize_query_deduplicates_case_insensitive(self):
        """Duplicate tokens (case-insensitive) collapse to first occurrence."""
        from src.retrieval.graph_retrieval import _tokenize_query

        tokens = _tokenize_query("FEP fep FEP")
        assert tokens == ["FEP"]

    def test_tokenize_query_empty_and_stop_only(self):
        """Queries with only stop words or empty produce no tokens."""
        from src.retrieval.graph_retrieval import _tokenize_query

        assert _tokenize_query("") == []
        assert _tokenize_query("what is the a an of") == []


class TestNaturalLanguageSearch:
    """Tests that GraphRetriever.search handles natural-language questions."""

    @pytest.fixture
    def graph_with_aliases(self, tmp_db_path):
        """Graph with entities having aliases for natural-language query tests."""
        from src.knowledge.graph import KnowledgeGraph
        from src.models.entity import Entity, EntityType

        kg = KnowledgeGraph(db_path=tmp_db_path)
        kg.create_entity(
            Entity(
                name="Free Energy Principle",
                type=EntityType.Concept,
                aliases=["FEP"],
                confidence=0.9,
                source_note_ids=[],
            )
        )
        kg.create_entity(
            Entity(
                name="Active interface",
                type=EntityType.Concept,
                confidence=0.8,
                source_note_ids=[],
            )
        )
        return kg

    @pytest.fixture
    def retriever_with_aliases(self, graph_with_aliases):
        from src.retrieval.graph_retrieval import GraphRetriever

        return GraphRetriever(graph=graph_with_aliases)

    def test_search_finds_entity_with_natural_language_query(self, retriever_with_aliases):
        """Query 'What is FEP?' finds entity 'Free Energy Principle' via alias 'FEP'."""
        results = retriever_with_aliases.search("What is FEP?")
        entity_results = [r for r in results if r.source_type == "entity"]
        assert len(entity_results) >= 1, (
            f"Expected >= 1 entity result for 'What is FEP?', got: {results}"
        )
        assert "Free Energy Principle" in entity_results[0].snippet

    def test_search_finds_entity_with_multi_word_query(self, retriever_with_aliases):
        """Query 'Tell me about Active interface' finds entity 'Active interface'."""
        results = retriever_with_aliases.search("Tell me about Active interface")
        entity_results = [r for r in results if r.source_type == "entity"]
        assert len(entity_results) >= 1, f"Expected >= 1 entity result, got: {results}"
        names = [r.metadata.get("entity_name", "") for r in entity_results]
        assert "Active interface" in names, f"'Active interface' not in entity names: {names}"

    def test_search_with_full_query_still_works(self, retriever_with_aliases):
        """Existing exact-match behavior is preserved (query 'FEP' still works)."""
        results = retriever_with_aliases.search("FEP")
        entity_results = [r for r in results if r.source_type == "entity"]
        assert len(entity_results) >= 1
        assert "Free Energy Principle" in entity_results[0].snippet

    def test_search_lowercase_query_finds_entity(self, retriever_with_aliases):
        """Lowercase query 'what is fep?' also finds entity via case-variant search."""
        results = retriever_with_aliases.search("what is fep?")
        entity_results = [r for r in results if r.source_type == "entity"]
        assert len(entity_results) >= 1, (
            f"Expected >= 1 entity result for lowercase query, got: {results}"
        )


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


class TestSyntheticEntityResults:
    """Tests for synthetic entity-backed results (orphaned entities)."""

    @pytest.fixture
    def graph_with_orphan(self, tmp_db_path):
        """Graph with an orphaned entity (no source_note_ids) and a note-backed entity."""
        from src.knowledge.graph import KnowledgeGraph
        from src.models.entity import Entity, EntityType

        kg = KnowledgeGraph(db_path=tmp_db_path)
        # Orphaned entity — no source_note_ids
        kg.create_entity(
            Entity(
                name="Free Energy Principle",
                type=EntityType.Concept,
                aliases=["FEP", "Active Inference"],
                confidence=0.9,
                source_note_ids=[],
            )
        )
        # Note-backed entity
        kg.create_entity(
            Entity(
                name="Alice",
                type=EntityType.Person,
                confidence=0.95,
                source_note_ids=["note-1"],
            )
        )
        return kg

    @pytest.fixture
    def retriever_with_orphan(self, graph_with_orphan):
        from src.retrieval.graph_retrieval import GraphRetriever

        return GraphRetriever(graph=graph_with_orphan)

    def test_search_finds_orphaned_entity_by_name(self, retriever_with_orphan):
        """Entity with empty source_note_ids produces synthetic result with source_type='entity'."""
        results = retriever_with_orphan.search("Free Energy")
        entity_results = [r for r in results if r.source_type == "entity"]
        assert len(entity_results) >= 1
        assert entity_results[0].note_id.startswith("entity:")

    def test_search_finds_entity_by_alias(self, retriever_with_orphan):
        """Query 'FEP' finds entity 'Free Energy Principle' that has 'FEP' in aliases."""
        results = retriever_with_orphan.search("FEP")
        # The entity should be found via alias matching
        entity_results = [r for r in results if r.source_type == "entity"]
        assert len(entity_results) >= 1
        assert "Free Energy Principle" in entity_results[0].snippet

    def test_synthetic_result_has_entity_source_type(self, retriever_with_orphan):
        """Synthetic result has source_type='entity'."""
        results = retriever_with_orphan.search("Free Energy")
        synthetic = [r for r in results if r.note_id.startswith("entity:")]
        assert len(synthetic) == 1
        assert synthetic[0].source_type == "entity"

    def test_synthetic_result_snippet_contains_entity_name(self, retriever_with_orphan):
        """Snippet includes entity name and type."""
        results = retriever_with_orphan.search("Free Energy")
        synthetic = [r for r in results if r.source_type == "entity"]
        assert len(synthetic) == 1
        snippet = synthetic[0].snippet
        assert "Free Energy Principle" in snippet
        assert "Concept" in snippet

    def test_synthetic_result_snippet_contains_relations(self, tmp_db_path):
        """Snippet includes relation info when entity has relations."""
        from src.knowledge.graph import KnowledgeGraph
        from src.models.entity import Entity, EntityType
        from src.models.relation import Relation, RelationType
        from src.retrieval.graph_retrieval import GraphRetriever

        kg = KnowledgeGraph(db_path=tmp_db_path)
        orphan = Entity(
            name="Active Inference",
            type=EntityType.Concept,
            confidence=0.85,
            source_note_ids=[],
        )
        related = Entity(
            name="Karl Friston",
            type=EntityType.Person,
            confidence=0.9,
            source_note_ids=[],
        )
        kg.create_entity(orphan)
        kg.create_entity(related)
        kg.create_relation(
            Relation(
                source_entity_id=related.id,
                target_entity_id=orphan.id,
                type=RelationType.created,
            )
        )

        retriever = GraphRetriever(graph=kg)
        results = retriever.search("Active Inference")
        synthetic = [r for r in results if r.source_type == "entity"]
        assert len(synthetic) == 1
        assert "Relations:" in synthetic[0].snippet
        assert "Karl Friston" in synthetic[0].snippet
        assert "created" in synthetic[0].snippet

    def test_mixed_provenance_entity_produces_both_note_and_entity_results(
        self, retriever_with_orphan
    ):
        """Two entities — one with notes, one without — both returned by search."""
        # Search with query "i" — matches "Alice" (name contains 'i') and
        # "Free Energy Principle" (alias "Active Inference" contains 'i').
        results = retriever_with_orphan.search("i")
        note_results = [r for r in results if r.source_type == "note"]
        entity_results = [r for r in results if r.source_type == "entity"]
        # Both types should be present
        assert len(note_results) >= 1
        assert len(entity_results) >= 1

    def test_entity_with_notes_and_conversation_turns_produces_note_results(self, tmp_db_path):
        """Entity with BOTH source_note_ids and conversation-turn provenance
        should produce note results (not synthetic entity results).

        The ``source_note_ids`` check takes precedence: a non-empty list means
        the entity is note-backed, even if it also has ``source_conversation_turns``
        in its properties.
        """
        from src.knowledge.graph import KnowledgeGraph
        from src.models.entity import Entity, EntityType
        from src.retrieval.graph_retrieval import GraphRetriever

        kg = KnowledgeGraph(db_path=tmp_db_path)
        entity = Entity(
            name="Dual Provenance Entity",
            type=EntityType.Concept,
            confidence=0.9,
            source_note_ids=["note-1", "note-2"],
            properties={
                "source_conversation_turns": ["turn-1", "turn-2"],
                "description": "Has both notes and conversation turns",
            },
        )
        kg.create_entity(entity)

        retriever = GraphRetriever(graph=kg)
        results = retriever.search("Dual Provenance")
        # Should produce note results, NOT synthetic entity results
        note_results = [r for r in results if r.source_type == "note"]
        entity_results = [r for r in results if r.source_type == "entity"]
        assert len(note_results) == 2
        assert len(entity_results) == 0
        note_ids = {r.note_id for r in note_results}
        assert note_ids == {"note-1", "note-2"}

    def test_snippet_truncation_for_many_relations(self, tmp_db_path):
        """Entity with >10 relations has truncated snippet (max 10 shown)."""
        from src.knowledge.graph import KnowledgeGraph
        from src.models.entity import Entity, EntityType
        from src.models.relation import Relation, RelationType
        from src.retrieval.graph_retrieval import GraphRetriever

        kg = KnowledgeGraph(db_path=tmp_db_path)
        main = Entity(
            name="Hub",
            type=EntityType.Concept,
            confidence=0.9,
            source_note_ids=[],
        )
        kg.create_entity(main)
        # Create 15 related entities
        for i in range(15):
            other = Entity(
                name=f"Related{i}",
                type=EntityType.Concept,
                confidence=0.5,
                source_note_ids=[],
            )
            kg.create_entity(other)
            kg.create_relation(
                Relation(
                    source_entity_id=main.id,
                    target_entity_id=other.id,
                    type=RelationType.related_to,
                )
            )

        retriever = GraphRetriever(graph=kg)
        results = retriever.search("Hub")
        synthetic = [r for r in results if r.source_type == "entity"]
        assert len(synthetic) == 1
        snippet = synthetic[0].snippet
        # Count relation lines (lines starting with "  - ")
        rel_lines = [line for line in snippet.split("\n") if line.startswith("  - ")]
        assert len(rel_lines) == 10

    def test_snippet_excludes_internal_properties(self, tmp_db_path):
        """source_conversation_turns not in snippet."""
        from src.knowledge.graph import KnowledgeGraph
        from src.models.entity import Entity, EntityType
        from src.retrieval.graph_retrieval import GraphRetriever

        kg = KnowledgeGraph(db_path=tmp_db_path)
        entity = Entity(
            name="Test Entity",
            type=EntityType.Concept,
            confidence=0.8,
            source_note_ids=[],
            properties={
                "source_conversation_turns": [1, 2, 3],
                "source_memory_ids": ["mem-1"],
                "description": "A test entity",
            },
        )
        kg.create_entity(entity)

        retriever = GraphRetriever(graph=kg)
        results = retriever.search("Test Entity")
        synthetic = [r for r in results if r.source_type == "entity"]
        assert len(synthetic) == 1
        snippet = synthetic[0].snippet
        assert "source_conversation_turns" not in snippet
        assert "source_memory_ids" not in snippet
        assert "description" in snippet
