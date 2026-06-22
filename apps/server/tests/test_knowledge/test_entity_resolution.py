"""Tests for EntityResolver class."""

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
    """Provide a fresh KnowledgeGraph instance."""
    from src.knowledge.graph import KnowledgeGraph

    return KnowledgeGraph(db_path=tmp_db_path)


@pytest.fixture
def sample_entity():
    """Provide a sample Entity."""
    from src.models.entity import Entity, EntityType

    return Entity(
        name="Alice",
        type=EntityType.Person,
        confidence=0.95,
    )


@pytest.fixture
def sample_entity2():
    """Provide a second sample Entity."""
    from src.models.entity import Entity, EntityType

    return Entity(
        name="Acme Corp",
        type=EntityType.Organization,
        confidence=0.88,
    )


class TestEntityResolverInit:
    """Test EntityResolver initialization."""

    def test_init_stores_graph_reference(self, graph):
        from src.knowledge.entity_resolution import EntityResolver

        resolver = EntityResolver(graph=graph)
        assert resolver.graph is graph


class TestFindCandidates:
    """Test find_candidates method."""

    def test_finds_by_name(self, graph, sample_entity):
        """Should find an entity whose name matches."""
        graph.create_entity(sample_entity)
        from src.knowledge.entity_resolution import EntityResolver

        resolver = EntityResolver(graph=graph)
        candidates = resolver.find_candidates(sample_entity)
        assert len(candidates) >= 1
        assert any(c.id == sample_entity.id for c in candidates)

    def test_finds_by_alias(self, graph):
        """Should find an entity that has the search entity as an alias."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        existing = Entity(
            name="Robert",
            type=EntityType.Person,
            aliases=["Bob", "Bobby"],
        )
        graph.create_entity(existing)

        query = Entity(name="Bob", type=EntityType.Person)
        resolver = EntityResolver(graph=graph)
        candidates = resolver.find_candidates(query)
        assert len(candidates) >= 1
        assert any(c.id == existing.id for c in candidates)

    def test_finds_by_type(self, graph, sample_entity, sample_entity2):
        """Should find entities of the same type."""
        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)

        from src.knowledge.entity_resolution import EntityResolver
        from src.models.entity import Entity, EntityType

        resolver = EntityResolver(graph=graph)
        query = Entity(name="Charlie", type=EntityType.Person)
        candidates = resolver.find_candidates(query)
        person_ids = {c.id for c in candidates}
        assert sample_entity.id in person_ids
        assert sample_entity2.id not in person_ids

    def test_empty_graph_returns_empty_list(self, graph):
        """Should return empty list when graph is empty."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        resolver = EntityResolver(graph=graph)
        query = Entity(name="Nobody", type=EntityType.Person)
        candidates = resolver.find_candidates(query)
        assert candidates == []


class TestComputeScore:
    """Test compute_score method."""

    def test_exact_name_match(self):
        """Exact name match should score 1.0."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        e1 = Entity(name="Alice", type=EntityType.Person)
        e2 = Entity(name="Alice", type=EntityType.Person)
        resolver = EntityResolver(graph=None)  # graph not needed for scoring
        score = resolver.compute_score(e1, e2)
        assert score == 1.0

    def test_alias_match(self):
        """Alias match should score 0.9."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        e1 = Entity(name="Bob", type=EntityType.Person)
        e2 = Entity(name="Robert", type=EntityType.Person, aliases=["Bob"])
        resolver = EntityResolver(graph=None)
        score = resolver.compute_score(e1, e2)
        assert score == 0.9

    def test_levenshtein_close_match(self):
        """Levenshtein distance < 3 should score 0.7."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        e1 = Entity(name="Alice", type=EntityType.Person)
        e2 = Entity(name="Alicia", type=EntityType.Organization)
        resolver = EntityResolver(graph=None)
        score = resolver.compute_score(e1, e2)
        assert score == 0.7

    def test_type_compatibility(self):
        """Same type should add 0.3."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        e1 = Entity(name="Something", type=EntityType.Person)
        e2 = Entity(name="Different", type=EntityType.Person)
        resolver = EntityResolver(graph=None)
        score = resolver.compute_score(e1, e2)
        assert score == 0.3

    def test_no_match(self):
        """Completely different entities should score 0.0."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        e1 = Entity(name="Alice", type=EntityType.Person)
        e2 = Entity(name="XyzzyCorp", type=EntityType.Organization)
        resolver = EntityResolver(graph=None)
        score = resolver.compute_score(e1, e2)
        assert score == 0.0

    def test_combined_signals(self):
        """Levenshtein + type should combine to 1.0 (0.7 + 0.3)."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        e1 = Entity(name="Alice", type=EntityType.Person)
        e2 = Entity(name="Alic", type=EntityType.Person)
        resolver = EntityResolver(graph=None)
        score = resolver.compute_score(e1, e2)
        assert score == 1.0


class TestResolve:
    """Test resolve method."""

    def test_resolve_exact_match_merges(self, graph):
        """Exact name match (>0.85) should return the existing entity (merge)."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        existing = Entity(name="Alice", type=EntityType.Person)
        graph.create_entity(existing)

        query = Entity(name="Alice", type=EntityType.Person)
        resolver = EntityResolver(graph=graph)
        result = resolver.resolve(query)
        assert result.id == existing.id

    def test_resolve_no_match_creates_new(self, graph):
        """No match found should create a new entity in the graph."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        query = Entity(name="NewPerson", type=EntityType.Person)
        resolver = EntityResolver(graph=graph)
        result = resolver.resolve(query)
        assert result.id == query.id
        assert result.name == "NewPerson"
        # Verify it was stored
        fetched = graph.get_entity(result.id)
        assert fetched is not None

    def test_resolve_suggests_below_merge_threshold(self, graph):
        """Score between 0.6 and 0.85 should create new entity (no merge)."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        # Levenshtein match (0.7) with different type: score = 0.7 (in suggest range)
        existing = Entity(name="Alice", type=EntityType.Person)
        graph.create_entity(existing)

        query = Entity(name="Alic", type=EntityType.Organization)
        resolver = EntityResolver(graph=graph)
        result = resolver.resolve(query)
        # Should create new entity, not merge
        assert result.id != existing.id

    def test_resolve_below_suggest_threshold(self, graph):
        """Score below 0.6 should create new entity."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        existing = Entity(name="Alice", type=EntityType.Person)
        graph.create_entity(existing)

        query = Entity(name="XyzzyCorp", type=EntityType.Organization)
        resolver = EntityResolver(graph=graph)
        result = resolver.resolve(query)
        assert result.id != existing.id


class TestMergeEntities:
    """Test merge_entities method."""

    def test_merge_combines_aliases(self, graph):
        """Merging should combine aliases from both entities."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        source = Entity(
            name="Robert",
            type=EntityType.Person,
            aliases=["Bob", "Bobby"],
            confidence=0.8,
        )
        target = Entity(
            name="Bob",
            type=EntityType.Person,
            aliases=["Rob"],
            confidence=0.9,
        )
        graph.create_entity(source)
        graph.create_entity(target)

        resolver = EntityResolver(graph=graph)
        result = resolver.merge_entities(source.id, target.id)

        assert "Bob" in result.aliases
        assert "Bobby" in result.aliases
        assert "Rob" in result.aliases

    def test_merge_keeps_higher_confidence(self, graph):
        """Merging should keep the higher confidence score."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        source = Entity(name="Alice", type=EntityType.Person, confidence=0.95)
        target = Entity(name="Alice", type=EntityType.Person, confidence=0.7)
        graph.create_entity(source)
        graph.create_entity(target)

        resolver = EntityResolver(graph=graph)
        result = resolver.merge_entities(source.id, target.id)
        assert result.confidence == 0.95

    def test_merge_combines_source_note_ids(self, graph):
        """Merging should combine source_note_ids from both entities."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        source = Entity(
            name="Alice",
            type=EntityType.Person,
            source_note_ids=["note-1", "note-2"],
        )
        target = Entity(
            name="Alice",
            type=EntityType.Person,
            source_note_ids=["note-3"],
        )
        graph.create_entity(source)
        graph.create_entity(target)

        resolver = EntityResolver(graph=graph)
        result = resolver.merge_entities(source.id, target.id)

        assert "note-1" in result.source_note_ids
        assert "note-2" in result.source_note_ids
        assert "note-3" in result.source_note_ids

    def test_merge_combines_properties(self, graph):
        """Merging should combine properties from both entities."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        source = Entity(
            name="Alice",
            type=EntityType.Person,
            properties={"email": "alice@example.com"},
        )
        target = Entity(
            name="Alice",
            type=EntityType.Person,
            properties={"phone": "555-1234"},
        )
        graph.create_entity(source)
        graph.create_entity(target)

        resolver = EntityResolver(graph=graph)
        result = resolver.merge_entities(source.id, target.id)

        assert result.properties.get("email") == "alice@example.com"
        assert result.properties.get("phone") == "555-1234"

    def test_merge_returns_target_entity(self, graph):
        """Merging should return the target entity (not source)."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        source = Entity(name="Bob", type=EntityType.Person)
        target = Entity(name="Robert", type=EntityType.Person)
        graph.create_entity(source)
        graph.create_entity(target)

        resolver = EntityResolver(graph=graph)
        result = resolver.merge_entities(source.id, target.id)
        assert result.id == target.id

    def test_merge_removes_source_entity(self, graph):
        """After merging, the source entity should be removed from the graph."""
        from src.models.entity import Entity, EntityType
        from src.knowledge.entity_resolution import EntityResolver

        source = Entity(name="Bob", type=EntityType.Person)
        target = Entity(name="Robert", type=EntityType.Person)
        graph.create_entity(source)
        graph.create_entity(target)

        resolver = EntityResolver(graph=graph)
        resolver.merge_entities(source.id, target.id)

        assert graph.get_entity(source.id) is None