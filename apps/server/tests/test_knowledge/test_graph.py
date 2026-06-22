"""Tests for KnowledgeGraph class using Kuzu embedded database."""

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


@pytest.fixture
def sample_entities():
    """Provide multiple sample entities."""
    from src.models.entity import Entity, EntityType

    return [
        Entity(name="Alice", type=EntityType.Person),
        Entity(name="Bob", type=EntityType.Person),
        Entity(name="Acme Corp", type=EntityType.Organization),
        Entity(name="Project X", type=EntityType.Project),
        Entity(name="NYC", type=EntityType.Place),
    ]


class TestKnowledgeGraphInit:
    """Test KnowledgeGraph initialization."""

    def test_init_creates_database(self, tmp_db_path):
        from src.knowledge.graph import KnowledgeGraph

        kg = KnowledgeGraph(db_path=tmp_db_path)
        assert kg is not None
        assert kg.db_path is not None

    def test_init_creates_entity_table(self, tmp_db_path):
        from src.knowledge.graph import KnowledgeGraph

        kg = KnowledgeGraph(db_path=tmp_db_path)
        # Should be able to count entities (schema exists)
        assert kg.count_entities() == 0

    def test_init_creates_relation_table(self, tmp_db_path):
        from src.knowledge.graph import KnowledgeGraph

        kg = KnowledgeGraph(db_path=tmp_db_path)
        # Count should work (schema exists) — no entities yet
        assert kg.count_entities() == 0


class TestCreateEntity:
    """Test entity creation."""

    def test_create_entity_returns_entity(self, graph, sample_entity):
        result = graph.create_entity(sample_entity)
        assert result.id == sample_entity.id
        assert result.name == "Alice"
        assert result.type == "Person"

    def test_create_entity_stores_in_db(self, graph, sample_entity):
        graph.create_entity(sample_entity)
        fetched = graph.get_entity(sample_entity.id)
        assert fetched is not None
        assert fetched.name == "Alice"

    def test_create_entity_increments_count(self, graph, sample_entity, sample_entity2):
        from src.models.entity import Entity, EntityType

        e1 = Entity(name="A", type=EntityType.Person)
        graph.create_entity(e1)
        assert graph.count_entities() == 1

        e2 = Entity(name="B", type=EntityType.Organization)
        graph.create_entity(e2)
        assert graph.count_entities() == 2

    def test_create_entity_with_all_fields(self, graph):
        from src.models.entity import Entity, EntityType

        entity = Entity(
            name="Test",
            type=EntityType.Project,
            aliases=["t1", "t2"],
            properties={"key": "value"},
            confidence=0.75,
            source_note_ids=["note-1"],
        )
        result = graph.create_entity(entity)
        fetched = graph.get_entity(result.id)
        assert fetched.name == "Test"
        assert fetched.aliases == ["t1", "t2"]
        assert fetched.properties["key"] == "value"
        assert fetched.confidence == 0.75


class TestGetEntity:
    """Test entity retrieval."""

    def test_get_existing_entity(self, graph, sample_entity):
        graph.create_entity(sample_entity)
        fetched = graph.get_entity(sample_entity.id)
        assert fetched is not None
        assert fetched.id == sample_entity.id
        assert fetched.name == "Alice"

    def test_get_nonexistent_entity_returns_none(self, graph):
        result = graph.get_entity("nonexistent-id")
        assert result is None


class TestUpdateEntity:
    """Test entity update."""

    def test_update_entity_changes_fields(self, graph, sample_entity):
        graph.create_entity(sample_entity)
        sample_entity.name = "Alice Updated"
        sample_entity.confidence = 0.5
        updated = graph.update_entity(sample_entity)
        assert updated.name == "Alice Updated"
        assert updated.confidence == 0.5

    def test_update_entity_persists(self, graph, sample_entity):
        graph.create_entity(sample_entity)
        sample_entity.name = "Alice Modified"
        graph.update_entity(sample_entity)
        fetched = graph.get_entity(sample_entity.id)
        assert fetched.name == "Alice Modified"


class TestDeleteEntity:
    """Test entity deletion."""

    def test_delete_existing_returns_true(self, graph, sample_entity):
        graph.create_entity(sample_entity)
        result = graph.delete_entity(sample_entity.id)
        assert result is True

    def test_delete_removes_from_db(self, graph, sample_entity):
        graph.create_entity(sample_entity)
        graph.delete_entity(sample_entity.id)
        assert graph.get_entity(sample_entity.id) is None

    def test_delete_decrements_count(self, graph, sample_entity):
        graph.create_entity(sample_entity)
        assert graph.count_entities() == 1
        graph.delete_entity(sample_entity.id)
        assert graph.count_entities() == 0

    def test_delete_nonexistent_returns_false(self, graph):
        result = graph.delete_entity("nonexistent-id")
        assert result is False


class TestCreateRelation:
    """Test relation creation between entities."""

    def test_create_relation_returns_relation(self, graph, sample_entity, sample_entity2):
        from src.models.relation import Relation, RelationType

        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)

        rel = Relation(
            source_entity_id=sample_entity.id,
            target_entity_id=sample_entity2.id,
            type=RelationType.works_at,
        )
        result = graph.create_relation(rel)
        assert result.id == rel.id
        assert result.source_entity_id == sample_entity.id
        assert result.target_entity_id == sample_entity2.id
        assert result.type == "works_at"

    def test_create_relation_persists(self, graph, sample_entity, sample_entity2):
        from src.models.relation import Relation, RelationType

        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)

        rel = Relation(
            source_entity_id=sample_entity.id,
            target_entity_id=sample_entity2.id,
            type=RelationType.works_at,
        )
        graph.create_relation(rel)

        # Verify by traversing: find_paths should find the path
        paths = graph.find_paths(sample_entity.id, sample_entity2.id)
        assert len(paths) == 1
        assert len(paths[0]) == 2  # Two entities in the path


class TestFindEntities:
    """Test entity search/filter."""

    def test_find_all_no_filters(self, graph, sample_entities):
        for e in sample_entities:
            graph.create_entity(e)
        results = graph.find_entities()
        assert len(results) == 5

    def test_find_by_type(self, graph, sample_entities):
        for e in sample_entities:
            graph.create_entity(e)
        results = graph.find_entities(type="Person")
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"Alice", "Bob"}

    def test_find_by_name_partial(self, graph, sample_entities):
        for e in sample_entities:
            graph.create_entity(e)
        results = graph.find_entities(name="Ali")
        assert len(results) == 1
        assert results[0].name == "Alice"

    def test_find_with_limit(self, graph, sample_entities):
        for e in sample_entities:
            graph.create_entity(e)
        results = graph.find_entities(limit=2)
        assert len(results) == 2

    def test_find_no_matches(self, graph, sample_entities):
        for e in sample_entities:
            graph.create_entity(e)
        results = graph.find_entities(type="Task")
        assert len(results) == 0


class TestFindPaths:
    """Test graph path finding."""

    def test_direct_path(self, graph, sample_entity, sample_entity2):
        from src.models.relation import Relation, RelationType

        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)
        rel = Relation(
            source_entity_id=sample_entity.id,
            target_entity_id=sample_entity2.id,
            type=RelationType.works_at,
        )
        graph.create_relation(rel)

        paths = graph.find_paths(sample_entity.id, sample_entity2.id)
        assert len(paths) == 1
        assert len(paths[0]) == 2
        # Verify entity IDs are real UUIDs, not internal kuzu offsets
        assert paths[0][0].id == sample_entity.id
        assert paths[0][1].id == sample_entity2.id
        assert "offset" not in paths[0][0].id
        assert "offset" not in paths[0][1].id

    def test_no_path_returns_empty(self, graph, sample_entity, sample_entity2):
        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)

        paths = graph.find_paths(sample_entity.id, sample_entity2.id)
        assert paths == []

    def test_same_source_and_target(self, graph, sample_entity):
        graph.create_entity(sample_entity)
        paths = graph.find_paths(sample_entity.id, sample_entity.id)
        assert len(paths) == 1
        assert paths[0][0].id == sample_entity.id


class TestCountEntities:
    """Test entity counting."""

    def test_empty_graph_returns_zero(self, graph):
        assert graph.count_entities() == 0

    def test_count_after_inserts(self, graph):
        from src.models.entity import Entity, EntityType

        for i in range(3):
            graph.create_entity(Entity(name=f"E{i}", type=EntityType.Concept))
        assert graph.count_entities() == 3


class TestCountRelations:
    """Test relation counting."""

    def test_empty_graph_returns_zero(self, graph):
        assert graph.count_relations() == 0

    def test_count_after_creating_relations(self, graph, sample_entity, sample_entity2):
        from src.models.relation import Relation, RelationType

        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)

        rel = Relation(
            source_entity_id=sample_entity.id,
            target_entity_id=sample_entity2.id,
            type=RelationType.works_at,
        )
        graph.create_relation(rel)
        assert graph.count_relations() == 1

    def test_count_with_multiple_relations(self, graph):
        from src.models.entity import Entity, EntityType
        from src.models.relation import Relation, RelationType

        e1 = Entity(name="A", type=EntityType.Person)
        e2 = Entity(name="B", type=EntityType.Person)
        e3 = Entity(name="C", type=EntityType.Person)
        graph.create_entity(e1)
        graph.create_entity(e2)
        graph.create_entity(e3)

        graph.create_relation(
            Relation(
                source_entity_id=e1.id,
                target_entity_id=e2.id,
                type=RelationType.works_at,
            )
        )
        graph.create_relation(
            Relation(
                source_entity_id=e2.id,
                target_entity_id=e3.id,
                type=RelationType.created,
            )
        )
        assert graph.count_relations() == 2


class TestGetAllRelations:
    """Test bulk relation retrieval across the whole graph."""

    def test_empty_graph_returns_empty(self, graph):
        assert graph.get_all_relations() == []

    def test_returns_all_relations_with_correct_fields(self, graph):
        from src.models.entity import Entity, EntityType
        from src.models.relation import Relation, RelationType

        e1 = Entity(name="Alice", type=EntityType.Person)
        e2 = Entity(name="Acme", type=EntityType.Organization)
        e3 = Entity(name="Project X", type=EntityType.Project)
        graph.create_entity(e1)
        graph.create_entity(e2)
        graph.create_entity(e3)

        graph.create_relation(
            Relation(
                source_entity_id=e1.id,
                target_entity_id=e2.id,
                type=RelationType.works_at,
                confidence=0.9,
            )
        )
        graph.create_relation(
            Relation(
                source_entity_id=e1.id,
                target_entity_id=e3.id,
                type=RelationType.created,
                confidence=0.7,
            )
        )

        relations = graph.get_all_relations()
        assert len(relations) == 2
        pairs = {(r.source_entity_id, r.target_entity_id, r.type.value) for r in relations}
        assert (e1.id, e2.id, "works_at") in pairs
        assert (e1.id, e3.id, "created") in pairs

    def test_respects_limit(self, graph):
        from src.models.entity import Entity, EntityType
        from src.models.relation import Relation, RelationType

        e1 = Entity(name="A", type=EntityType.Person)
        e2 = Entity(name="B", type=EntityType.Person)
        e3 = Entity(name="C", type=EntityType.Person)
        graph.create_entity(e1)
        graph.create_entity(e2)
        graph.create_entity(e3)
        graph.create_relation(
            Relation(
                source_entity_id=e1.id,
                target_entity_id=e2.id,
                type=RelationType.related_to,
            )
        )
        graph.create_relation(
            Relation(
                source_entity_id=e2.id,
                target_entity_id=e3.id,
                type=RelationType.related_to,
            )
        )

        assert len(graph.get_all_relations(limit=1)) == 1


class TestFindRelation:
    """Test find_relation lookup by endpoints + type + predicate."""

    def test_find_existing_relation(self, graph, sample_entity, sample_entity2):
        from src.models.relation import Relation, RelationType

        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)
        graph.create_relation(
            Relation(
                source_entity_id=sample_entity.id,
                target_entity_id=sample_entity2.id,
                type=RelationType.works_at,
                properties={"predicate": "is employed by"},
            )
        )
        found = graph.find_relation(
            sample_entity.id, sample_entity2.id, "works_at", predicate="is employed by"
        )
        assert found is not None
        assert found.type == "works_at"
        assert found.properties["predicate"] == "is employed by"

    def test_find_relation_predicate_mismatch_returns_none(
        self, graph, sample_entity, sample_entity2
    ):
        from src.models.relation import Relation, RelationType

        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)
        graph.create_relation(
            Relation(
                source_entity_id=sample_entity.id,
                target_entity_id=sample_entity2.id,
                type=RelationType.works_at,
                properties={"predicate": "is employed by"},
            )
        )
        assert (
            graph.find_relation(
                sample_entity.id, sample_entity2.id, "works_at", predicate="founded"
            )
            is None
        )

    def test_find_nonexistent_relation_returns_none(self, graph, sample_entity, sample_entity2):
        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)
        assert graph.find_relation(sample_entity.id, sample_entity2.id, "works_at") is None


class TestDeleteRelation:
    """Test delete_relation, including predicate-preserving delete-and-restore."""

    def test_delete_relation_removes_edge(self, graph, sample_entity, sample_entity2):
        from src.models.relation import Relation, RelationType

        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)
        graph.create_relation(
            Relation(
                source_entity_id=sample_entity.id,
                target_entity_id=sample_entity2.id,
                type=RelationType.works_at,
            )
        )
        removed = graph.delete_relation(sample_entity.id, sample_entity2.id, "works_at")
        assert removed == 1
        assert graph.count_relations() == 0

    def test_delete_relation_predicate_preserves_other(self, graph, sample_entity, sample_entity2):
        from src.models.relation import Relation, RelationType

        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)
        # Two relations of the same RelationType between the same pair, different predicates.
        graph.create_relation(
            Relation(
                source_entity_id=sample_entity.id,
                target_entity_id=sample_entity2.id,
                type=RelationType.related_to,
                properties={"predicate": "discovered"},
            )
        )
        graph.create_relation(
            Relation(
                source_entity_id=sample_entity.id,
                target_entity_id=sample_entity2.id,
                type=RelationType.related_to,
                properties={"predicate": "admired"},
            )
        )
        removed = graph.delete_relation(
            sample_entity.id, sample_entity2.id, "related_to", predicate="discovered"
        )
        assert removed == 1
        survivor = graph.find_relation(
            sample_entity.id, sample_entity2.id, "related_to", predicate="admired"
        )
        assert survivor is not None
        assert (
            graph.find_relation(
                sample_entity.id, sample_entity2.id, "related_to", predicate="discovered"
            )
            is None
        )

    def test_delete_nonexistent_relation_returns_zero(self, graph, sample_entity, sample_entity2):
        graph.create_entity(sample_entity)
        graph.create_entity(sample_entity2)
        assert graph.delete_relation(sample_entity.id, sample_entity2.id, "works_at") == 0
