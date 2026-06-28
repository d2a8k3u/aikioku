"""Integration tests for Knowledge Pipeline.

Tests entity and relation CRUD, type-based queries, and path finding
using a real Kuzu database in a temp directory.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest

from src.knowledge.graph import KnowledgeGraph
from src.models.entity import Entity, EntityType
from src.models.relation import Relation, RelationType


@pytest.fixture
def kg():
    """Create a KnowledgeGraph backed by a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.kg")
        yield KnowledgeGraph(db_path=db_path)


def test_entity_creation_and_retrieval(kg):
    """Create entity via KnowledgeGraph, retrieve it, verify fields."""
    entity = Entity(
        id=str(uuid.uuid4()),
        name="Python",
        type=EntityType.Concept,
        aliases=["py", "python3"],
        properties={"category": "programming"},
        confidence=0.95,
        source_note_ids=["note-1"],
    )
    kg.create_entity(entity)

    retrieved = kg.get_entity(entity.id)
    assert retrieved is not None
    assert retrieved.id == entity.id
    assert retrieved.name == "Python"
    assert retrieved.type == EntityType.Concept
    assert retrieved.aliases == ["py", "python3"]
    assert retrieved.properties == {"category": "programming"}
    assert retrieved.confidence == 0.95
    assert retrieved.source_note_ids == ["note-1"]


def test_entity_update(kg):
    """Create entity, update name, verify change persisted."""
    entity = Entity(
        id=str(uuid.uuid4()),
        name="OldName",
        type=EntityType.Concept,
        confidence=0.8,
    )
    kg.create_entity(entity)

    entity.name = "NewName"
    entity.confidence = 0.9
    kg.update_entity(entity)

    retrieved = kg.get_entity(entity.id)
    assert retrieved is not None
    assert retrieved.name == "NewName"
    assert retrieved.confidence == 0.9


def test_entity_delete(kg):
    """Create entity, delete, verify not found."""
    entity = Entity(
        id=str(uuid.uuid4()),
        name="ToDelete",
        type=EntityType.Concept,
    )
    kg.create_entity(entity)
    assert kg.get_entity(entity.id) is not None

    result = kg.delete_entity(entity.id)
    assert result is True
    assert kg.get_entity(entity.id) is None


def test_relation_creation(kg):
    """Create two entities, create relation between them, verify."""
    entity_a = Entity(
        id=str(uuid.uuid4()),
        name="Alice",
        type=EntityType.Person,
    )
    entity_b = Entity(
        id=str(uuid.uuid4()),
        name="CompanyX",
        type=EntityType.Organization,
    )
    kg.create_entity(entity_a)
    kg.create_entity(entity_b)

    relation = Relation(
        id=str(uuid.uuid4()),
        source_entity_id=entity_a.id,
        target_entity_id=entity_b.id,
        type=RelationType.works_at,
        confidence=0.85,
        properties={"since": "2020"},
    )
    kg.create_relation(relation)

    assert kg.count_relations() == 1


def test_find_entities_by_type(kg):
    """Create entities of different types, find by type."""
    person = Entity(
        id=str(uuid.uuid4()),
        name="Bob",
        type=EntityType.Person,
    )
    concept = Entity(
        id=str(uuid.uuid4()),
        name="Machine Learning",
        type=EntityType.Concept,
    )
    org = Entity(
        id=str(uuid.uuid4()),
        name="Acme Corp",
        type=EntityType.Organization,
    )
    kg.create_entity(person)
    kg.create_entity(concept)
    kg.create_entity(org)

    persons = kg.find_entities(type=EntityType.Person.value)
    assert len(persons) == 1
    assert persons[0].name == "Bob"

    concepts = kg.find_entities(type=EntityType.Concept.value)
    assert len(concepts) == 1
    assert concepts[0].name == "Machine Learning"

    all_entities = kg.find_entities(limit=10)
    assert len(all_entities) == 3


def test_find_paths(kg):
    """Create chain A -> B -> C, find paths from A to C."""
    entity_a = Entity(
        id=str(uuid.uuid4()),
        name="Alice",
        type=EntityType.Person,
    )
    entity_b = Entity(
        id=str(uuid.uuid4()),
        name="CompanyX",
        type=EntityType.Organization,
    )
    entity_c = Entity(
        id=str(uuid.uuid4()),
        name="ProjectY",
        type=EntityType.Project,
    )
    kg.create_entity(entity_a)
    kg.create_entity(entity_b)
    kg.create_entity(entity_c)

    kg.create_relation(
        Relation(
            id=str(uuid.uuid4()),
            source_entity_id=entity_a.id,
            target_entity_id=entity_b.id,
            type=RelationType.works_at,
            confidence=0.8,
        )
    )
    kg.create_relation(
        Relation(
            id=str(uuid.uuid4()),
            source_entity_id=entity_b.id,
            target_entity_id=entity_c.id,
            type=RelationType.created,
            confidence=0.7,
        )
    )

    paths = kg.find_paths(entity_a.id, entity_c.id, max_depth=2)
    assert len(paths) == 1
    assert len(paths[0]) == 3
    assert paths[0][0].name == "Alice"
    assert paths[0][1].name == "CompanyX"
    assert paths[0][2].name == "ProjectY"
