"""Tests for Relation model."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError


class TestRelationCreation:
    """Test basic Relation creation."""

    def test_create_relation_with_required_fields(self, fixed_uuid, fixed_uuid2):
        from src.models.relation import Relation, RelationType

        relation = Relation(
            source_entity_id=fixed_uuid,
            target_entity_id=fixed_uuid2,
            type=RelationType.works_at,
        )
        assert relation.source_entity_id == fixed_uuid
        assert relation.target_entity_id == fixed_uuid2
        assert relation.type == RelationType.works_at

    def test_id_is_auto_generated_uuid4(self):
        from src.models.relation import Relation, RelationType

        relation = Relation(
            source_entity_id=str(uuid.uuid4()),
            target_entity_id=str(uuid.uuid4()),
            type=RelationType.related_to,
        )
        parsed = uuid.UUID(relation.id)
        assert parsed.version == 4

    def test_confidence_defaults_to_zero(self):
        from src.models.relation import Relation, RelationType

        relation = Relation(
            source_entity_id=str(uuid.uuid4()),
            target_entity_id=str(uuid.uuid4()),
            type=RelationType.related_to,
        )
        assert relation.confidence == 0.0

    def test_properties_defaults_to_empty_dict(self):
        from src.models.relation import Relation, RelationType

        relation = Relation(
            source_entity_id=str(uuid.uuid4()),
            target_entity_id=str(uuid.uuid4()),
            type=RelationType.related_to,
        )
        assert relation.properties == {}


class TestRelationWithAllFields:
    """Test Relation creation with all fields."""

    def test_create_relation_with_all_fields(self, fixed_uuid, fixed_uuid2):
        from src.models.relation import Relation, RelationType

        relation = Relation(
            id=fixed_uuid,
            source_entity_id=fixed_uuid,
            target_entity_id=fixed_uuid2,
            type=RelationType.depends_on,
            confidence=0.85,
            properties={"weight": 0.7, "bidirectional": False},
        )
        assert relation.id == fixed_uuid
        assert relation.confidence == 0.85
        assert relation.properties == {"weight": 0.7, "bidirectional": False}


class TestRelationTypeEnum:
    """Test RelationType enum values."""

    def test_all_relation_types_exist(self):
        from src.models.relation import RelationType

        assert RelationType.works_at == "works_at"
        assert RelationType.created == "created"
        assert RelationType.depends_on == "depends_on"
        assert RelationType.related_to == "related_to"
        assert RelationType.part_of == "part_of"
        assert RelationType.located_in == "located_in"
        assert RelationType.mentions == "mentions"
        assert RelationType.follows == "follows"


class TestRelationValidation:
    """Test Relation validation rules."""

    def test_confidence_below_zero_raises_error(self):
        from src.models.relation import Relation, RelationType

        with pytest.raises(ValidationError):
            Relation(
                source_entity_id=str(uuid.uuid4()),
                target_entity_id=str(uuid.uuid4()),
                type=RelationType.related_to,
                confidence=-0.1,
            )

    def test_confidence_above_one_raises_error(self):
        from src.models.relation import Relation, RelationType

        with pytest.raises(ValidationError):
            Relation(
                source_entity_id=str(uuid.uuid4()),
                target_entity_id=str(uuid.uuid4()),
                type=RelationType.related_to,
                confidence=1.5,
            )

    def test_confidence_at_boundaries_is_valid(self):
        from src.models.relation import Relation, RelationType

        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())
        r1 = Relation(
            source_entity_id=id1, target_entity_id=id2,
            type=RelationType.related_to, confidence=0.0,
        )
        r2 = Relation(
            source_entity_id=id1, target_entity_id=id2,
            type=RelationType.related_to, confidence=1.0,
        )
        assert r1.confidence == 0.0
        assert r2.confidence == 1.0

    def test_model_is_serializable(self):
        from src.models.relation import Relation, RelationType

        relation = Relation(
            source_entity_id=str(uuid.uuid4()),
            target_entity_id=str(uuid.uuid4()),
            type=RelationType.mentions,
            confidence=0.9,
            properties={"context": "cited"},
        )
        data = relation.model_dump()
        assert data["type"] == "mentions"
        assert data["confidence"] == 0.9
        assert data["properties"] == {"context": "cited"}
