"""Tests for Entity model."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError


class TestEntityCreation:
    """Test basic Entity creation."""

    def test_create_entity_with_required_fields(self):
        from src.models.entity import Entity, EntityType

        entity = Entity(name="Alice", type=EntityType.Person)
        assert entity.name == "Alice"
        assert entity.type == EntityType.Person

    def test_id_is_auto_generated_uuid4(self):
        from src.models.entity import Entity, EntityType

        entity = Entity(name="Bob", type=EntityType.Person)
        parsed = uuid.UUID(entity.id)
        assert parsed.version == 4

    def test_aliases_defaults_to_empty_list(self):
        from src.models.entity import Entity, EntityType

        entity = Entity(name="Bob", type=EntityType.Person)
        assert entity.aliases == []

    def test_properties_defaults_to_empty_dict(self):
        from src.models.entity import Entity, EntityType

        entity = Entity(name="Bob", type=EntityType.Person)
        assert entity.properties == {}

    def test_confidence_defaults_to_zero(self):
        from src.models.entity import Entity, EntityType

        entity = Entity(name="Bob", type=EntityType.Person)
        assert entity.confidence == 0.0

    def test_source_note_ids_defaults_to_empty_list(self):
        from src.models.entity import Entity, EntityType

        entity = Entity(name="Bob", type=EntityType.Person)
        assert entity.source_note_ids == []


class TestEntityWithAllFields:
    """Test Entity creation with all fields."""

    def test_create_entity_with_all_fields(self, fixed_uuid, fixed_uuid2):
        from src.models.entity import Entity, EntityType

        entity = Entity(
            id=fixed_uuid,
            name="Acme Corp",
            type=EntityType.Organization,
            aliases=["ACME", "Acme"],
            properties={"industry": "tech"},
            confidence=0.95,
            source_note_ids=[fixed_uuid2],
        )
        assert entity.id == fixed_uuid
        assert entity.aliases == ["ACME", "Acme"]
        assert entity.properties == {"industry": "tech"}
        assert entity.confidence == 0.95
        assert entity.source_note_ids == [fixed_uuid2]


class TestEntityTypeEnum:
    """Test EntityType enum values."""

    def test_all_entity_types_exist(self):
        from src.models.entity import EntityType

        assert EntityType.Person == "Person"
        assert EntityType.Place == "Place"
        assert EntityType.Concept == "Concept"
        assert EntityType.Project == "Project"
        assert EntityType.Event == "Event"
        assert EntityType.Organization == "Organization"
        assert EntityType.Document == "Document"
        assert EntityType.Task == "Task"


class TestEntityValidation:
    """Test Entity validation rules."""

    def test_empty_name_raises_error(self):
        from src.models.entity import Entity, EntityType

        with pytest.raises(ValidationError):
            Entity(name="", type=EntityType.Person)

    def test_confidence_below_zero_raises_error(self):
        from src.models.entity import Entity, EntityType

        with pytest.raises(ValidationError):
            Entity(name="Test", type=EntityType.Person, confidence=-0.1)

    def test_confidence_above_one_raises_error(self):
        from src.models.entity import Entity, EntityType

        with pytest.raises(ValidationError):
            Entity(name="Test", type=EntityType.Person, confidence=1.1)

    def test_confidence_at_boundaries_is_valid(self):
        from src.models.entity import Entity, EntityType

        e1 = Entity(name="Low", type=EntityType.Person, confidence=0.0)
        e2 = Entity(name="High", type=EntityType.Person, confidence=1.0)
        assert e1.confidence == 0.0
        assert e2.confidence == 1.0

    def test_model_is_serializable(self):
        from src.models.entity import Entity, EntityType

        entity = Entity(
            name="Test",
            type=EntityType.Concept,
            aliases=["t"],
            confidence=0.5,
        )
        data = entity.model_dump()
        assert data["name"] == "Test"
        assert data["type"] == "Concept"
        assert data["confidence"] == 0.5
