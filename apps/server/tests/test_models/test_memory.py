"""Tests for Memory model."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError


class TestMemoryCreation:
    """Test basic Memory creation."""

    def test_create_memory_with_required_fields(self):
        from src.models.memory import Memory

        memory = Memory(
            subject="Python",
            predicate="is_a",
            object="programming language",
            source="note-001",
        )
        assert memory.subject == "Python"
        assert memory.predicate == "is_a"
        assert memory.object == "programming language"
        assert memory.source == "note-001"

    def test_confidence_defaults_to_zero(self):
        from src.models.memory import Memory

        memory = Memory(subject="A", predicate="relates_to", object="B", source="src")
        assert memory.confidence == 0.0

    def test_vitality_score_defaults_to_zero(self):
        from src.models.memory import Memory

        memory = Memory(subject="A", predicate="relates_to", object="B", source="src")
        assert memory.vitality_score == 0.0

    def test_tier_defaults_to_hot(self):
        from src.models.memory import Memory, MemoryTier

        memory = Memory(subject="A", predicate="relates_to", object="B", source="src")
        assert memory.tier == MemoryTier.hot

    def test_created_and_modified_default_to_utcnow(self):
        from src.models.memory import Memory

        before = datetime.utcnow()
        memory = Memory(subject="A", predicate="relates_to", object="B", source="src")
        after = datetime.utcnow()
        assert before <= memory.created <= after
        assert before <= memory.modified <= after


class TestMemoryWithAllFields:
    """Test Memory creation with all fields."""

    def test_create_memory_with_all_fields(self, fixed_uuid, fixed_datetime, fixed_datetime_later):
        from src.models.memory import Memory, MemoryTier

        memory = Memory(
            id=fixed_uuid,
            subject="ML",
            predicate="requires",
            object="math",
            confidence=0.92,
            source="research-notes",
            created=fixed_datetime,
            modified=fixed_datetime_later,
            vitality_score=0.75,
            tier=MemoryTier.warm,
        )
        assert memory.id == fixed_uuid
        assert memory.confidence == 0.92
        assert memory.created == fixed_datetime
        assert memory.modified == fixed_datetime_later
        assert memory.vitality_score == 0.75
        assert memory.tier == MemoryTier.warm


class TestMemoryTierEnum:
    """Test MemoryTier enum values."""

    def test_all_memory_tiers_exist(self):
        from src.models.memory import MemoryTier

        assert MemoryTier.hot == "hot"
        assert MemoryTier.warm == "warm"
        assert MemoryTier.cold == "cold"


class TestMemoryValidation:
    """Test Memory validation rules."""

    def test_empty_subject_raises_error(self):
        from src.models.memory import Memory

        with pytest.raises(ValidationError):
            Memory(subject="", predicate="is", object="B", source="src")

    def test_empty_predicate_raises_error(self):
        from src.models.memory import Memory

        with pytest.raises(ValidationError):
            Memory(subject="A", predicate="", object="B", source="src")

    def test_empty_object_raises_error(self):
        from src.models.memory import Memory

        with pytest.raises(ValidationError):
            Memory(subject="A", predicate="is", object="", source="src")

    def test_empty_source_raises_error(self):
        from src.models.memory import Memory

        with pytest.raises(ValidationError):
            Memory(subject="A", predicate="is", object="B", source="")

    def test_confidence_below_zero_raises_error(self):
        from src.models.memory import Memory

        with pytest.raises(ValidationError):
            Memory(
                subject="A",
                predicate="is",
                object="B",
                source="src",
                confidence=-0.1,
            )

    def test_confidence_above_one_raises_error(self):
        from src.models.memory import Memory

        with pytest.raises(ValidationError):
            Memory(
                subject="A",
                predicate="is",
                object="B",
                source="src",
                confidence=1.5,
            )

    def test_confidence_at_boundaries_is_valid(self):
        from src.models.memory import Memory

        m1 = Memory(
            subject="A",
            predicate="is",
            object="B",
            source="src",
            confidence=0.0,
        )
        m2 = Memory(
            subject="A",
            predicate="is",
            object="B",
            source="src",
            confidence=1.0,
        )
        assert m1.confidence == 0.0
        assert m2.confidence == 1.0

    def test_model_is_serializable(self):
        from src.models.memory import Memory, MemoryTier

        memory = Memory(
            subject="AI",
            predicate="enables",
            object="automation",
            confidence=0.88,
            source="tech-notes",
            tier=MemoryTier.hot,
            vitality_score=0.95,
        )
        data = memory.model_dump()
        assert data["subject"] == "AI"
        assert data["tier"] == "hot"
        assert data["confidence"] == 0.88
