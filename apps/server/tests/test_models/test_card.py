"""Tests for Card model (spaced repetition flashcards)."""

from __future__ import annotations


import pytest
from pydantic import ValidationError


class TestCardCreation:
    """Test basic Card creation."""

    def test_create_card_with_required_fields(self, fixed_uuid, fixed_uuid2, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        card = Card(
            note_id=fixed_uuid,
            type=CardType.qa,
            front="What is Python?",
            back="A programming language",
            next_review=fixed_datetime,
            status=CardStatus.new,
        )
        assert card.note_id == fixed_uuid
        assert card.front == "What is Python?"
        assert card.back == "A programming language"
        assert card.next_review == fixed_datetime
        assert card.status == CardStatus.new

    def test_id_is_auto_generated_uuid4(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        card = Card(
            note_id=fixed_uuid,
            type=CardType.qa,
            front="Q",
            back="A",
            next_review=fixed_datetime,
            status=CardStatus.new,
        )
        import uuid

        parsed = uuid.UUID(card.id)
        assert parsed.version == 4

    def test_ease_factor_defaults_to_2_5(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        card = Card(
            note_id=fixed_uuid,
            type=CardType.qa,
            front="Q",
            back="A",
            next_review=fixed_datetime,
            status=CardStatus.new,
        )
        assert card.ease_factor == 2.5

    def test_interval_defaults_to_zero(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        card = Card(
            note_id=fixed_uuid,
            type=CardType.qa,
            front="Q",
            back="A",
            next_review=fixed_datetime,
            status=CardStatus.new,
        )
        assert card.interval == 0

    def test_repetitions_defaults_to_zero(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        card = Card(
            note_id=fixed_uuid,
            type=CardType.qa,
            front="Q",
            back="A",
            next_review=fixed_datetime,
            status=CardStatus.new,
        )
        assert card.repetitions == 0


class TestCardWithAllFields:
    """Test Card creation with all fields."""

    def test_create_card_with_all_fields(self, fixed_uuid, fixed_uuid2, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        card = Card(
            id=fixed_uuid,
            note_id=fixed_uuid2,
            type=CardType.cloze,
            front="The capital of France is {{Paris}}.",
            back="The capital of France is Paris.",
            ease_factor=2.0,
            interval=7,
            repetitions=3,
            next_review=fixed_datetime,
            status=CardStatus.review,
        )
        assert card.id == fixed_uuid
        assert card.note_id == fixed_uuid2
        assert card.ease_factor == 2.0
        assert card.interval == 7
        assert card.repetitions == 3
        assert card.status == CardStatus.review


class TestCardTypeStatusEnums:
    """Test CardType and CardStatus enum values."""

    def test_all_card_types_exist(self):
        from src.models.card import CardType

        assert CardType.cloze == "cloze"
        assert CardType.qa == "qa"
        assert CardType.connection == "connection"

    def test_all_card_statuses_exist(self):
        from src.models.card import CardStatus

        assert CardStatus.new == "new"
        assert CardStatus.learning == "learning"
        assert CardStatus.review == "review"
        assert CardStatus.suspended == "suspended"


class TestCardValidation:
    """Test Card validation rules."""

    def test_ease_factor_below_minimum_raises_error(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        with pytest.raises(ValidationError):
            Card(
                note_id=fixed_uuid,
                type=CardType.qa,
                front="Q",
                back="A",
                next_review=fixed_datetime,
                status=CardStatus.new,
                ease_factor=1.0,
            )

    def test_ease_factor_at_minimum_is_valid(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        card = Card(
            note_id=fixed_uuid,
            type=CardType.qa,
            front="Q",
            back="A",
            next_review=fixed_datetime,
            status=CardStatus.new,
            ease_factor=1.3,
        )
        assert card.ease_factor == 1.3

    def test_negative_interval_raises_error(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        with pytest.raises(ValidationError):
            Card(
                note_id=fixed_uuid,
                type=CardType.qa,
                front="Q",
                back="A",
                next_review=fixed_datetime,
                status=CardStatus.new,
                interval=-1,
            )

    def test_negative_repetitions_raises_error(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        with pytest.raises(ValidationError):
            Card(
                note_id=fixed_uuid,
                type=CardType.qa,
                front="Q",
                back="A",
                next_review=fixed_datetime,
                status=CardStatus.new,
                repetitions=-5,
            )

    def test_zero_interval_is_valid(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        card = Card(
            note_id=fixed_uuid,
            type=CardType.qa,
            front="Q",
            back="A",
            next_review=fixed_datetime,
            status=CardStatus.new,
            interval=0,
        )
        assert card.interval == 0

    def test_zero_repetitions_is_valid(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        card = Card(
            note_id=fixed_uuid,
            type=CardType.qa,
            front="Q",
            back="A",
            next_review=fixed_datetime,
            status=CardStatus.new,
            repetitions=0,
        )
        assert card.repetitions == 0

    def test_empty_front_raises_error(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        with pytest.raises(ValidationError):
            Card(
                note_id=fixed_uuid,
                type=CardType.qa,
                front="",
                back="A",
                next_review=fixed_datetime,
                status=CardStatus.new,
            )

    def test_model_is_serializable(self, fixed_uuid, fixed_datetime):
        from src.models.card import Card, CardType, CardStatus

        card = Card(
            note_id=fixed_uuid,
            type=CardType.connection,
            front="AI -> Machine Learning",
            back="Subset relationship",
            ease_factor=2.5,
            interval=14,
            repetitions=5,
            next_review=fixed_datetime,
            status=CardStatus.review,
        )
        data = card.model_dump()
        assert data["front"] == "AI -> Machine Learning"
        assert data["ease_factor"] == 2.5
        assert data["interval"] == 14
        assert data["repetitions"] == 5
        assert data["status"] == "review"
