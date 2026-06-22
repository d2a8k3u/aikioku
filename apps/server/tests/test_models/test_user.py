"""Tests for User model."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError


class TestUserCreation:
    """Test basic User creation with required and default fields."""

    def test_create_user_with_required_fields(self):
        from src.models.user import User

        user = User(name="Alice", email="alice@example.com")
        assert user.name == "Alice"
        assert user.email == "alice@example.com"

    def test_id_is_auto_generated_uuid4(self):
        from src.models.user import User

        user = User(name="Bob", email="bob@example.com")
        parsed = uuid.UUID(user.id)
        assert parsed.version == 4

    def test_preferences_defaults_to_empty_dict(self):
        from src.models.user import User

        user = User(name="Charlie", email="charlie@example.com")
        assert user.preferences == {}

    def test_created_at_and_updated_at_default_to_utcnow(self):
        from src.models.user import User

        before = datetime.utcnow()
        user = User(name="Dave", email="dave@example.com")
        after = datetime.utcnow()
        assert before <= user.created_at <= after
        assert before <= user.updated_at <= after


class TestUserWithAllFields:
    """Test User creation with all fields specified."""

    def test_create_user_with_all_fields(self, fixed_uuid, fixed_datetime):
        from src.models.user import User

        user = User(
            id=fixed_uuid,
            name="Eve",
            email="eve@example.com",
            preferences={"theme": "dark"},
            created_at=fixed_datetime,
            updated_at=fixed_datetime,
        )
        assert user.id == fixed_uuid
        assert user.name == "Eve"
        assert user.email == "eve@example.com"
        assert user.preferences == {"theme": "dark"}
        assert user.created_at == fixed_datetime
        assert user.updated_at == fixed_datetime


class TestUserValidation:
    """Test User validation rules."""

    def test_empty_name_raises_error(self):
        from src.models.user import User

        with pytest.raises(ValidationError):
            User(name="", email="test@example.com")

    def test_empty_email_raises_error(self):
        from src.models.user import User

        with pytest.raises(ValidationError):
            User(name="Test", email="")

    def test_dict_is_serializable(self):
        from src.models.user import User

        user = User(
            name="Frank",
            email="frank@example.com",
            preferences={"lang": "en"},
        )
        data = user.model_dump()
        assert data["name"] == "Frank"
        assert data["email"] == "frank@example.com"
        assert data["preferences"] == {"lang": "en"}
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data
