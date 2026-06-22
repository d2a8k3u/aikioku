"""Tests for JWT auth helpers and the LOCAL-TRUST optional-auth model (Phase 4).

Post-refactor: the signing secret is persisted via the encrypted secrets store
(``auth.get_secret_key()``), users live in the ``users`` SQLite table (no in-memory
``_users_db``), and ``require_auth`` reads the DB-backed ``runtime_config.auth_required``.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException


OLD_HARDCODED_SECRET = "aikioku-dev-secret-change-me"


def _ensure_user(auth, username: str, email: str = "u@local", pw: str = "pw-secret"):
    if auth._get_user(username) is None:
        auth.register_user(username, email, pw)


class TestSigningSecret:
    """The signing secret must never be the old checked-in constant."""

    def test_secret_key_is_not_old_hardcoded_constant(self):
        from src import auth

        secret = auth.get_secret_key()
        assert secret != OLD_HARDCODED_SECRET
        assert secret  # non-empty


class TestTokenRoundTrip:
    """A token signed with the active secret round-trips through get_current_user."""

    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self):
        from src import auth

        username = "round_trip_user"
        _ensure_user(auth, username, "rt@local")

        token = auth.create_access_token({"sub": username})
        user = await auth.get_current_user(token)

        assert user is not None
        assert user.username == username

    @pytest.mark.asyncio
    async def test_token_signed_with_old_constant_is_rejected(self):
        """A token signed with the retired hardcoded secret must not decode."""
        from jose import jwt

        from src import auth

        username = "old_secret_user"
        _ensure_user(auth, username, "os@local")

        forged = jwt.encode(
            {"sub": username}, OLD_HARDCODED_SECRET, algorithm=auth.ALGORITHM
        )
        # Only valid if the active secret happens to equal the old one (it must not).
        user = await auth.get_current_user(forged)
        assert user is None


class TestRequireAuthLocalTrust:
    """auth_required=False (default): optional auth — anonymous allowed."""

    @pytest.mark.asyncio
    async def test_no_token_returns_anonymous_when_auth_not_required(self, monkeypatch):
        from src import auth

        monkeypatch.setattr(auth.runtime_config, "auth_required", lambda: False)
        user = await auth.require_auth(None)
        assert user is not None
        assert user.username == "anonymous"

    @pytest.mark.asyncio
    async def test_valid_token_returns_real_user_when_auth_not_required(
        self, monkeypatch
    ):
        from src import auth

        monkeypatch.setattr(auth.runtime_config, "auth_required", lambda: False)
        username = "local_user"
        _ensure_user(auth, username, "lu@local")
        token = auth.create_access_token({"sub": username})
        user_obj = await auth.get_current_user(token)
        result = await auth.require_auth(user_obj)
        assert result.username == username


class TestRequireAuthEnforced:
    """auth_required=True: a missing/invalid user raises 401; valid user passes."""

    @pytest.mark.asyncio
    async def test_no_token_raises_401_when_auth_required(self, monkeypatch):
        from src import auth

        monkeypatch.setattr(auth.runtime_config, "auth_required", lambda: True)
        with pytest.raises(HTTPException) as exc_info:
            await auth.require_auth(None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_user_passes_when_auth_required(self, monkeypatch):
        from src import auth

        monkeypatch.setattr(auth.runtime_config, "auth_required", lambda: True)
        username = "enforced_user"
        _ensure_user(auth, username, "eu@local")
        token = auth.create_access_token({"sub": username})
        user_obj = await auth.get_current_user(token)
        result = await auth.require_auth(user_obj)
        assert result.username == username
