"""Tests for the encrypted secrets store and DB-backed runtime config."""

from __future__ import annotations

import sqlite3

from src import secrets_store, runtime_config
from src.config import settings


class TestSecretsRoundTrip:
    def test_encrypt_decrypt_round_trip(self):
        secrets_store.set_secret("ollama_api_key", "sk-abc-123")
        assert secrets_store.get_secret("ollama_api_key") == "sk-abc-123"

    def test_value_stored_as_ciphertext_not_plaintext(self):
        secrets_store.set_secret("ollama_api_key", "sk-pla...pear")
        raw = sqlite3.connect(settings.sqlite_db_path).execute(
            "SELECT value FROM app_secrets WHERE key='ollama_api_key'"
        ).fetchone()[0]
        assert b"sk-pla...pear" not in raw

    def test_list_returns_names_only(self):
        secrets_store.set_secret("hf_api_key", "v1")
        keys = secrets_store.list_secret_keys()
        assert "hf_api_key" in keys

    def test_empty_value_deletes(self):
        secrets_store.set_secret("hf_api_key", "v")
        secrets_store.set_secret("hf_api_key", "")
        assert secrets_store.get_secret("hf_api_key") is None

    def test_delete_secret(self):
        secrets_store.set_secret("ollama_api_key", "v")
        secrets_store.delete_secret("ollama_api_key")
        assert secrets_store.get_secret("ollama_api_key") is None


class TestRuntimeConfig:
    def test_default_then_db_override(self):
        assert runtime_config.llm_provider() == "ollama"
        runtime_config.set_app_setting("llm_provider", "ollama_remote")
        assert runtime_config.llm_provider() == "ollama_remote"

    def test_bool_and_int_parsing(self):
        runtime_config.set_app_setting("embedding_strict", "false")
        assert runtime_config.embedding_strict() is False
        runtime_config.set_app_setting("embedding_dimension", "768")
        assert runtime_config.embedding_dimension() == 768

    def test_is_configured_flag(self):
        assert runtime_config.is_configured() is False
        runtime_config.set_app_setting("setup_complete", "true")
        assert runtime_config.is_configured() is True

    def test_jwt_secret_persists_and_is_stable(self):
        s1 = runtime_config.jwt_secret()
        s2 = runtime_config.jwt_secret()
        assert s1 == s2 and len(s1) > 20
        assert "jwt_secret" in secrets_store.list_secret_keys()
