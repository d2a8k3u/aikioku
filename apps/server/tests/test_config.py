"""Tests for application configuration."""

from __future__ import annotations

import warnings


class TestConfigStructure:
    """Ensure Settings uses modern Pydantic v2 ConfigDict, not deprecated class Config."""

    def test_settings_uses_configdict_not_inner_config_class(self):
        from src.config import Settings

        assert hasattr(Settings, "model_config")
        assert not hasattr(Settings, "Config")
        assert isinstance(Settings.model_config, dict)
        assert "env_file" in Settings.model_config
        assert "env_prefix" in Settings.model_config

    def test_settings_instantiation_emits_no_pydantic_deprecation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from src.config import Settings  # noqa: F401

            pydantic_warnings = [
                warning for warning in w if "PydanticDeprecatedSince20" in str(warning.category)
            ]
            assert len(pydantic_warnings) == 0, "Deprecated class Config triggers warning"


class TestConfigDefaults:
    """Verify default values and field coverage."""

    def test_default_llm_provider_is_string(self):
        from src.config import settings

        assert isinstance(settings.llm_provider, str)
        assert settings.llm_provider != ""

    def test_default_llm_model(self):
        from src.config import settings

        assert settings.llm_model == "kimi-k2.6:cloud"

    def test_default_sqlite_db_path(self):
        from src.config import Settings

        s = Settings()
        assert s.sqlite_db_path == "/data/sqlite/aikioku.db"

    def test_default_backend_port(self):
        from src.config import settings

        assert settings.backend_port == 8000

    def test_all_docker_compose_env_vars_have_field(self):
        """Every env var set in docker-compose (except obsolete ones) must map to a Settings field."""
        from src.config import Settings

        env_vars = [
            "LLM_PROVIDER",
            "OLLAMA_BASE_URL",
            "NOTES_DIR",
            "SQLITE_DB_PATH",
            "JWT_SECRET",
            "CORS_ORIGINS",
            "AUTH_REQUIRED",
        ]
        fields = set(Settings.model_fields.keys())
        for var in env_vars:
            field_name = var.lower()
            assert field_name in fields, f"Missing Settings field for env var {var}"


class TestSecuritySettings:
    """LOCAL-TRUST security configuration defaults."""

    # These assert the CODE defaults via the model field defaults, which is
    # env-independent — the running container sets AUTH_REQUIRED/JWT_SECRET/
    # CORS_ORIGINS, so constructing Settings() here would read the env, not the
    # default we intend to verify.
    def test_auth_required_defaults_false(self):
        """Local-trust default: optional auth (no login wall)."""
        from src.config import Settings

        assert Settings.model_fields["auth_required"].default is False

    def test_jwt_secret_defaults_empty(self):
        """Empty default means a random per-process secret is generated at import."""
        from src.config import Settings

        assert Settings.model_fields["jwt_secret"].default == ""

    def test_cors_origins_default_parses_to_localhost_allowlist(self):
        """Default CORS origins is a comma-separated localhost allowlist (never wildcard)."""
        from src.config import Settings

        default = Settings.model_fields["cors_origins"].default
        origins = [o.strip() for o in default.split(",") if o.strip()]
        assert origins == [
            "http://localhost:3369",
            "http://localhost:3000",
            "http://127.0.0.1:3369",
        ]
        assert "*" not in origins
