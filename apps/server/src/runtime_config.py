"""Effective runtime configuration.

Resolves config from layered sources so the few *dynamic* consumers (LLM/retriever
construction, CORS, auth gate) read DB-backed values without touching the many
static ``settings.<field>`` reads scattered across the codebase.

Resolution order (lowest to highest precedence):
    code default + OS env (``settings`` singleton)  <  ``app_settings`` table (plain)
Secrets resolve from ``secrets_store`` first, falling back to ``settings``/OS env.

These canonical ``app_settings`` helpers are the single source of truth; the
settings/setup API import them from here (one-directional: this module never
imports an api.* module, avoiding circular imports).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from src.config import settings
from src import secrets_store

_DB_TABLE = """
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
"""

# In-memory cache for app_settings. Invalidated on set_app_setting.
_settings_cache: dict[str, str] | None = None
_settings_cache_ts: float = 0.0
_settings_cache_path: str | None = None
_SETTINGS_CACHE_TTL = 5.0  # seconds


# --- app_settings table helpers -------------------------------------------------


def _ensure_table() -> None:
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(_DB_TABLE)


def load_app_settings() -> dict[str, str]:
    global _settings_cache, _settings_cache_ts, _settings_cache_path
    db_path = settings.sqlite_db_path
    if (
        _settings_cache is not None
        and _settings_cache_path == db_path
        and (time.monotonic() - _settings_cache_ts) < _SETTINGS_CACHE_TTL
    ):
        return _settings_cache
    _ensure_table()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT key, value FROM app_settings")
        _settings_cache = {row[0]: row[1] for row in cursor.fetchall()}
        _settings_cache_ts = time.monotonic()
        _settings_cache_path = db_path
        return _settings_cache


def get_app_setting(key: str) -> str | None:
    return load_app_settings().get(key)


def set_app_setting(key: str, value: str) -> None:
    global _settings_cache
    _ensure_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
    _settings_cache = None  # invalidate


# --- typed effective getters ----------------------------------------------------


def _str(key: str, default: str) -> str:
    val = get_app_setting(key)
    return val if val is not None else default


def _bool(key: str, default: bool) -> bool:
    val = get_app_setting(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


def _int(key: str, default: int) -> int:
    val = get_app_setting(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    val = get_app_setting(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def llm_provider() -> str:
    return _str("llm_provider", settings.llm_provider)


def llm_model() -> str:
    return _str("llm_model", settings.llm_model)


def ollama_base_url() -> str:
    return _str("ollama_base_url", settings.ollama_base_url)


def ollama_embedding_base_url() -> str:
    return _str("ollama_embedding_base_url", settings.ollama_embedding_base_url)


def ollama_embedding_model() -> str:
    return _str("ollama_embedding_model", settings.ollama_embedding_model)


def hf_embedding_model() -> str:
    return _str("hf_embedding_model", settings.hf_embedding_model)


def openai_base_url() -> str:
    return _str("openai_base_url", settings.openai_base_url)


def openai_embedding_model() -> str:
    return _str("openai_embedding_model", settings.openai_embedding_model)


def openrouter_base_url() -> str:
    return _str("openrouter_base_url", settings.openrouter_base_url)


def embedding_model() -> str:
    return _str("embedding_model", settings.embedding_model)


def embedding_provider() -> str:
    return _str("embedding_provider", settings.embedding_provider)


def embedding_dimension() -> int:
    return _int("embedding_dimension", settings.embedding_dimension)


def embedding_strict() -> bool:
    return _bool("embedding_strict", settings.embedding_strict)


def auto_extract() -> bool:
    return _bool("auto_extract", settings.auto_extract)


def auto_consolidation() -> bool:
    return _bool("auto_consolidation", settings.auto_consolidation)


def auto_entity_extraction() -> bool:
    return _bool("auto_entity_extraction", settings.auto_entity_extraction)


def auto_memory_extraction() -> bool:
    return _bool("auto_memory_extraction", settings.auto_memory_extraction)


def auto_title() -> bool:
    return _bool("auto_title", settings.auto_title)


def llm_daily_budget_usd() -> float:
    return _float("llm_daily_budget_usd", settings.llm_daily_budget_usd)


def llm_budget_warning_fraction() -> float:
    return _float("llm_budget_warning_fraction", settings.llm_budget_warning_fraction)


def auth_required() -> bool:
    return _bool("auth_required", settings.auth_required)


def cors_origins() -> list[str]:
    raw = _str("cors_origins", settings.cors_origins)
    return [o.strip() for o in raw.split(",") if o.strip()]


# --- secret getters (encrypted store, OS-env fallback) --------------------------


def _secret(key: str, fallback: str) -> str:
    val = secrets_store.get_secret(key)
    if val:
        return val
    return fallback


def ollama_api_key() -> str:
    return _secret("ollama_api_key", settings.ollama_api_key)


def openrouter_api_key() -> str:
    return _secret("openrouter_api_key", settings.openrouter_api_key)


def hf_api_key() -> str:
    return _secret("hf_api_key", settings.hf_api_key)


def openai_api_key() -> str:
    return _secret("openai_api_key", settings.openai_api_key)


def jwt_secret() -> str:
    """Persisted JWT signing secret. Generated + stored on first use so issued
    tokens survive restarts. OS-env ``JWT_SECRET`` still wins if explicitly set."""
    if settings.jwt_secret:
        return settings.jwt_secret
    existing = secrets_store.get_secret("jwt_secret")
    if existing:
        return existing
    import secrets as _secrets

    generated = _secrets.token_urlsafe(48)
    secrets_store.set_secret("jwt_secret", generated)
    return generated


# --- configured flag ------------------------------------------------------------


def is_configured() -> bool:
    return _bool("setup_complete", False)
