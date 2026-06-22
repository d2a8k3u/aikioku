"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "kimi-k2.6:cloud"
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str = ""
    ollama_embedding_base_url: str = "http://host.docker.internal:11434"
    ollama_embedding_model: str = "mxbai-embed-large"
    # OpenRouter (OpenAI-compatible chat proxy; no embeddings endpoint).
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""

    # Storage
    notes_dir: str = "/data/notes"
    sqlite_db_path: str = "/data/sqlite/aikioku.db"
    # Master key file for the encrypted secrets table. Lives OUTSIDE the DB
    # (bootstrap: can't decrypt the DB from inside it). Auto-generated on first
    # run. Empty -> derived next to sqlite_db_path (".../secret.key").
    secret_key_file: str = ""

    # Embedding
    embedding_provider: str = "ollama"  # ollama | huggingface | openai
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1024
    embedding_strict: bool = True
    hf_api_key: str = ""
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # OpenAI embeddings (OpenAI-compatible; base normalized to /v1)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com"
    openai_embedding_model: str = "text-embedding-3-small"

    # Automation
    auto_extract: bool = True
    auto_consolidation: bool = True
    auto_title: bool = True
    llm_daily_budget_usd: float = 5.0
    # Fraction of the daily budget at which the UI flags a near-limit warning
    # (still processing). At 1.0 the budget is exhausted and processing pauses.
    llm_budget_warning_fraction: float = 0.9

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # Security (LOCAL-TRUST defaults)
    # Empty jwt_secret -> a random per-process secret is generated at import
    # (see auth.py); set JWT_SECRET to persist tokens across restarts.
    jwt_secret: str = ""
    # Comma-separated CORS allowlist (never a wildcard with credentials).
    cors_origins: str = "http://localhost:3369,http://localhost:3000,http://127.0.0.1:3369"
    # Local-trust default: optional auth (no login wall). When True, endpoints
    # using require_auth demand a valid token.
    auth_required: bool = False

    # Logging
    log_level: str = "INFO"

    # No .env file: secrets live in the encrypted DB (see secrets_store.py) and
    # runtime config in the DB (see runtime_config.py), set via the setup wizard.
    # BaseSettings still reads real OS env vars, so bootstrap paths
    # (sqlite_db_path, notes_dir) remain overridable by the container.
    model_config = SettingsConfigDict(
        env_file=None,
        env_prefix="",
    )


settings = Settings()
