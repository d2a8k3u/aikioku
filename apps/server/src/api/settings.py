"""Application settings API (authenticated; post-setup editing).

Reads/writes the DB-backed effective config via ``runtime_config`` and manages
encrypted secrets via ``secrets_store``. Provider-affecting changes hot-reload the
LLM runtime through ``build_runtime`` — no restart. Secret values are never
returned; only their key names are listed.
"""

import logging

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from src import runtime_config, secrets_store
from src.auth import UserInDB, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Changing any of these requires rebuilding the LLM provider / retrievers.
_PROVIDER_KEYS = {
    "llm_provider",
    "llm_model",
    "ollama_base_url",
    "ollama_embedding_base_url",
    "ollama_embedding_model",
    "openrouter_base_url",
    "embedding_provider",
    "embedding_model",
    "embedding_dimension",
    "embedding_strict",
    "hf_embedding_model",
    "openai_base_url",
    "openai_embedding_model",
}
_SECRET_KEYS = {
    "ollama_api_key",
    "openrouter_api_key",
    "hf_api_key",
    "openai_api_key",
}


class SettingsUpdate(BaseModel):
    llm_provider: str | None = None
    llm_model: str | None = None
    ollama_base_url: str | None = None
    ollama_embedding_base_url: str | None = None
    ollama_embedding_model: str | None = None
    openrouter_base_url: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    embedding_strict: bool | None = None
    hf_embedding_model: str | None = None
    openai_base_url: str | None = None
    openai_embedding_model: str | None = None
    auto_extract: bool | None = None
    auto_consolidation: bool | None = None
    llm_daily_budget_usd: float | None = None
    cors_origins: str | None = None


class SecretUpdate(BaseModel):
    key: str
    value: str


def _to_str(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _settings_to_dict() -> dict:
    """Return current effective settings (no secret values, only their names)."""
    return {
        "llm_provider": runtime_config.llm_provider(),
        "llm_model": runtime_config.llm_model(),
        "ollama_base_url": runtime_config.ollama_base_url(),
        "ollama_embedding_base_url": runtime_config.ollama_embedding_base_url(),
        "ollama_embedding_model": runtime_config.ollama_embedding_model(),
        "openrouter_base_url": runtime_config.openrouter_base_url(),
        "embedding_provider": runtime_config.embedding_provider(),
        "embedding_model": runtime_config.embedding_model(),
        "embedding_dimension": runtime_config.embedding_dimension(),
        "embedding_strict": runtime_config.embedding_strict(),
        "hf_embedding_model": runtime_config.hf_embedding_model(),
        "openai_base_url": runtime_config.openai_base_url(),
        "openai_embedding_model": runtime_config.openai_embedding_model(),
        "auto_extract": runtime_config.auto_extract(),
        "auto_consolidation": runtime_config.auto_consolidation(),
        "llm_daily_budget_usd": runtime_config.llm_daily_budget_usd(),
        "auth_required": runtime_config.auth_required(),
        "cors_origins": ",".join(runtime_config.cors_origins()),
        "secret_keys": secrets_store.list_secret_keys(),
    }


def _maybe_rebuild(request: Request) -> None:
    if getattr(request.app.state, "configured", False):
        import asyncio

        from src.main import build_runtime

        build_runtime(request.app)
        # The model/provider (or retrieval config) changed, so cached answers may
        # be from the old model. Flush the semantic cache as note writes do, else a
        # repeated question would return a stale answer until the cache TTL expires.
        from src.cache.semantic_cache import cache_invalidate

        asyncio.ensure_future(cache_invalidate())
        # If the change altered the effective embedding config, rebuild the
        # vector store in the background (atomic swap, no search blackout).
        from src.knowledge.reembed import maybe_schedule_reembed

        maybe_schedule_reembed(request.app)


async def _apply_budget_change(request: Request) -> None:
    """Apply a daily-budget change to the live cost tracker without a full runtime
    rebuild, then resume any deferred work the new budget now allows.

    A budget change is a single number — rebuilding the LLM provider + retrievers
    for it would be wasteful, so the value is pushed onto the existing tracker
    directly.
    """
    tracker = getattr(request.app.state, "cost_tracker", None)
    if tracker is None:
        return
    tracker.daily_budget = runtime_config.llm_daily_budget_usd()
    import asyncio

    from src.processing.budget_gate import drain
    from src.processing.budget_status import broadcast_budget_status

    await broadcast_budget_status(request.app, force=True)
    if not tracker.is_exhausted():
        asyncio.ensure_future(drain(request.app))


@router.get("/")
async def get_settings(_user: UserInDB = Depends(require_auth)) -> dict:
    """Return current application settings."""
    return _settings_to_dict()


@router.put("/")
async def update_settings(
    body: SettingsUpdate,
    request: Request,
    _user: UserInDB = Depends(require_auth),
) -> dict:
    """Update application settings and persist to the database."""
    update_data = body.model_dump(exclude_unset=True)
    rebuild = False
    for key, value in update_data.items():
        if value is None:
            continue
        runtime_config.set_app_setting(key, _to_str(value))
        if key in _PROVIDER_KEYS:
            rebuild = True
    if rebuild:
        _maybe_rebuild(request)
    if update_data.get("llm_daily_budget_usd") is not None:
        await _apply_budget_change(request)
    return _settings_to_dict()


@router.get("/secrets")
async def list_secrets(_user: UserInDB = Depends(require_auth)) -> dict:
    """Return the names of stored secrets (never the values)."""
    return {"keys": secrets_store.list_secret_keys()}


@router.put("/secrets")
async def set_secret(
    body: SecretUpdate,
    request: Request,
    _user: UserInDB = Depends(require_auth),
) -> dict:
    """Set or rotate an encrypted secret."""
    if body.key not in _SECRET_KEYS:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown secret key: {body.key}",
        )
    secrets_store.set_secret(body.key, body.value)
    _maybe_rebuild(request)
    return {"keys": secrets_store.list_secret_keys()}


@router.delete("/secrets/{key}")
async def delete_secret(
    key: str,
    request: Request,
    _user: UserInDB = Depends(require_auth),
) -> dict:
    """Remove an encrypted secret."""
    secrets_store.delete_secret(key)
    _maybe_rebuild(request)
    return {"keys": secrets_store.list_secret_keys()}


# ---------------------------------------------------------------------------
# BOD 4: Dynamic provider model listing
# ---------------------------------------------------------------------------


def _filter_models(models: list[dict], q: str | None) -> list[dict]:
    """Case-insensitive substring filter on model id/name (search-as-you-type)."""
    if not q:
        return models
    ql = q.lower()
    return [m for m in models if ql in m["id"].lower() or ql in (m.get("name") or "").lower()]


@router.get("/models")
async def list_models(
    provider: str = Query(..., description="ollama | ollama_remote | openrouter"),
    base_url: str | None = Query(
        None, description="Override the saved base URL (preview unsaved config)"
    ),
    api_key: str | None = Query(
        None, description="Override the saved API key (preview unsaved config)"
    ),
    q: str | None = Query(None, description="Case-insensitive substring filter on model id/name"),
    _user: UserInDB = Depends(require_auth),
) -> dict:
    """Return available chat/embedding models for the given provider.

    Fetches live from ``base_url``/``api_key`` when supplied, else the saved
    config. The override lets the settings form preview models for a URL/key
    the user has typed but not yet saved; ``q`` narrows the list server-side.
    Returns an empty list + error message on failure rather than a 500 so the
    frontend can surface it gracefully. The error reflects reachability of the
    provider, not an empty filter result.
    """
    from src.llm.model_list import (
        list_ollama_models,
        list_openrouter_models,
        classify_ollama_models,
    )

    if provider in ("ollama", "ollama_remote"):
        url = base_url or runtime_config.ollama_base_url()
        key = api_key or (runtime_config.ollama_api_key() if provider == "ollama_remote" else "")
        models = await list_ollama_models(url, key)
        tagged = _filter_models(classify_ollama_models(models), q)
        return {"models": tagged, "error": None if models else f"Could not reach Ollama at {url}"}

    if provider == "openrouter":
        url = base_url or runtime_config.openrouter_base_url()
        key = api_key or runtime_config.openrouter_api_key()
        models = await list_openrouter_models(url, key)
        # All OpenRouter models are chat models (no embeddings endpoint).
        tagged = _filter_models([{**m, "type": "chat"} for m in models], q)
        return {
            "models": tagged,
            "error": None if models else f"Could not reach OpenRouter at {url}",
        }

    return {"models": [], "error": f"Unknown provider: {provider}"}


# ---------------------------------------------------------------------------
# Embedding model listing — per provider, live where possible (mirrors /models)
# ---------------------------------------------------------------------------

_OLLAMA_KNOWN_EMBEDDING_MODELS: list[dict] = [
    {
        "id": "mxbai-embed-large",
        "name": "mxbai-embed-large",
        "provider": "ollama",
        "dimensions": 1024,
    },
    {"id": "nomic-embed-text", "name": "nomic-embed-text", "provider": "ollama", "dimensions": 768},
    {"id": "bge-m3", "name": "bge-m3", "provider": "ollama", "dimensions": 1024},
    {"id": "bge-large", "name": "bge-large", "provider": "ollama", "dimensions": 1024},
    {
        "id": "snowflake-arctic-embed:l",
        "name": "snowflake-arctic-embed:l",
        "provider": "ollama",
        "dimensions": 1024,
    },
    {
        "id": "snowflake-arctic-embed:m",
        "name": "snowflake-arctic-embed:m",
        "provider": "ollama",
        "dimensions": 768,
    },
    {"id": "all-minilm:l6-v2", "name": "all-minilm:l6-v2", "provider": "ollama", "dimensions": 384},
    {
        "id": "all-minilm:l12-v2",
        "name": "all-minilm:l12-v2",
        "provider": "ollama",
        "dimensions": 384,
    },
]

_OPENAI_EMBEDDING_MODELS: list[dict] = [
    {
        "id": "text-embedding-3-small",
        "name": "text-embedding-3-small",
        "provider": "openai",
        "dimensions": 1536,
    },
    {
        "id": "text-embedding-3-large",
        "name": "text-embedding-3-large",
        "provider": "openai",
        "dimensions": 3072,
    },
    {
        "id": "text-embedding-ada-002",
        "name": "text-embedding-ada-002",
        "provider": "openai",
        "dimensions": 1536,
    },
]


@router.get("/embedding-models")
async def list_embedding_models(
    provider: str = Query("ollama", description="ollama | huggingface | openai"),
    base_url: str | None = Query(
        None, description="Override the saved base URL (preview unsaved config)"
    ),
    api_key: str | None = Query(
        None, description="Override the saved API key (preview unsaved config)"
    ),
    q: str | None = Query(None, description="Case-insensitive substring filter on model id/name"),
    _user: UserInDB = Depends(require_auth),
) -> dict:
    """Return selectable embedding models for the given provider.

    - ``ollama`` → live ``/api/tags`` (embedding-family) merged with a curated
      known list, from ``base_url`` (override) else the saved embedding base URL.
    - ``huggingface`` → live public Hub search (sentence-similarity), narrowed by ``q``.
    - ``openai`` → curated list (text-embedding-3-small/large, ada-002).

    Errors reflect provider reachability, not an empty filter result.
    """
    from src.llm.model_list import (
        classify_ollama_models,
        list_hf_embedding_models,
        list_ollama_models,
    )

    if provider in ("ollama", "ollama_remote"):
        url = base_url or runtime_config.ollama_embedding_base_url()
        key = api_key or (runtime_config.ollama_api_key() if provider == "ollama_remote" else "")
        raw = await list_ollama_models(url, key)
        tagged = classify_ollama_models(raw)
        dynamic = [
            {"id": m["id"], "name": m["name"], "provider": "ollama", "dimensions": None}
            for m in tagged
            if m.get("type") == "embedding"
        ]
        dynamic_ids = {m["id"] for m in dynamic}
        merged = dynamic + [m for m in _OLLAMA_KNOWN_EMBEDDING_MODELS if m["id"] not in dynamic_ids]
        return {
            "models": _filter_models(merged, q),
            "error": None if raw else f"Could not reach Ollama at {url}",
        }

    if provider == "huggingface":
        models = await list_hf_embedding_models(q or "")
        return {"models": models, "error": None}

    if provider == "openai":
        return {"models": _filter_models(list(_OPENAI_EMBEDDING_MODELS), q), "error": None}

    return {"models": [], "error": f"Unknown provider: {provider}"}
