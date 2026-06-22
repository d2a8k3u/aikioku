"""First-run setup wizard API (unauthenticated until configured).

Replaces the old ``.env`` bootstrap: the client posts provider/embedding/storage
settings, API-key secrets, and an admin account; this persists them (config to the
``app_settings`` table, secrets Fernet-encrypted in ``app_secrets``), creates the
login account, builds the LLM runtime, and flips the ``setup_complete`` flag.

These endpoints are intentionally unauthenticated so they're reachable before the
app is configured. ``POST /api/setup`` returns 409 once configured, so it can't be
used to overwrite an existing install; further changes go through the authenticated
``/api/settings`` API. Compose binds the backend to 127.0.0.1 only.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from src import runtime_config, secrets_store
from src.auth import register_user
from src.llm.urls import OLLAMA_TAGS, OPENROUTER_MODELS, join

router = APIRouter(prefix="/api/setup", tags=["setup"])

# Secret field name -> handled via secrets_store (encrypted), never app_settings.
_SECRET_FIELDS = (
    "ollama_api_key",
    "openrouter_api_key",
    "hf_api_key",
    "openai_api_key",
)
# Non-secret config fields persisted to app_settings.
_CONFIG_FIELDS = (
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
    "auto_extract",
    "auto_consolidation",
    "llm_daily_budget_usd",
    "cors_origins",
)


class Account(BaseModel):
    username: str
    password: str
    email: str | None = None


class SetupPayload(BaseModel):
    # Non-secret config (all optional; defaults apply when omitted)
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
    # Secrets (encrypted)
    ollama_api_key: str | None = None
    openrouter_api_key: str | None = None
    hf_api_key: str | None = None
    openai_api_key: str | None = None
    # Admin account (required: the wizard creates the login)
    account: Account


class TestPayload(BaseModel):
    llm_provider: str = "ollama"
    ollama_base_url: str | None = None
    ollama_api_key: str | None = None
    openrouter_base_url: str | None = None
    openrouter_api_key: str | None = None


def _to_str(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


@router.get("/status")
async def setup_status() -> dict:
    """Report first-run setup state (unauthenticated; used by the frontend gate)."""
    return {
        "configured": runtime_config.is_configured(),
        "auth_required": runtime_config.auth_required(),
    }


@router.post("", status_code=status.HTTP_201_CREATED)
@router.post("/", status_code=status.HTTP_201_CREATED)
async def run_setup(payload: SetupPayload, request: Request) -> dict:
    """Persist first-run configuration, create the admin account, and go live."""
    if runtime_config.is_configured():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Application is already configured.",
        )

    data = payload.model_dump(exclude_unset=True)

    # 1. Persist non-secret config to app_settings.
    for field in _CONFIG_FIELDS:
        if field in data and data[field] is not None:
            runtime_config.set_app_setting(field, _to_str(data[field]))

    # 2. Persist secrets (encrypted). Empty strings are ignored by set_secret.
    for field in _SECRET_FIELDS:
        if field in data and data[field]:
            secrets_store.set_secret(field, data[field])

    # 3. Create the admin login account.
    register_user(
        username=payload.account.username,
        email=payload.account.email or payload.account.username,
        password=payload.account.password,
    )

    # 4. Persist the JWT signing secret (so tokens survive restarts) and enable
    #    the auth wall now that an account exists.
    runtime_config.jwt_secret()
    runtime_config.set_app_setting("auth_required", "true")

    # 5. Mark configured and build the live runtime (no restart needed).
    runtime_config.set_app_setting("setup_complete", "true")
    from src.main import build_runtime

    build_runtime(request.app)
    request.app.state.configured = True

    # Establish the embedding fingerprint baseline and embed any pre-setup
    # knowledge base (imported before the provider existed) in the background.
    from src.knowledge.reembed import schedule_reembed
    from src.knowledge.reembed_fingerprint import effective_embedding_fingerprint

    fp = effective_embedding_fingerprint()
    runtime_config.set_app_setting("embedding_active_fingerprint", fp)
    schedule_reembed(request.app, fp)

    return {"configured": True}


@router.post("/test")
async def test_connection(payload: TestPayload) -> dict:
    """Best-effort reachability check for the chosen LLM provider."""
    if payload.llm_provider == "openrouter":
        headers = {}
        if payload.openrouter_api_key:
            headers["Authorization"] = f"Bearer {payload.openrouter_api_key}"
        url = join(
            payload.openrouter_base_url or "https://openrouter.ai/api/v1",
            OPENROUTER_MODELS,
            dialect="openrouter",
        )
    else:
        if not (payload.ollama_base_url or "").strip():
            return {"ok": False, "detail": "No base URL provided."}
        headers = {}
        if payload.ollama_api_key:
            headers["Authorization"] = f"Bearer {payload.ollama_api_key}"
        url = join(payload.ollama_base_url, OLLAMA_TAGS, dialect="ollama")
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, headers=headers)
        ok = resp.status_code < 500
        return {"ok": ok, "detail": f"HTTP {resp.status_code}"}
    except httpx.HTTPError as exc:
        return {"ok": False, "detail": str(exc)}
