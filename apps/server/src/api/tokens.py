"""Personal access token (PAT) management API (authenticated, owner-only).

Tokens authenticate machine clients (MCP, external apps) against the ``/mcp``
surface. They are created here by the logged-in owner, stored one-way hashed by
``access_tokens``, and shown in plaintext exactly once — in the POST response.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src import access_tokens
from src.auth import UserInDB, require_auth

router = APIRouter(prefix="/api/settings/tokens", tags=["tokens"])


class TokenCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    scope: str = Field(default="full")


def _public(token: access_tokens.AccessToken) -> dict[str, str | None]:
    return {
        "id": token.id,
        "name": token.name,
        "scope": token.scope,
        "prefix": token.prefix,
        "created": token.created,
        "last_used": token.last_used,
    }


@router.get("/")
async def list_tokens(_user: UserInDB = Depends(require_auth)) -> list[dict[str, str | None]]:
    """List access tokens (never returns the secret value)."""
    return [_public(t) for t in access_tokens.list_tokens()]


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_token(
    body: TokenCreate,
    user: UserInDB = Depends(require_auth),
) -> dict[str, str | None]:
    """Create a token. The plaintext ``token`` is returned only in this response."""
    if body.scope not in access_tokens.VALID_SCOPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scope: {body.scope!r}",
        )
    record, plaintext = access_tokens.create_token(
        name=body.name, username=user.username, scope=body.scope
    )
    return {**_public(record), "token": plaintext}


@router.delete("/{token_id}")
async def delete_token(
    token_id: str,
    _user: UserInDB = Depends(require_auth),
) -> dict[str, bool]:
    """Revoke a token by id."""
    removed = access_tokens.delete_token(token_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found")
    return {"ok": True}
