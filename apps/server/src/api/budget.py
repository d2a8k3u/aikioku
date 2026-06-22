"""Daily LLM budget status endpoint.

Exposes the current budget state (``active`` | ``warning`` | ``paused``), today's
spend, and the deferred-work queue depth so the dashboard can show the budget
banner and the time until the UTC-midnight reset. Live transitions are pushed
separately as ``budget.status`` WebSocket events (see ``processing.budget_status``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from src.auth import UserInDB, require_auth
from src.processing.budget_status import compute_status

router = APIRouter(prefix="/api/budget", tags=["budget"])


@router.get("/status")
async def budget_status(
    request: Request, _user: UserInDB = Depends(require_auth)
) -> dict[str, Any]:
    """Return the current daily-budget state, spend figures, and queue depth."""
    return compute_status(request.app)
