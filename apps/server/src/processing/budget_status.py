"""Budget status snapshot + change-triggered broadcast for the daily LLM budget.

``compute_status`` is the single source of the status shape used by the
``/api/budget/status`` endpoint and the ``budget.status`` WebSocket event.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI

from src import runtime_config
from src.processing import pending_work

logger = logging.getLogger(__name__)


def _next_utc_midnight_iso() -> str:
    """ISO timestamp of the next UTC midnight — when today's spend window resets."""
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    reset = datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc)
    return reset.isoformat()


def compute_status(app: FastAPI) -> dict[str, Any]:
    """Current budget state: ``active`` | ``warning`` | ``paused`` plus figures."""
    tracker = getattr(app.state, "cost_tracker", None)
    warn = runtime_config.llm_budget_warning_fraction()
    pending = pending_work.count()
    if tracker is None:
        budget = runtime_config.llm_daily_budget_usd()
        return {
            "state": "active",
            "daily_budget": budget,
            "today_cost": 0.0,
            "remaining": budget,
            "fraction": 0.0,
            "pending_count": pending,
            "warning_fraction": warn,
            "reset_at": _next_utc_midnight_iso(),
        }
    if tracker.is_exhausted():
        state = "paused"
    elif tracker.is_warning(warn):
        state = "warning"
    else:
        state = "active"
    return {
        "state": state,
        "daily_budget": tracker.daily_budget,
        "today_cost": tracker.get_today_cost(),
        "remaining": tracker.remaining(),
        "fraction": tracker.fraction(),
        "pending_count": pending,
        "warning_fraction": warn,
        "reset_at": _next_utc_midnight_iso(),
    }


async def broadcast_budget_status(app: FastAPI, *, force: bool = False) -> None:
    """Broadcast a ``budget.status`` WS event when the state changes (or forced).

    Best-effort: a status/broadcast failure is logged, never raised, so it can't
    break the ingestion path that triggered it.
    """
    try:
        status = compute_status(app)
        last = getattr(app.state, "_last_budget_state", None)
        if not force and last == status["state"]:
            return
        app.state._last_budget_state = status["state"]
        from src.api.websocket import get_broadcaster

        broadcaster = get_broadcaster()
        if broadcaster is not None:
            await broadcaster.broadcast("budget.status", status)
    except Exception:
        logger.debug("budget status broadcast failed", exc_info=True)
