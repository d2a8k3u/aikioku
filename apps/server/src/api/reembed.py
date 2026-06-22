"""Reembed job status endpoint.

Reports progress of the background reembed run (triggered when the effective
embedding configuration changes). Mirrors the ``admin.py`` reextract status
convention. Polled by the frontend for the initial banner state; live updates
arrive over ``/ws/events``.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/reembed/status")
async def reembed_status() -> dict:
    """Return progress of the most recent / in-flight reembed run."""
    from src.knowledge.reembed import get_status

    return get_status()
