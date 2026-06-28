"""Git sync API endpoints for version control."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from src.storage.git_sync import GitSync

router = APIRouter(prefix="/api/sync/git", tags=["git-sync"])


def _get_git_sync(request: Request) -> GitSync:
    """Get or create a GitSync from app state."""
    gs = getattr(request.app.state, "git_sync", None)
    if gs is None:
        from src.config import settings

        gs = GitSync(settings.notes_dir)
        request.app.state.git_sync = gs
    return gs


@router.post("/commit")
async def git_commit(request: Request, message: str) -> dict[str, str | bool]:
    """Stage all changes and commit with the given message."""
    gs = _get_git_sync(request)
    try:
        committed = gs.commit(message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"committed": committed, "message": message}


@router.get("/history")
async def git_history(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Get commit history."""
    gs = _get_git_sync(request)
    try:
        history = gs.get_history(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return history


@router.get("/diff/{note_id}")
async def git_diff(request: Request, note_id: str) -> dict[str, str]:
    """Get the latest diff for a specific note file."""
    gs = _get_git_sync(request)
    try:
        diff = gs.get_diff(note_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"note_id": note_id, "diff": diff}
