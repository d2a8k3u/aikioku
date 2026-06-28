"""Serendipity API endpoints for random walks and surprise scoring."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from src.augmentation.serendipity import SerendipityEngine

router = APIRouter(prefix="/api/serendipity", tags=["serendipity"])


def _get_engine(request: Request) -> SerendipityEngine:
    """Get or create a SerendipityEngine from app state."""
    engine = getattr(request.app.state, "serendipity_engine", None)
    if engine is None:
        graph = request.app.state.knowledge_graph
        engine = SerendipityEngine(graph)
        request.app.state.serendipity_engine = engine
    return engine


@router.post("/walk")
async def random_walk(
    request: Request,
    start_entity_id: str,
    steps: int = Query(5, ge=1, le=50),
) -> dict[str, Any]:
    """Perform a random walk from a starting entity."""
    engine = _get_engine(request)
    try:
        path = engine.random_walk(start_entity_id, steps=steps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"path": path}


@router.get("/surprise")
async def surprise_score(
    request: Request,
    entity_id: str | None = Query(None, min_length=1),
) -> dict[str, Any]:
    """Compute a surprise score for an entity.

    If entity_id is omitted, returns a random surprise from the graph.
    """
    engine = _get_engine(request)
    try:
        if entity_id is not None:
            score = engine.surprise_score(entity_id)
        else:
            entity_id, score = engine.random_surprise()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"entity_id": entity_id, "score": score}
