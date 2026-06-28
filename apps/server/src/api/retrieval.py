"""Retrieval API endpoints using HybridFusion."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from src.retrieval.fusion import HybridFusion
from src.retrieval.search_result import SearchResult

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


# ------------------------------------------------------------------ request models


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=20, ge=1, le=100)


# ------------------------------------------------------------------ helpers (patchable in tests)


async def _hybrid_search(request: Request, query: str, limit: int) -> list[dict[str, Any]]:
    """Run a hybrid search across dense, sparse, and graph retrievers.

    Uses the HybridFusion instance attached to the application state.
    """
    fusion: HybridFusion = request.app.state.hybrid_fusion
    results: list[SearchResult] = await fusion.search(query, limit=limit)
    return [
        {
            "note_id": r.note_id,
            "score": r.score,
            "source": r.source,
            "snippet": r.snippet,
            "metadata": r.metadata,
        }
        for r in results
    ]


# ------------------------------------------------------------------ endpoints


@router.post("/hybrid")
async def hybrid_search(request: Request, body: HybridSearchRequest) -> list[dict[str, Any]]:
    """Hybrid search across dense, sparse, and graph retrievers.

    Uses weighted reciprocal rank fusion (RRF) to combine results.
    """
    return await _hybrid_search(request, body.query, body.limit)
