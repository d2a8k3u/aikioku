"""Schema API endpoints: type registry and LLM-driven schema induction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request

from src.knowledge.schema import SchemaRegistry
from src.knowledge.schema_induction import SchemaInducer

if TYPE_CHECKING:
    from src.knowledge.graph import KnowledgeGraph
    from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/schema", tags=["schema"])


def _get_registry(request: Request) -> SchemaRegistry:
    """Get or lazily create the SchemaRegistry singleton on app state."""
    reg = getattr(request.app.state, "schema_registry", None)
    if reg is None:
        reg = SchemaRegistry()
        request.app.state.schema_registry = reg
    return reg


def _get_llm(request: Request) -> LLMProvider:
    """Get the LLM provider from app state."""
    llm: LLMProvider | None = getattr(request.app.state, "llm_provider", None)
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM provider not configured. Complete setup first.",
        )
    return llm


def _get_graph(request: Request) -> KnowledgeGraph:
    """Get the KnowledgeGraph from app state."""
    graph: KnowledgeGraph = request.app.state.knowledge_graph
    return graph


@router.get("/types")
async def list_types(request: Request) -> dict[str, Any]:
    """Return all registered entity and relation types."""
    reg = _get_registry(request)
    return reg.list_types()


@router.post("/induce")
async def induce_schema(request: Request) -> dict[str, Any]:
    """Run LLM-driven schema induction to discover new entity/relation types.

    Analyzes existing graph data and returns suggestions for human review.
    Suggestions are NOT auto-applied — use a separate approval step to register them.
    """
    llm = _get_llm(request)
    graph = _get_graph(request)
    reg = _get_registry(request)

    inducer = SchemaInducer(
        llm_provider=llm,
        schema_registry=reg,
        knowledge_graph=graph,
    )

    try:
        result = await inducer.run_induction()
    except Exception as exc:
        logger.exception("Schema induction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Schema induction failed: {exc}",
        )

    return result
