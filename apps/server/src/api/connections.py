"""Connection discovery API endpoint for the knowledge graph."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.reasoning.connections import ConnectionDiscovery

router = APIRouter(prefix="/api/connections", tags=["connections"])


def _get_connection_discovery(request: Request) -> ConnectionDiscovery:
    """Get or create a ConnectionDiscovery from app state."""
    cd = getattr(request.app.state, "connection_discovery", None)
    if cd is None:
        graph = request.app.state.knowledge_graph
        embedding_store = request.app.state.embedding_store
        cd = ConnectionDiscovery(graph, embedding_store)
        request.app.state.connection_discovery = cd
    return cd


@router.post("/discover")
async def discover_connections(
    request: Request,
    entity_id: str | None = None,
    max_distance: int = 3,
) -> dict[str, Any]:
    """Discover indirect connections from an entity via graph traversal + embedding similarity."""
    if entity_id is None:
        raise HTTPException(status_code=400, detail="entity_id query parameter is required")
    cd = _get_connection_discovery(request)
    try:
        connections = cd.discover_connections(entity_id, max_distance=max_distance)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "entity_id": entity_id,
        "connections": [
            {
                "path": c.path,
                "strength": c.strength,
                "explanation": c.explanation,
            }
            for c in connections
        ],
    }
