"""Entities API endpoints (flat /api/entities path)."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query, Request

from src.knowledge.graph import KnowledgeGraph

router = APIRouter(prefix="/api/entities", tags=["entities"])


def _get_graph(request: Request) -> KnowledgeGraph:
    return cast(KnowledgeGraph, request.app.state.knowledge_graph)


@router.get("/")
async def list_entities(
    request: Request,
    type: str | None = None,
    search: str | None = None,
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, Any]]:
    graph = _get_graph(request)
    entities = graph.find_entities(type=type, name=search, limit=limit)
    return [
        {
            "id": e.id,
            "name": e.name,
            "type": e.type.value,
            "aliases": e.aliases,
            "properties": e.properties,
            "confidence": e.confidence,
            "source_note_ids": e.source_note_ids,
        }
        for e in entities
    ]


@router.get("/{entity_id}")
async def get_entity(request: Request, entity_id: str) -> dict[str, Any]:
    graph = _get_graph(request)
    entity = graph.get_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type.value,
        "aliases": entity.aliases,
        "properties": entity.properties,
        "confidence": entity.confidence,
        "source_note_ids": entity.source_note_ids,
    }


@router.get("/{entity_id}/subgraph")
async def get_subgraph(
    request: Request,
    entity_id: str,
    depth: int = Query(2, ge=1, le=5),
) -> dict[str, Any]:
    graph = _get_graph(request)
    entity = graph.get_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    # BFS neighborhood collection
    visited: set[str] = {entity_id}
    frontier: list[str] = [entity_id]
    nodes: dict[str, dict[str, Any]] = {
        entity_id: {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type.value,
            "aliases": entity.aliases,
            "properties": entity.properties,
            "confidence": entity.confidence,
            "source_note_ids": entity.source_note_ids,
        }
    }
    edges: list[dict[str, Any]] = []

    for _ in range(depth):
        next_frontier: list[str] = []
        for current_id in frontier:
            relations = graph.get_relations(current_id)
            for rel in relations:
                other_id = (
                    rel.target_entity_id
                    if rel.source_entity_id == current_id
                    else rel.source_entity_id
                )
                if other_id not in nodes:
                    other = graph.get_entity(other_id)
                    if other:
                        nodes[other_id] = {
                            "id": other.id,
                            "name": other.name,
                            "type": other.type.value,
                            "aliases": other.aliases,
                            "properties": other.properties,
                            "confidence": other.confidence,
                            "source_note_ids": other.source_note_ids,
                        }
                if rel.id not in {e["id"] for e in edges}:
                    edges.append(
                        {
                            "id": rel.id,
                            "source_entity_id": rel.source_entity_id,
                            "target_entity_id": rel.target_entity_id,
                            "type": rel.type.value,
                            "confidence": rel.confidence,
                            "properties": rel.properties,
                        }
                    )
                if other_id not in visited:
                    visited.add(other_id)
                    next_frontier.append(other_id)
        frontier = next_frontier
        if not frontier:
            break

    return {
        "root_id": entity_id,
        "depth": depth,
        "nodes": list(nodes.values()),
        "edges": edges,
    }
