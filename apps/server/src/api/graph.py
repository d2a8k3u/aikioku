"""Graph API endpoints for entity-relation knowledge graph."""

from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query, Request

from src.knowledge.graph import KnowledgeGraph
from src.models.entity import Entity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


def _get_graph(request: Request) -> KnowledgeGraph:
    """Get KnowledgeGraph from app state."""
    return cast(KnowledgeGraph, request.app.state.knowledge_graph)


def _entity_to_dict(entity: Entity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type.value,
        "aliases": entity.aliases,
        "properties": entity.properties,
        "confidence": entity.confidence,
        "source_note_ids": entity.source_note_ids,
    }


@router.get("/entities")
async def list_entities(
    request: Request,
    type: str | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    graph = _get_graph(request)
    entities = graph.find_entities(type=type, limit=limit)
    return [_entity_to_dict(e) for e in entities]


@router.get("/entities/{entity_id}")
async def get_entity(request: Request, entity_id: str) -> dict[str, Any]:
    graph = _get_graph(request)
    entity = graph.get_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    result = _entity_to_dict(entity)
    relations = graph.get_relations(entity_id)
    result["relations"] = [
        {
            "id": r.id,
            "source_entity_id": r.source_entity_id,
            "target_entity_id": r.target_entity_id,
            "type": r.type.value,
            "confidence": r.confidence,
            "properties": r.properties,
        }
        for r in relations
    ]
    return result


@router.get("/entities/{entity_id}/relations")
async def get_entity_relations(request: Request, entity_id: str) -> list[dict[str, Any]]:
    graph = _get_graph(request)
    entity = graph.get_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    relations = graph.get_relations(entity_id)
    return [
        {
            "id": r.id,
            "source_entity_id": r.source_entity_id,
            "target_entity_id": r.target_entity_id,
            "type": r.type.value,
            "confidence": r.confidence,
            "properties": r.properties,
        }
        for r in relations
    ]


@router.get("/relations")
async def list_relations(
    request: Request,
    limit: int = Query(500, ge=1, le=5000),
) -> list[dict[str, Any]]:
    graph = _get_graph(request)
    relations = graph.get_all_relations(limit=limit)
    return [
        {
            "id": r.id,
            "source_entity_id": r.source_entity_id,
            "target_entity_id": r.target_entity_id,
            "type": r.type.value,
            "confidence": r.confidence,
        }
        for r in relations
    ]


@router.get("/full")
async def graph_full(
    request: Request,
    limit: int = Query(2000, ge=1, le=10000),
    relation_limit: int = Query(10000, ge=1, le=50000),
) -> dict[str, Any]:
    """Return the whole graph (nodes + edges) in one request for visualization."""
    graph = _get_graph(request)
    entities = graph.find_entities(limit=limit)
    relations = graph.get_all_relations(limit=relation_limit)
    return {
        "nodes": [_entity_to_dict(e) for e in entities],
        "edges": [
            {
                "source_entity_id": r.source_entity_id,
                "target_entity_id": r.target_entity_id,
                "type": r.type.value,
                "confidence": r.confidence,
            }
            for r in relations
        ],
    }


@router.get("/paths")
async def find_paths(
    request: Request,
    source: str,
    target: str,
    max_depth: int = Query(3, ge=1, le=10),
) -> dict[str, Any]:
    graph = _get_graph(request)
    paths = graph.find_paths(source, target, max_depth)
    return {"paths": [[_entity_to_dict(e) for e in path] for path in paths]}


@router.get("/stats")
async def graph_stats(request: Request) -> dict[str, Any]:
    graph = _get_graph(request)
    return {
        "entities": graph.count_entities(),
        "relations": graph.count_relations(),
        "types": graph.get_entity_types(),
    }
