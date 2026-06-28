"""Anomaly detection API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from src.reasoning.anomaly import AnomalyDetector

router = APIRouter(prefix="/api/anomaly", tags=["anomaly"])


def _get_detector(request: Request) -> AnomalyDetector:
    detector = getattr(request.app.state, "anomaly_detector", None)
    if detector is None:
        from src.config import settings

        detector = AnomalyDetector(settings.sqlite_db_path)
        request.app.state.anomaly_detector = detector
    return detector


@router.post("/scan")
async def scan_anomalies(request: Request) -> dict[str, Any]:
    """Run all anomaly detection checks and return results."""
    detector = _get_detector(request)
    graph = request.app.state.knowledge_graph
    from src.config import settings

    try:
        results = detector.run_all(graph, db_path=settings.sqlite_db_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "count": len(results),
        "anomalies": [
            {
                "type": r.type,
                "severity": r.severity,
                "description": r.description,
                "entity_id": r.entity_id,
                "note_id": r.note_id,
                "metric_value": r.metric_value,
                "threshold": r.threshold,
            }
            for r in results
        ],
    }


@router.get("/recent")
async def recent_anomalies(
    request: Request,
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Return recent unresolved anomalies."""
    detector = _get_detector(request)
    return detector.get_recent(hours=hours, limit=limit)


@router.post("/{anomaly_id}/resolve")
async def resolve_anomaly(request: Request, anomaly_id: str) -> dict[str, str]:
    """Mark an anomaly as resolved."""
    detector = _get_detector(request)
    detector.resolve(anomaly_id)
    return {"status": "resolved", "id": anomaly_id}
