"""Cognitive state API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.augmentation.cognitive_state import CognitiveStateTracker

router = APIRouter(prefix="/api/cognitive", tags=["cognitive"])


def _get_tracker(request: Request) -> CognitiveStateTracker:
    """Get or create a CognitiveStateTracker from app state."""
    tracker = getattr(request.app.state, "cognitive_tracker", None)
    if tracker is None:
        event_bus = request.app.state.event_bus
        tracker = CognitiveStateTracker(event_bus=event_bus)
        request.app.state.cognitive_tracker = tracker
    return tracker


@router.get("/state")
async def get_state(request: Request) -> dict:
    """Get the current cognitive state and recommended intervention."""
    tracker = _get_tracker(request)
    try:
        state = tracker.get_state()
        intervention = tracker.get_intervention_recommendation(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "state": state.value,
        "intervention": intervention.value,
    }


@router.post("/state")
async def record_signal(request: Request, signal_type: str, value: float) -> dict:
    """Record a behavioural signal for cognitive state tracking."""
    tracker = _get_tracker(request)
    try:
        tracker.record_signal(signal_type, value)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "recorded", "signal_type": signal_type, "value": value}
