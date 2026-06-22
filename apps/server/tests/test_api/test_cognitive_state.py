"""Tests for Cognitive State API endpoints."""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI TestClient with a fresh EventBus and CognitiveStateTracker."""
    from src.main import app

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "events.db")
        from src.events import EventBus

        app.state.event_bus = EventBus(db_path=db_path)

        # Ensure no stale tracker
        if hasattr(app.state, "cognitive_tracker"):
            delattr(app.state, "cognitive_tracker")

        yield TestClient(app)


class TestGetState:
    def test_returns_state_and_intervention(self, client):
        response = client.get("/api/cognitive/state")
        assert response.status_code == 200
        data = response.json()
        assert "state" in data
        assert "intervention" in data
        assert data["state"] in {
            "flow",
            "thinking",
            "exploring",
            "frustrated",
            "idle",
        }
        assert data["intervention"] in {
            "background_only",
            "suggest",
            "full",
            "proactive",
            "defer",
        }

    def test_state_is_idle_with_no_signals(self, client):
        response = client.get("/api/cognitive/state")
        data = response.json()
        assert data["state"] == "idle"


class TestRecordSignal:
    def test_records_signal(self, client):
        response = client.post("/api/cognitive/state?signal_type=typing_speed&value=8.0")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "recorded"
        assert data["signal_type"] == "typing_speed"
        assert data["value"] == 8.0

    def test_signal_affects_state(self, client):
        # Record enough signals for FRUSTRATED classification. Each signal is
        # timestamped "now" by the tracker, which is fine for the window.
        for _ in range(30):
            client.post("/api/cognitive/state?signal_type=typing_speed&value=2.0")
            client.post("/api/cognitive/state?signal_type=deletion_rate&value=0.6")
            client.post("/api/cognitive/state?signal_type=switch_rate&value=0.5")

        response = client.get("/api/cognitive/state")
        data = response.json()
        # With high deletion + low typing, should be frustrated
        assert data["state"] == "frustrated"
        assert data["intervention"] == "proactive"

    def test_missing_params_returns_422(self, client):
        response = client.post("/api/cognitive/state")
        assert response.status_code == 422
