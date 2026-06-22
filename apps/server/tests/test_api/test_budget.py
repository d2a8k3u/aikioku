"""Tests for the daily-budget status API and live budget application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.main import app

    yield TestClient(app)


class TestBudgetStatus:
    def test_status_shape_defaults_active(self, client):
        resp = client.get("/api/budget/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] in {"active", "warning", "paused"}
        for key in (
            "daily_budget",
            "today_cost",
            "remaining",
            "fraction",
            "pending_count",
            "warning_fraction",
            "reset_at",
        ):
            assert key in data
        assert data["pending_count"] == 0

    def test_status_reflects_budget_update(self, client):
        client.put("/api/settings", json={"llm_daily_budget_usd": 12.5})
        resp = client.get("/api/budget/status")
        assert resp.status_code == 200
        assert resp.json()["daily_budget"] == 12.5
