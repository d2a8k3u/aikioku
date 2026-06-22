"""Tests for the observability wiring (Prometheus /metrics + structured logging)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src import observability


def test_metrics_endpoint_returns_prometheus_payload():
    from src.main import app
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "aikioku_http_requests_total" in body
    assert "aikioku_embedding_deterministic_fallback_total" in body


def test_request_counter_increments():
    from src.main import app
    with TestClient(app) as client:
        before = _counter_value()
        client.get("/health")
        after = _counter_value()
    assert after >= before + 1


def _counter_value() -> float:
    total = 0.0
    for metric in observability.HTTP_REQUESTS.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                total += sample.value
    return total


def test_embedding_degraded_gauge_reflects_fallback_count():
    from src.llm import ollama_remote

    ollama_remote.DETERMINISTIC_FALLBACK_COUNT = 3
    try:
        observability.render_metrics()
        # Gauge should now read 3
        value = observability.EMBEDDING_DEGRADED._value.get()
        assert value == 3
    finally:
        ollama_remote.DETERMINISTIC_FALLBACK_COUNT = 0
        observability.render_metrics()


def test_configure_logging_is_idempotent():
    # Should not raise when called repeatedly.
    observability.configure_logging("INFO")
    observability.configure_logging("DEBUG")
