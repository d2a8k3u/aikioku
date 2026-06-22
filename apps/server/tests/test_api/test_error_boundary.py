"""Tests for typed exception handlers at the API boundary (src/main.py).

The global handlers must map upstream LLM/network failures to the right status
codes (503/502) instead of an opaque 500, while still letting FastAPI's
HTTPException and the generic fallback work.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.llm.json_parse import LLMOutputParseError
from src.llm.ollama_remote import EmbeddingUnavailableError

# Routes are registered lazily inside the fixture so the autouse
# reset_app_state temp-DB patch is active before src.main is imported.
_ROUTES_REGISTERED = False


def _register_boundary_routes(app) -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return

    @app.get("/_test_boundary/connect-error")
    async def _connect_error():
        raise httpx.ConnectError("connection refused")

    @app.get("/_test_boundary/timeout")
    async def _timeout():
        raise httpx.TimeoutException("read timed out")

    @app.get("/_test_boundary/http-error")
    async def _http_error():
        raise httpx.HTTPError("generic http error")

    @app.get("/_test_boundary/embedding-unavailable")
    async def _embedding_unavailable():
        raise EmbeddingUnavailableError("no embedding")

    @app.get("/_test_boundary/parse-error")
    async def _parse_error():
        raise LLMOutputParseError("bad json")

    @app.get("/_test_boundary/generic")
    async def _generic():
        raise RuntimeError("unexpected")

    @app.get("/_test_boundary/http-exception")
    async def _http_exception():
        raise HTTPException(status_code=404, detail="not found")

    _ROUTES_REGISTERED = True


@pytest.fixture
def boundary_client():
    """A TestClient that lets the app's handlers (not the harness) handle errors.

    Function-scoped so the autouse ``reset_app_state`` temp-DB patch is active
    while the lifespan runs (avoids Kuzu lock contention with the live backend).
    """
    from src.main import app

    _register_boundary_routes(app)

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestErrorBoundary:
    def test_connect_error_returns_503(self, boundary_client):
        resp = boundary_client.get("/_test_boundary/connect-error")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Upstream language model unavailable."

    def test_timeout_returns_503(self, boundary_client):
        resp = boundary_client.get("/_test_boundary/timeout")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Upstream language model unavailable."

    def test_http_error_returns_503(self, boundary_client):
        resp = boundary_client.get("/_test_boundary/http-error")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Upstream language model unavailable."

    def test_embedding_unavailable_returns_503(self, boundary_client):
        resp = boundary_client.get("/_test_boundary/embedding-unavailable")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Embedding model unavailable."

    def test_parse_error_returns_502(self, boundary_client):
        resp = boundary_client.get("/_test_boundary/parse-error")
        assert resp.status_code == 502
        assert resp.json()["detail"] == "The language model returned invalid output."

    def test_generic_exception_returns_500(self, boundary_client):
        resp = boundary_client.get("/_test_boundary/generic")
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"

    def test_http_exception_404_preserved(self, boundary_client):
        resp = boundary_client.get("/_test_boundary/http-exception")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "not found"
