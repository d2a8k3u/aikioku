"""Tests for lifespan background task and service initialization."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestLifespanServices:
    """Verify that lifespan initializes all required services."""

    def test_health_endpoint_available(self):
        from src.main import app

        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_app_state_has_event_bus(self):
        from src.main import app

        with TestClient(app) as client:
            assert hasattr(client.app.state, "event_bus")
            assert client.app.state.event_bus is not None

    def test_app_state_has_knowledge_graph(self):
        from src.main import app

        with TestClient(app) as client:
            assert hasattr(client.app.state, "knowledge_graph")
            assert client.app.state.knowledge_graph is not None

    def test_app_state_has_embedding_store(self):
        from src.main import app

        with TestClient(app) as client:
            assert hasattr(client.app.state, "embedding_store")
            assert client.app.state.embedding_store is not None

    def test_llm_provider_deferred_until_configured(self):
        # Fresh install is unconfigured: the secret-dependent LLM runtime is
        # deferred (None) until the setup wizard runs, then build_runtime wires
        # it up with no restart.
        from src import main, runtime_config
        from src.main import app

        with TestClient(app) as client:
            assert hasattr(client.app.state, "llm_provider")
            assert client.app.state.llm_provider is None

            runtime_config.set_app_setting("setup_complete", "true")
            main.build_runtime(client.app)
            assert client.app.state.llm_provider is not None

    def test_hybrid_fusion_deferred_until_configured(self):
        from src import main, runtime_config
        from src.main import app

        with TestClient(app) as client:
            assert hasattr(client.app.state, "hybrid_fusion")
            assert client.app.state.hybrid_fusion is None

            runtime_config.set_app_setting("setup_complete", "true")
            main.build_runtime(client.app)
            assert client.app.state.hybrid_fusion is not None


class TestCORSAllowlist:
    """CORS must reflect an explicit allowlist, never a wildcard."""

    def test_allowed_origin_is_reflected(self):
        from src.main import app

        with TestClient(app) as client:
            resp = client.get("/health", headers={"Origin": "http://localhost:3369"})
            allow_origin = resp.headers.get("access-control-allow-origin")
            assert allow_origin == "http://localhost:3369"
            assert allow_origin != "*"

    def test_disallowed_origin_is_not_reflected(self):
        from src.main import app

        with TestClient(app) as client:
            resp = client.get("/health", headers={"Origin": "http://evil.example.com"})
            allow_origin = resp.headers.get("access-control-allow-origin")
            assert allow_origin != "http://evil.example.com"
            assert allow_origin != "*"

    def test_preflight_allowed_origin_reflected(self):
        from src.main import app

        with TestClient(app) as client:
            resp = client.options(
                "/api/notes/",
                headers={
                    "Origin": "http://localhost:3369",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert resp.headers.get("access-control-allow-origin") == "http://localhost:3369"
