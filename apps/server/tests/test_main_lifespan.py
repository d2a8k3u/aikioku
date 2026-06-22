"""Tests for lifespan background task and service initialization."""
from __future__ import annotations

import asyncio

import pytest
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
    """CORS must reflect an explicit allowlist, never a wildcard (Phase 4)."""

    def test_allowed_origin_is_reflected(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.get(
                "/health", headers={"Origin": "http://localhost:3369"}
            )
            allow_origin = resp.headers.get("access-control-allow-origin")
            assert allow_origin == "http://localhost:3369"
            assert allow_origin != "*"

    def test_disallowed_origin_is_not_reflected(self):
        from src.main import app
        with TestClient(app) as client:
            resp = client.get(
                "/health", headers={"Origin": "http://evil.example.com"}
            )
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
            assert (
                resp.headers.get("access-control-allow-origin")
                == "http://localhost:3369"
            )


class TestConsolidationWorker:
    """Test the background consolidation worker logic."""

    @pytest.mark.asyncio
    async def test_worker_runs_and_exits(self):
        from src.main import _consolidation_worker
        from types import SimpleNamespace

        # The worker now reads app.state each iteration. Unconfigured
        # (llm_provider is None) -> it skips cleanly. Verify it starts and
        # cancels without hanging.
        fake_app = SimpleNamespace(state=SimpleNamespace(llm_provider=None))
        task = asyncio.create_task(
            _consolidation_worker(fake_app, interval_hours=0)
        )
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # If we got here without hanging, the worker starts and stops correctly
        assert True

    @pytest.mark.asyncio
    async def test_run_consolidation_once_loads_and_persists(self, tmp_path):
        """The per-run helper loads persisted memories, runs the consolidator
        against the SHARED graph, and persists the processed results back."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.main import _run_consolidation_once
        from src.events import EventBus
        from src.api import memory as memory_api
        from src.models.memory import Memory, MemoryTier

        # conftest.reset_app_state already points sqlite_db_path at a tmp db.
        memory_api._ensure_table()
        seed = Memory(subject="Seed", predicate="is", object="Stored",
                      confidence=0.9, source="conversation",
                      vitality_score=0.9, tier=MemoryTier.warm)
        memory_api._store_memories([seed])

        shared_graph = MagicMock(name="shared_graph")
        captured = {}

        async def _fake_run(memories):
            captured["input_ids"] = [m.id for m in memories]
            processed = [seed.model_copy(update={"tier": MemoryTier.hot})]
            return {
                "input_count": len(memories),
                "output_count": len(processed),
                "duplicates_removed": 0,
                "conflicts_detected": 0,
                "archived_count": 0,
                "memories": processed,
            }

        stub = MagicMock()
        stub.run = AsyncMock(side_effect=_fake_run)

        event_bus = EventBus(str(tmp_path / "events.db"))

        with patch("src.main.MemoryConsolidator", return_value=stub) as ctor, \
             patch("src.main.KnowledgeGraph") as kg_cls:
            summary = await _run_consolidation_once(
                event_bus, graph=shared_graph, llm_provider=None
            )

        # Loaded the seeded memory and fed it in.
        assert seed.id in captured["input_ids"]
        # Used the shared graph; never opened a new Kuzu handle.
        ctor.assert_called_once()
        assert ctor.call_args.args[0] is shared_graph
        kg_cls.assert_not_called()
        # Persisted the processed (tier-updated) memory back.
        rows = {r["id"]: r for r in memory_api._load_memories()}
        assert rows[seed.id]["tier"] == "hot"
        assert summary["output_count"] == 1
