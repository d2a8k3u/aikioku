"""Tests for production gap modules with low coverage."""
from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# LLM Router / CostTracker / CircuitBreaker
# ---------------------------------------------------------------------------

class TestCostTracker:
    def test_init_creates_table(self):
        from src.llm.router import CostTracker, CostRecord
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        ct = CostTracker(path, daily_budget_usd=2.0)
        ct.record(CostRecord(provider="openai", model="gpt-4", prompt_tokens=1000, completion_tokens=500, cost_usd=0.003))
        assert ct.get_today_cost() == 0.003
        assert ct.check_budget(1.0) is True
        assert ct.check_budget(10.0) is False
        stats = ct.get_stats(days=1)
        assert stats["daily_budget"] == 2.0
        assert stats["today_cost"] == 0.003
        assert len(stats["providers"]) == 1

    def test_estimate_cost(self):
        from src.llm.router import CostTracker
        ct = CostTracker(":memory:")
        # All remaining providers (ollama, ollama_remote, openrouter) have 0.0 pricing.
        cost = ct.estimate_cost("ollama", 2000, 1000)
        assert cost == 0.0


class TestCircuitState:
    def test_open_after_threshold(self):
        from src.llm.router import CircuitState
        cs = CircuitState()
        assert cs.is_open(threshold=3, cooldown_seconds=10) is False
        for _ in range(3):
            cs.record_failure()
        assert cs.is_open(threshold=3, cooldown_seconds=10) is True
        cs.record_success()
        assert cs.is_open(threshold=3, cooldown_seconds=10) is False

    def test_open_resets_after_cooldown(self):
        from src.llm.router import CircuitState
        cs = CircuitState()
        cs.record_failure()
        cs.record_failure()
        cs.record_failure()
        cs.open_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert cs.is_open(threshold=3, cooldown_seconds=0) is False


class TestLLMRouter:
    @pytest.mark.asyncio
    async def test_complete_uses_first_available(self):
        from src.llm.router import LLMRouter
        p = MagicMock()
        p.is_available.return_value = True
        p.complete = AsyncMock(return_value="ok")
        router = LLMRouter([p])
        result = await router.complete("hi")
        assert result == "ok"
        p.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complete_fallback_on_failure(self):
        from src.llm.router import LLMRouter
        p1 = MagicMock()
        p1.is_available.return_value = True
        p1.complete = AsyncMock(side_effect=RuntimeError("fail"))
        p2 = MagicMock()
        p2.is_available.return_value = True
        p2.complete = AsyncMock(return_value="fallback")
        router = LLMRouter([p1, p2], cooldown_seconds=0)
        result = await router.complete("hi")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_stream_fallback(self):
        from src.llm.router import LLMRouter
        p1 = MagicMock()
        p1.is_available.return_value = True
        async def _bad_stream(*a, **k):
            raise RuntimeError("fail")
        p1.stream = _bad_stream
        p2 = MagicMock()
        p2.is_available.return_value = True
        p2.complete = AsyncMock(return_value="fallback-text")
        router = LLMRouter([p1, p2], cooldown_seconds=0)
        chunks = [c async for c in router.stream("hi")]
        assert chunks == ["fallback-text"]

    @pytest.mark.asyncio
    async def test_embed_fallback(self):
        from src.llm.router import LLMRouter
        p1 = MagicMock()
        p1.is_available.return_value = True
        p1.embed = AsyncMock(side_effect=RuntimeError("fail"))
        p2 = MagicMock()
        p2.is_available.return_value = True
        p2.embed = AsyncMock(return_value=[0.1, 0.2])
        router = LLMRouter([p1, p2], cooldown_seconds=0)
        vec = await router.embed("hi")
        assert vec == [0.1, 0.2]

    def test_is_available(self):
        from src.llm.router import LLMRouter
        p = MagicMock()
        p.is_available.return_value = True
        router = LLMRouter([p])
        assert router.is_available() is True

    def test_get_circuit_states(self):
        from src.llm.router import LLMRouter
        p = MagicMock()
        p.is_available.return_value = True
        router = LLMRouter([p])
        states = router.get_circuit_states()
        assert len(states) == 1
        assert "failures" in states[0]


# ---------------------------------------------------------------------------
# Plugin API
# ---------------------------------------------------------------------------

class TestPluginAPI:
    def test_register_and_call_hook(self):
        from src.plugins.api import PluginAPI
        api = PluginAPI()
        called = []
        def handler(*a, **k):
            called.append(True)
        api.register("onNoteSave", handler)
        api.call("onNoteSave", {"id": "1"})
        assert len(called) == 1

    def test_call_missing_hook_noops(self):
        from src.plugins.api import PluginAPI
        api = PluginAPI()
        api.call("onQuery", {"q": "x"})

    def test_call_async(self):
        from src.plugins.api import PluginAPI
        api = PluginAPI()
        async def async_handler(x):
            return x["v"]
        api.register("onNoteSave", async_handler)
        result = asyncio.run(api.call_async("onNoteSave", {"v": 42}))
        assert result == [42]

    def test_get_registered_hooks(self):
        from src.plugins.api import PluginAPI
        api = PluginAPI()
        api.register("onNoteSave", lambda x: x)
        hooks = api.get_registered_hooks()
        assert "onNoteSave" in hooks
        assert len(hooks["onNoteSave"]) == 1

    def test_register_invalid_hook_raises(self):
        from src.plugins.api import PluginAPI
        api = PluginAPI()
        with pytest.raises(ValueError):
            api.register("onInvalid", lambda: None)


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

class TestAnomalyDetector:
    def test_record_and_get_recent(self):
        from src.reasoning.anomaly import AnomalyDetector, AnomalyResult
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        ad = AnomalyDetector(path)
        ad.record(AnomalyResult(type="test", severity="low", description="d"))
        recent = ad.get_recent(hours=1)
        assert len(recent) == 1
        ad.resolve(recent[0]["id"])
        assert len(ad.get_recent(hours=1)) == 0

    def test_run_all_with_empty_graph(self):
        from src.reasoning.anomaly import AnomalyDetector
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        ad = AnomalyDetector(path)
        graph = MagicMock()
        graph.find_entities = MagicMock(return_value=[])
        results = ad.run_all(graph, path)
        assert isinstance(results, list)


class TestAnomalyAPI:
    def test_recent_returns_list(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            resp = client.get("/api/anomaly/recent")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)


# ---------------------------------------------------------------------------
# Auto Tagging
# ---------------------------------------------------------------------------

class TestAutoTagger:
    def test_rule_based_tags(self):
        from src.models.note import Note
        from src.augmentation.auto_tag import AutoTagger
        engine = AutoTagger()
        note = Note(title="T", content="#python #api Learn FastAPI today!", path="x.md")
        tags = engine.suggest_tags_for_text(note.content)
        assert "python" in tags
        assert "api" in tags

    @pytest.mark.skip(reason="Requires JWT auth token")
    def test_api_returns_tags(self):
        from fastapi.testclient import TestClient
        from src.main import app
        with TestClient(app) as client:
            # create a note first
            create = client.post("/api/notes/", json={"title": "ML Note", "content": "Machine Learning in Python", "path": "ml.md"})
            assert create.status_code == 201
            note_id = create.json()["id"]
            resp = client.post(f"/api/tags/auto/{note_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data.get("tags"), list)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

class TestWebSocketEndpoint:
    def test_websocket_connect_and_disconnect(self):
        from fastapi.testclient import TestClient
        from src.main import app
        from src.api import websocket as ws_mod
        ws_mod.set_broadcaster(MagicMock())
        with TestClient(app) as client:
            with client.websocket_connect("/ws/events") as ws:
                ws.send_text("ping")


# ---------------------------------------------------------------------------
# Memory consolidation stage 4 (summarize)
# ---------------------------------------------------------------------------

class TestMemoryConsolidationStage4:
    @pytest.mark.asyncio
    async def test_stage_summarize_runs(self):
        from src.memory.consolidation import MemoryConsolidator
        from src.models.memory import Memory
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        from src.events import EventBus
        eb = EventBus(path)
        mc = MemoryConsolidator(None, eb)
        memories = [
            Memory(subject="Alice", predicate="knows", object="Bob", confidence=0.9, source="note1"),
            Memory(subject="Alice", predicate="knows", object="Charlie", confidence=0.8, source="note2"),
        ]
        result = await mc.stage_summarize(memories)
        assert isinstance(result, list)
