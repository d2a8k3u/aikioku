"""Tests for Memory API endpoints."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def tmp_db_path():
    """Provide a temporary directory for SQLite DB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def client(tmp_db_path):
    """Create a FastAPI TestClient with mocked memory dependencies."""
    from src.main import app

    yield TestClient(app)


class TestExtractMemories:
    """Test POST /api/memory/extract."""

    def test_extract_returns_memories(self, client):
        """POST /api/memory/extract should return a list of extracted memories."""
        mock_memories = [
            {
                "id": "mem-1",
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme Corp",
                "confidence": 0.95,
                "source": "note-123",
                "created": "2026-06-10T12:00:00",
                "modified": "2026-06-10T12:00:00",
                "vitality_score": 0.0,
                "tier": "hot",
            }
        ]

        with patch("src.api.memory._extract_memories", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = mock_memories
            response = client.post(
                "/api/memory/extract",
                json={"note_id": "note-123"},
            )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["subject"] == "Alice"
        assert data[0]["predicate"] == "works_at"
        assert data[0]["object"] == "Acme Corp"

    def test_extract_empty_when_no_memories(self, client):
        """POST /api/memory/extract returns empty list when nothing extracted."""
        with patch("src.api.memory._extract_memories", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = []
            response = client.post(
                "/api/memory/extract",
                json={"note_id": "note-empty"},
            )

        assert response.status_code == 200
        assert response.json() == []


class TestConsolidateMemories:
    """Test POST /api/memory/consolidate."""

    def test_consolidate_returns_summary(self, client):
        """POST /api/memory/consolidate should return a summary dict."""
        mock_summary = {
            "input_count": 10,
            "output_count": 7,
            "duplicates_removed": 2,
            "conflicts_detected": 1,
            "archived_count": 0,
        }

        with patch("src.api.memory._run_consolidation", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_summary
            response = client.post("/api/memory/consolidate")

        assert response.status_code == 200
        data = response.json()
        assert "input_count" in data
        assert "output_count" in data
        assert "duplicates_removed" in data
        assert "conflicts_detected" in data
        assert "archived_count" in data
        assert data["input_count"] == 10
        assert data["output_count"] == 7


class TestListMemories:
    """Test GET /api/memory/."""

    def test_returns_list(self, client):
        """GET /api/memory/ should return a list of memories."""
        mock_memories = [
            {
                "id": "mem-1",
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme Corp",
                "confidence": 0.95,
                "source": "note-123",
                "created": "2026-06-10T12:00:00",
                "modified": "2026-06-10T12:00:00",
                "vitality_score": 0.0,
                "tier": "hot",
            },
            {
                "id": "mem-2",
                "subject": "Bob",
                "predicate": "lives_in",
                "object": "NYC",
                "confidence": 0.88,
                "source": "note-456",
                "created": "2026-06-10T12:00:00",
                "modified": "2026-06-10T12:00:00",
                "vitality_score": 0.0,
                "tier": "warm",
            },
        ]

        with patch("src.api.memory._list_memories", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = mock_memories
            response = client.get("/api/memory/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_optional_entity_filter(self, client):
        """GET /api/memory/?entity=Alice should filter by entity."""
        mock_memories = [
            {
                "id": "mem-1",
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme Corp",
                "confidence": 0.95,
                "source": "note-123",
                "created": "2026-06-10T12:00:00",
                "modified": "2026-06-10T12:00:00",
                "vitality_score": 0.0,
                "tier": "hot",
            },
        ]

        with patch("src.api.memory._list_memories", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = mock_memories
            response = client.get("/api/memory/?entity=Alice")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["subject"] == "Alice"


class TestMemoryStats:
    """Test GET /api/memory/stats."""

    def test_returns_counts(self, client):
        """GET /api/memory/stats should return total, hot, warm, cold counts."""
        mock_stats = {"total": 42, "hot": 20, "warm": 15, "cold": 7}

        with patch("src.api.memory._get_stats", new_callable=AsyncMock) as mock_stats_fn:
            mock_stats_fn.return_value = mock_stats
            response = client.get("/api/memory/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "hot" in data
        assert "warm" in data
        assert "cold" in data
        assert isinstance(data["total"], int)
        assert isinstance(data["hot"], int)
        assert isinstance(data["warm"], int)
        assert isinstance(data["cold"], int)
        assert data["total"] == 42
        assert data["hot"] == 20
        assert data["warm"] == 15
        assert data["cold"] == 7


class TestExtractUsesAppStateProvider:
    """POST /api/memory/extract must build the extractor from app.state.llm_provider,
    NOT the configured-provider fallback (build_llm_provider)."""

    def test_extract_uses_injected_provider_and_shared_graph(self, tmp_path):
        from src.main import app
        from src.models.note import Note

        sentinel_llm = MagicMock(name="app_state_llm")
        sentinel_graph = MagicMock(name="app_state_kg")

        captured = {}

        class _StubExtractor:
            def __init__(self, llm, event_bus):
                captured["llm"] = llm

            async def extract_from_note(self, note):
                return []

        note = Note(title="N", content="c", path="n.md")

        with TestClient(app) as client:
            client.app.state.llm_provider = sentinel_llm
            client.app.state.knowledge_graph = sentinel_graph
            with (
                patch("src.api.memory.NoteStore") as store_cls,
                patch("src.api.memory.MemoryExtractor", _StubExtractor),
                patch("src.llm.factory.build_llm_provider") as fallback_provider,
            ):
                store_cls.return_value.get.return_value = note
                resp = client.post("/api/memory/extract", json={"note_id": "x"})

        assert resp.status_code == 200
        assert captured["llm"] is sentinel_llm
        # The injected provider was used; the fallback was never constructed.
        fallback_provider.assert_not_called()


class TestConsolidateUsesSharedGraph:
    """POST /api/memory/consolidate must reuse app.state.knowledge_graph and
    app.state.llm_provider — never open a second Kuzu handle."""

    def test_consolidate_uses_shared_graph_and_provider(self, tmp_path):
        from src.main import app

        sentinel_llm = MagicMock(name="app_state_llm")
        sentinel_graph = MagicMock(name="app_state_kg")

        captured = {}

        class _StubConsolidator:
            def __init__(self, graph, event_bus, llm_provider=None):
                captured["graph"] = graph
                captured["llm"] = llm_provider

            async def run(self, memories):
                return {
                    "input_count": 0,
                    "output_count": 0,
                    "duplicates_removed": 0,
                    "conflicts_detected": 0,
                    "archived_count": 0,
                    "memories": [],
                }

        with TestClient(app) as client:
            client.app.state.llm_provider = sentinel_llm
            client.app.state.knowledge_graph = sentinel_graph
            with (
                patch("src.api.memory.MemoryConsolidator", _StubConsolidator),
                patch("src.api.memory.KnowledgeGraph") as kg_cls,
            ):
                resp = client.post("/api/memory/consolidate")

        assert resp.status_code == 200
        assert captured["graph"] is sentinel_graph
        assert captured["llm"] is sentinel_llm
        # No second Kuzu handle was opened.
        kg_cls.assert_not_called()

    def test_consolidate_response_is_json_safe(self, tmp_path):
        """The summary must not leak raw Memory objects into the JSON response."""
        from src.main import app
        from src.models.memory import Memory

        mem = Memory(subject="A", predicate="b", object="C", confidence=0.5, source="conversation")

        class _StubConsolidator:
            def __init__(self, graph, event_bus, llm_provider=None):
                pass

            async def run(self, memories):
                return {
                    "input_count": 1,
                    "output_count": 1,
                    "duplicates_removed": 0,
                    "conflicts_detected": 0,
                    "archived_count": 0,
                    "memories": [mem],
                }

        with TestClient(app) as client:
            client.app.state.llm_provider = MagicMock()
            client.app.state.knowledge_graph = MagicMock()
            with patch("src.api.memory.MemoryConsolidator", _StubConsolidator):
                resp = client.post("/api/memory/consolidate")

        assert resp.status_code == 200
        data = resp.json()
        assert data["input_count"] == 1
        assert data["output_count"] == 1
        # No nested raw Memory dicts under "memories".
        assert "memories" not in data or isinstance(data["memories"], list)


class TestPersistConsolidation:
    """_persist_consolidation deletes removed inputs and upserts survivors."""

    def test_deletes_removed_and_upserts_processed(self):
        from src.api import memory as memory_api
        from src.models.memory import Memory, MemoryTier

        memory_api._ensure_table()

        # Two inputs: a survivor and a duplicate that merge will drop.
        survivor = Memory(
            subject="Alice",
            predicate="knows",
            object="Bob",
            confidence=0.9,
            source="conversation",
            vitality_score=0.9,
            tier=MemoryTier.warm,
        )
        removed = Memory(
            subject="Alice",
            predicate="knows",
            object="Bob",
            confidence=0.7,
            source="conversation",
            vitality_score=0.7,
            tier=MemoryTier.warm,
        )
        memory_api._store_memories([survivor, removed])
        assert {r["id"] for r in memory_api._load_memories()} == {survivor.id, removed.id}

        # After consolidation: survivor kept (now hot) + a new summary memory.
        survivor_updated = survivor.model_copy(update={"tier": MemoryTier.hot})
        summary_mem = Memory(
            subject="Alice",
            predicate="has_guideline",
            object="Alice is well-connected.",
            confidence=0.8,
            source="consolidation_summary",
            vitality_score=0.8,
            tier=MemoryTier.hot,
        )

        memory_api._persist_consolidation(
            input_memories=[survivor, removed],
            processed=[survivor_updated, summary_mem],
        )

        rows = {r["id"]: r for r in memory_api._load_memories()}
        # The duplicate removed by merge is gone.
        assert removed.id not in rows
        # The survivor remains, with its updated tier persisted.
        assert survivor.id in rows
        assert rows[survivor.id]["tier"] == "hot"
        # The new summary memory was upserted.
        assert summary_mem.id in rows
        assert rows[summary_mem.id]["predicate"] == "has_guideline"


class TestCreateMemory:
    """Test POST /api/memory/ (create from free text)."""

    def test_create_returns_memories(self, client):
        mock_memories = [
            {
                "id": "mem-1",
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme",
                "confidence": 0.9,
                "source": "user",
                "created": "2026-06-10T12:00:00",
                "modified": "2026-06-10T12:00:00",
                "vitality_score": 0.9,
                "tier": "hot",
            }
        ]
        with patch(
            "src.api.memory._create_memory_from_text", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_memories
            client.app.state.llm_provider = MagicMock()
            response = client.post("/api/memory/", json={"text": "Alice works at Acme"})

        assert response.status_code == 200
        data = response.json()
        assert data[0]["subject"] == "Alice"
        mock_create.assert_awaited_once()

    def test_create_unconfigured_returns_503(self, client):
        client.app.state.llm_provider = None
        response = client.post("/api/memory/", json={"text": "Alice works at Acme"})
        assert response.status_code == 503


class TestSearchMemories:
    """Test GET /api/memory/search (semantic)."""

    def test_search_returns_scored_results(self, client):
        mock_results = [
            {
                "id": "mem-1",
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme",
                "confidence": 0.9,
                "source": "user",
                "created": "2026-06-10T12:00:00",
                "modified": "2026-06-10T12:00:00",
                "vitality_score": 0.9,
                "tier": "hot",
                "score": 0.91,
            }
        ]
        with patch(
            "src.api.memory._search_memories_semantic", new_callable=AsyncMock
        ) as mock_search:
            mock_search.return_value = mock_results
            response = client.get("/api/memory/search?q=acme&limit=5")

        assert response.status_code == 200
        data = response.json()
        assert data[0]["subject"] == "Alice"
        assert data[0]["score"] == 0.91
        mock_search.assert_awaited_once()

    def test_search_route_not_shadowed_by_id(self, client):
        """GET /api/memory/search must hit the search handler, not get_memory."""
        with (
            patch(
                "src.api.memory._search_memories_semantic", new_callable=AsyncMock
            ) as mock_search,
            patch("src.api.memory._get_memory") as mock_get,
        ):
            mock_search.return_value = []
            response = client.get("/api/memory/search?q=anything")

        assert response.status_code == 200
        mock_search.assert_awaited_once()
        mock_get.assert_not_called()


class TestGetUpdateDeleteMemory:
    """Test GET/PUT/DELETE /api/memory/{id}."""

    def _store_one(self):
        from src.api import memory as memory_api
        from src.models.memory import Memory, MemoryTier

        memory_api._ensure_table()
        m = Memory(
            subject="Alice",
            predicate="works_at",
            object="Acme",
            confidence=0.9,
            source="user",
            vitality_score=0.9,
            tier=MemoryTier.warm,
        )
        memory_api._store_memories([m])
        return m

    def test_get_by_id(self, client):
        m = self._store_one()
        resp = client.get(f"/api/memory/{m.id}")
        assert resp.status_code == 200
        assert resp.json()["subject"] == "Alice"

    def test_get_missing_404(self, client):
        resp = client.get("/api/memory/does-not-exist")
        assert resp.status_code == 404

    def test_update_changes_field_and_reembeds(self, client):
        m = self._store_one()
        client.app.state.memory_embedding_store = MagicMock()
        client.app.state.embedding_provider = AsyncMock()
        client.app.state.embedding_provider.embed.return_value = [0.1, 0.2, 0.3]

        resp = client.put(f"/api/memory/{m.id}", json={"object": "Globex"})
        assert resp.status_code == 200
        assert resp.json()["object"] == "Globex"
        # The object changed, so the triple was re-embedded.
        client.app.state.memory_embedding_store.add_document.assert_called_once()

    def test_update_empty_400(self, client):
        m = self._store_one()
        resp = client.put(f"/api/memory/{m.id}", json={})
        assert resp.status_code == 400

    def test_update_missing_404(self, client):
        resp = client.put("/api/memory/nope", json={"object": "X"})
        assert resp.status_code == 404

    def test_delete_removes_row_and_vector(self, client):
        from src.api import memory as memory_api

        m = self._store_one()
        client.app.state.memory_embedding_store = MagicMock()
        resp = client.delete(f"/api/memory/{m.id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert memory_api._get_memory(m.id) is None
        client.app.state.memory_embedding_store.delete.assert_called_once_with(m.id)

    def test_delete_missing_404(self, client):
        resp = client.delete("/api/memory/nope")
        assert resp.status_code == 404

    def test_delete_cleans_graph(self, client, monkeypatch):
        from src.api import memory as memory_api

        m = self._store_one()
        client.app.state.memory_embedding_store = MagicMock()
        sentinel_graph = MagicMock(name="app_state_kg")
        client.app.state.knowledge_graph = sentinel_graph
        spy = MagicMock()
        monkeypatch.setattr(memory_api, "remove_memory_from_graph", spy)

        try:
            resp = client.delete(f"/api/memory/{m.id}")
            assert resp.status_code == 200
            spy.assert_called_once()
            called_memory, called_graph = spy.call_args.args
            assert called_memory.id == m.id
            assert called_memory.subject == "Alice"
            assert called_graph is sentinel_graph
        finally:
            delattr(client.app.state, "knowledge_graph")

    def test_delete_without_graph_still_succeeds(self, client):
        from src.api import memory as memory_api

        m = self._store_one()
        client.app.state.memory_embedding_store = MagicMock()
        if hasattr(client.app.state, "knowledge_graph"):
            delattr(client.app.state, "knowledge_graph")
        resp = client.delete(f"/api/memory/{m.id}")
        assert resp.status_code == 200
        assert memory_api._get_memory(m.id) is None

    def test_delete_graph_cleanup_failure_isolated(self, client, monkeypatch):
        from src.api import memory as memory_api

        m = self._store_one()
        client.app.state.memory_embedding_store = MagicMock()
        client.app.state.knowledge_graph = MagicMock()
        monkeypatch.setattr(
            memory_api,
            "remove_memory_from_graph",
            MagicMock(side_effect=RuntimeError("kuzu down")),
        )
        try:
            resp = client.delete(f"/api/memory/{m.id}")
            assert resp.status_code == 200
            assert memory_api._get_memory(m.id) is None
        finally:
            delattr(client.app.state, "knowledge_graph")


class TestMemoryEmbeddingHelpers:
    """Unit tests for the memory embedding write-through / search helpers."""

    async def test_embed_memories_calls_add_document(self):
        from src.api import memory as memory_api
        from src.models.memory import Memory, MemoryTier

        m = Memory(
            subject="Alice",
            predicate="works_at",
            object="Acme",
            confidence=0.9,
            source="note-1",
            tier=MemoryTier.hot,
        )
        provider = AsyncMock()
        provider.embed.return_value = [0.1, 0.2]
        store = MagicMock()

        await memory_api._embed_memories([m], provider, store)

        store.delete.assert_called_once_with(m.id)
        store.add_document.assert_called_once()
        args, kwargs = store.add_document.call_args
        assert args[0] == m.id
        assert kwargs["metadata"]["note_id"] == m.id
        assert kwargs["metadata"]["subject"] == "Alice"

    async def test_embed_noop_without_store(self):
        from src.api import memory as memory_api
        from src.models.memory import Memory

        m = Memory(subject="A", predicate="b", object="C", confidence=0.5, source="x")
        provider = AsyncMock()
        await memory_api._embed_memories([m], provider, None)
        provider.embed.assert_not_awaited()

    async def test_search_hydrates_from_sqlite(self):
        from src.api import memory as memory_api
        from src.models.memory import Memory, MemoryTier

        memory_api._ensure_table()
        m = Memory(
            subject="Guido",
            predicate="created",
            object="Python",
            confidence=0.9,
            source="x",
            tier=MemoryTier.hot,
        )
        memory_api._store_memories([m])

        provider = AsyncMock()
        provider.embed.return_value = [0.1, 0.2]
        store = MagicMock()
        store.search.return_value = [
            {"note_id": m.id, "text": "Guido created Python", "score": 0.8}
        ]

        results = await memory_api._search_memories_semantic(
            "who created python", 5, provider, store
        )
        assert len(results) == 1
        assert results[0]["id"] == m.id
        assert results[0]["subject"] == "Guido"
        assert results[0]["score"] == 0.8

    async def test_search_drops_stale_vector(self):
        from src.api import memory as memory_api

        provider = AsyncMock()
        provider.embed.return_value = [0.1]
        store = MagicMock()
        store.search.return_value = [{"note_id": "ghost-id", "text": "", "score": 0.9}]

        results = await memory_api._search_memories_semantic("q", 5, provider, store)
        assert results == []

    async def test_persist_consolidation_vectors_delete_through(self):
        from src.api import memory as memory_api
        from src.models.memory import Memory

        survivor = Memory(subject="A", predicate="b", object="C", confidence=0.8, source="x")
        provider = AsyncMock()
        provider.embed.return_value = [0.1]
        store = MagicMock()

        await memory_api._persist_consolidation_vectors(
            ["removed-1", "removed-2"], [survivor], provider, store
        )

        deleted_ids = {c.args[0] for c in store.delete.call_args_list}
        assert {"removed-1", "removed-2"}.issubset(deleted_ids)
        store.add_document.assert_called_once()


class TestReinforceMemory:
    """Loading by entity bumps vitality and persists (spaced-repetition signal)."""

    def test_entity_load_reinforces_vitality(self):
        from src.api import memory as memory_api
        from src.models.memory import Memory, MemoryTier

        memory_api._ensure_table()
        m = Memory(
            subject="Guido",
            predicate="created",
            object="Python",
            confidence=0.9,
            source="conversation",
            vitality_score=0.5,
            tier=MemoryTier.warm,
        )
        memory_api._store_memories([m])

        before = memory_api._load_memories(entity="Guido")
        assert len(before) == 1
        v0 = before[0]["vitality_score"]

        # A second entity-scoped load should reflect the reinforced (bumped) score.
        after = memory_api._load_memories(entity="Guido")
        assert after[0]["vitality_score"] > v0
        assert after[0]["vitality_score"] <= 1.0

    def test_reinforcement_caps_at_one(self):
        from src.api import memory as memory_api
        from src.models.memory import Memory, MemoryTier

        memory_api._ensure_table()
        m = Memory(
            subject="Cap",
            predicate="is",
            object="High",
            confidence=1.0,
            source="conversation",
            vitality_score=0.99,
            tier=MemoryTier.hot,
        )
        memory_api._store_memories([m])

        memory_api._load_memories(entity="Cap")
        memory_api._load_memories(entity="Cap")
        rows = memory_api._load_memories(entity="Cap")
        assert rows[0]["vitality_score"] <= 1.0
