"""Tests for Connection Discovery API endpoint."""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI TestClient with a seeded KnowledgeGraph and mock EmbeddingStore."""
    from src.knowledge.graph import KnowledgeGraph
    from src.models.entity import Entity, EntityType
    from src.models.relation import Relation, RelationType
    from src.main import app

    with tempfile.TemporaryDirectory() as tmpdir:
        kg_path = os.path.join(tmpdir, "test_kg.db")
        kg = KnowledgeGraph(db_path=kg_path)
        app.state.knowledge_graph = kg

        e1 = Entity(name="Alice", type=EntityType.Person, confidence=0.95)
        e2 = Entity(name="Bob", type=EntityType.Person, confidence=0.88)
        e3 = Entity(name="Acme Corp", type=EntityType.Organization, confidence=0.90)

        kg.create_entity(e1)
        kg.create_entity(e2)
        kg.create_entity(e3)

        kg.create_relation(Relation(
            source_entity_id=e1.id, target_entity_id=e2.id, type=RelationType.works_at,
        ))
        kg.create_relation(Relation(
            source_entity_id=e2.id, target_entity_id=e3.id, type=RelationType.works_at,
        ))

        # Mock embedding store with minimal interface needed by ConnectionDiscovery
        class FakeEmbeddingStore:
            db_path = os.path.join(tmpdir, "fake_emb.db")

        app.state.embedding_store = FakeEmbeddingStore()

        yield TestClient(app), e1, e2, e3


class TestDiscoverConnections:
    def test_returns_connections(self, client):
        cli, e1, e2, e3 = client
        response = cli.post(f"/api/connections/discover?entity_id={e1.id}&max_distance=3")
        assert response.status_code == 200
        data = response.json()
        assert data["entity_id"] == e1.id
        assert "connections" in data
        assert isinstance(data["connections"], list)

    def test_connections_have_fields(self, client):
        cli, e1, e2, e3 = client
        response = cli.post(f"/api/connections/discover?entity_id={e1.id}&max_distance=3")
        data = response.json()
        for conn in data["connections"]:
            assert "path" in conn
            assert "strength" in conn
            assert "explanation" in conn
            assert isinstance(conn["path"], list)
            assert isinstance(conn["strength"], float)

    def test_missing_entity_id_returns_400(self, client):
        cli, *rest = client
        response = cli.post("/api/connections/discover?max_distance=3")
        assert response.status_code == 400

    def test_empty_connections_for_isolated(self, client):
        cli, e1, e2, e3 = client
        from src.models.entity import Entity, EntityType
        isolated = Entity(name="Solo", type=EntityType.Concept, confidence=0.90)
        app = cli.app
        app.state.knowledge_graph.create_entity(isolated)
        response = cli.post(f"/api/connections/discover?entity_id={isolated.id}&max_distance=3")
        assert response.status_code == 200
        data = response.json()
        assert data["connections"] == []
