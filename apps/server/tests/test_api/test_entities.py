"""Tests for flat Entities API endpoints at /api/entities."""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def tmp_db_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def client(tmp_db_path):
    """Create a FastAPI TestClient with a real KnowledgeGraph in app state."""
    from src.knowledge.graph import KnowledgeGraph
    from src.models.entity import Entity, EntityType
    from src.models.relation import Relation, RelationType
    from src.main import app

    kg = KnowledgeGraph(db_path=tmp_db_path)
    app.state.knowledge_graph = kg

    # Seed test data
    e1 = Entity(name="Alice", type=EntityType.Person, confidence=0.95)
    e2 = Entity(name="Bob", type=EntityType.Person, confidence=0.88)
    e3 = Entity(name="Acme Corp", type=EntityType.Organization, confidence=0.90)

    kg.create_entity(e1)
    kg.create_entity(e2)
    kg.create_entity(e3)

    kg.create_relation(Relation(
        source_entity_id=e1.id, target_entity_id=e2.id, type=RelationType.works_at,
    ))

    yield TestClient(app), kg, e1, e2, e3


class TestListEntities:
    def test_returns_list(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/entities")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_optional_type_filter(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/entities?type=Person")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        names = {e["name"] for e in data}
        assert names == {"Alice", "Bob"}

    def test_entities_have_expected_fields(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/entities")
        data = response.json()
        for entity in data:
            assert "id" in entity
            assert "name" in entity
            assert "type" in entity
            assert "aliases" in entity
            assert "properties" in entity
            assert "confidence" in entity
            assert "source_note_ids" in entity


class TestGetEntity:
    def test_returns_entity(self, client):
        cli, kg, e1, *rest = client
        response = cli.get(f"/api/entities/{e1.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == e1.id
        assert data["name"] == "Alice"
        assert data["type"] == "Person"

    def test_nonexistent_returns_404(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/entities/nonexistent-id")
        assert response.status_code == 404


class TestGetSubgraph:
    def test_returns_subgraph(self, client):
        cli, kg, e1, e2, e3 = client
        response = cli.get(f"/api/entities/{e1.id}/subgraph")
        assert response.status_code == 200
        data = response.json()
        assert data["root_id"] == e1.id
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)

    def test_subgraph_depth_param(self, client):
        cli, kg, e1, e2, e3 = client
        response = cli.get(f"/api/entities/{e1.id}/subgraph?depth=1")
        assert response.status_code == 200
        data = response.json()
        assert data["depth"] == 1
        assert data["root_id"] == e1.id

    def test_nonexistent_subgraph_returns_404(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/entities/nonexistent-id/subgraph")
        assert response.status_code == 404
