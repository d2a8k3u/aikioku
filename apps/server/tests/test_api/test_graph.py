"""Tests for Graph API endpoints."""

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

    # Inject a real KnowledgeGraph into app state
    kg = KnowledgeGraph(db_path=tmp_db_path)
    app.state.knowledge_graph = kg

    # Seed test data
    e1 = Entity(name="Alice", type=EntityType.Person, confidence=0.95)
    e2 = Entity(name="Bob", type=EntityType.Person, confidence=0.88)
    e3 = Entity(name="Acme Corp", type=EntityType.Organization, confidence=0.90)
    e4 = Entity(name="Project X", type=EntityType.Project, confidence=0.80)

    kg.create_entity(e1)
    kg.create_entity(e2)
    kg.create_entity(e3)
    kg.create_entity(e4)

    kg.create_relation(
        Relation(
            source_entity_id=e1.id,
            target_entity_id=e2.id,
            type=RelationType.works_at,
        )
    )
    kg.create_relation(
        Relation(
            source_entity_id=e1.id,
            target_entity_id=e3.id,
            type=RelationType.created,
        )
    )
    kg.create_relation(
        Relation(
            source_entity_id=e2.id,
            target_entity_id=e4.id,
            type=RelationType.depends_on,
        )
    )

    yield TestClient(app), kg, e1, e2, e3, e4


class TestListEntities:
    def test_returns_list(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/graph/entities")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 4

    def test_optional_type_filter(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/graph/entities?type=Person")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        names = {e["name"] for e in data}
        assert names == {"Alice", "Bob"}

    def test_entities_have_expected_fields(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/graph/entities")
        data = response.json()
        for entity in data:
            assert "id" in entity
            assert "name" in entity
            assert "type" in entity


class TestGetEntity:
    def test_returns_entity(self, client):
        cli, kg, e1, *rest = client
        response = cli.get(f"/api/graph/entities/{e1.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == e1.id
        assert data["name"] == "Alice"

    def test_returns_entity_with_relations(self, client):
        cli, kg, e1, *rest = client
        response = cli.get(f"/api/graph/entities/{e1.id}")
        data = response.json()
        assert "relations" in data
        assert isinstance(data["relations"], list)

    def test_nonexistent_returns_404(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/graph/entities/nonexistent-id")
        assert response.status_code == 404


class TestGetEntityRelations:
    def test_returns_relations(self, client):
        cli, kg, e1, *rest = client
        response = cli.get(f"/api/graph/entities/{e1.id}/relations")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2


class TestListRelations:
    def test_returns_all_relations(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/graph/relations")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_relations_have_expected_fields(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/graph/relations")
        data = response.json()
        for rel in data:
            assert "source_entity_id" in rel
            assert "target_entity_id" in rel
            assert "type" in rel
            assert "confidence" in rel

    def test_relations_reflect_real_edges(self, client):
        cli, kg, e1, e2, e3, e4 = client
        response = cli.get("/api/graph/relations")
        data = response.json()
        triples = {(r["source_entity_id"], r["target_entity_id"], r["type"]) for r in data}
        assert (e1.id, e2.id, "works_at") in triples
        assert (e1.id, e3.id, "created") in triples
        assert (e2.id, e4.id, "depends_on") in triples

    def test_respects_limit(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/graph/relations?limit=1")
        assert response.status_code == 200
        assert len(response.json()) == 1


class TestGraphFull:
    def test_returns_nodes_and_edges(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/graph/full")
        assert response.status_code == 200
        data = response.json()
        assert len(data["nodes"]) == 4
        assert len(data["edges"]) == 3

    def test_node_fields(self, client):
        cli, kg, *rest = client
        data = cli.get("/api/graph/full").json()
        for node in data["nodes"]:
            for field in (
                "id",
                "name",
                "type",
                "aliases",
                "properties",
                "confidence",
                "source_note_ids",
            ):
                assert field in node

    def test_edge_fields(self, client):
        cli, kg, *rest = client
        data = cli.get("/api/graph/full").json()
        for edge in data["edges"]:
            assert set(edge) == {"source_entity_id", "target_entity_id", "type", "confidence"}

    def test_edges_reflect_real_edges(self, client):
        cli, kg, e1, e2, e3, e4 = client
        data = cli.get("/api/graph/full").json()
        triples = {(e["source_entity_id"], e["target_entity_id"], e["type"]) for e in data["edges"]}
        assert (e1.id, e2.id, "works_at") in triples
        assert (e1.id, e3.id, "created") in triples
        assert (e2.id, e4.id, "depends_on") in triples

    def test_respects_limits(self, client):
        cli, kg, *rest = client
        assert len(cli.get("/api/graph/full?limit=1").json()["nodes"]) == 1
        assert len(cli.get("/api/graph/full?relation_limit=1").json()["edges"]) == 1

    def test_limit_bounds(self, client):
        cli, kg, *rest = client
        assert cli.get("/api/graph/full?limit=0").status_code == 422
        assert cli.get("/api/graph/full?limit=99999").status_code == 422


class TestFindPaths:
    def test_returns_paths(self, client):
        cli, kg, e1, e2, e3, e4 = client
        response = cli.get(f"/api/graph/paths?source={e1.id}&target={e4.id}")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert isinstance(data["paths"], list)

    def test_direct_path(self, client):
        cli, kg, e1, e2, e3, e4 = client
        response = cli.get(f"/api/graph/paths?source={e1.id}&target={e2.id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["paths"]) >= 1
        path = data["paths"][0]
        assert len(path) >= 2
        assert path[0]["id"] == e1.id
        assert path[-1]["id"] == e2.id


class TestGraphStats:
    def test_returns_counts(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/graph/stats")
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "relations" in data
        assert "types" in data

    def test_counts_match_data(self, client):
        cli, kg, *rest = client
        response = cli.get("/api/graph/stats")
        data = response.json()
        assert data["entities"] == 4
        assert data["relations"] == 3
        assert data["types"]["Person"] == 2
