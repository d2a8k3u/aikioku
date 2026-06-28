"""Tests for Serendipity API endpoints."""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI TestClient with a seeded KnowledgeGraph."""
    from src.knowledge.graph import KnowledgeGraph
    from src.models.entity import Entity, EntityType
    from src.models.relation import Relation, RelationType
    from src.main import app

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_kg.db")
        kg = KnowledgeGraph(db_path=db_path)
        app.state.knowledge_graph = kg

        e1 = Entity(name="Alice", type=EntityType.Person, confidence=0.95)
        e2 = Entity(name="Bob", type=EntityType.Person, confidence=0.88)
        e3 = Entity(name="Charlie", type=EntityType.Person, confidence=0.80)

        kg.create_entity(e1)
        kg.create_entity(e2)
        kg.create_entity(e3)

        kg.create_relation(
            Relation(
                source_entity_id=e1.id,
                target_entity_id=e2.id,
                type=RelationType.works_at,
            )
        )
        kg.create_relation(
            Relation(
                source_entity_id=e2.id,
                target_entity_id=e3.id,
                type=RelationType.works_at,
            )
        )

        yield TestClient(app), e1, e2, e3


class TestRandomWalk:
    def test_returns_path(self, client):
        cli, e1, e2, e3 = client
        response = cli.post(f"/api/serendipity/walk?start_entity_id={e1.id}&steps=3")
        assert response.status_code == 200
        data = response.json()
        assert "path" in data
        assert isinstance(data["path"], list)
        assert data["path"][0] == e1.id

    def test_path_length_within_steps(self, client):
        cli, e1, e2, e3 = client
        response = cli.post(f"/api/serendipity/walk?start_entity_id={e1.id}&steps=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["path"]) <= 3  # start + up to 2 steps

    def test_missing_start_entity_returns_422(self, client):
        cli, *rest = client
        response = cli.post("/api/serendipity/walk?steps=3")
        assert response.status_code == 422


class TestSurpriseScore:
    def test_returns_score(self, client):
        cli, e1, e2, e3 = client
        response = cli.get(f"/api/serendipity/surprise?entity_id={e3.id}")
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_random_surprise_without_entity_id_returns_200(self, client):
        cli, *rest = client
        response = cli.get("/api/serendipity/surprise")
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_isolated_entity_high_score(self, client):
        cli, e1, e2, e3 = client
        # e3 has one relation (to e2), e1 has one relation (to e2)
        # Create an isolated entity
        from src.models.entity import Entity, EntityType

        isolated = Entity(name="Isolated", type=EntityType.Concept, confidence=0.90)
        app = cli.app
        app.state.knowledge_graph.create_entity(isolated)
        response = cli.get(f"/api/serendipity/surprise?entity_id={isolated.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["score"] == 1.0
