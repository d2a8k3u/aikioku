"""Tests for ConnectionDiscovery."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.graph import KnowledgeGraph
from src.models.entity import Entity, EntityType
from src.models.relation import Relation, RelationType
from src.reasoning.connections import ConnectionDiscovery


@pytest.fixture
def tmp_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        kg = KnowledgeGraph(db_path=os.path.join(tmpdir, "kg.db"))
        emb = EmbeddingStore(db_path=os.path.join(tmpdir, "emb.db"))

        e1 = Entity(name="Alice", type=EntityType.Person)
        e2 = Entity(name="Bob", type=EntityType.Person)
        e3 = Entity(name="Acme Corp", type=EntityType.Organization)
        e4 = Entity(name="Project X", type=EntityType.Project)
        e5 = Entity(name="Charlie", type=EntityType.Person)

        for e in (e1, e2, e3, e4, e5):
            kg.create_entity(e)

        kg.create_relation(Relation(source_entity_id=e1.id, target_entity_id=e2.id, type=RelationType.related_to, confidence=0.9))
        kg.create_relation(Relation(source_entity_id=e2.id, target_entity_id=e3.id, type=RelationType.works_at, confidence=0.8))
        kg.create_relation(Relation(source_entity_id=e3.id, target_entity_id=e4.id, type=RelationType.created, confidence=0.7))
        kg.create_relation(Relation(source_entity_id=e1.id, target_entity_id=e5.id, type=RelationType.related_to, confidence=0.6))

        # Seed embeddings for source notes linked to entities
        # Use simple orthogonal-ish vectors so similarity is deterministic
        emb.add("n1", "Alice note", [1.0, 0.0, 0.0])
        emb.add("n2", "Bob note", [0.9, 0.1, 0.0])
        emb.add("n3", "Acme note", [0.0, 1.0, 0.0])
        emb.add("n4", "Project X note", [0.0, 0.9, 0.1])
        emb.add("n5", "Charlie note", [0.0, 0.0, 1.0])

        # Link entities to notes
        e1.source_note_ids = ["n1"]
        e2.source_note_ids = ["n2"]
        e3.source_note_ids = ["n3"]
        e4.source_note_ids = ["n4"]
        e5.source_note_ids = ["n5"]
        for e in (e1, e2, e3, e4, e5):
            kg.update_entity(e)

        yield kg, emb, e1, e2, e3, e4, e5


class TestConnectionDiscovery:
    def test_discover_returns_connections(self, tmp_graph):
        kg, emb, e1, *_ = tmp_graph
        cd = ConnectionDiscovery(graph=kg, embedding_store=emb)
        conns = cd.discover_connections(e1.id, max_distance=3)
        assert isinstance(conns, list)
        assert len(conns) > 0

    def test_connection_has_path_strength_explanation(self, tmp_graph):
        kg, emb, e1, *_ = tmp_graph
        cd = ConnectionDiscovery(graph=kg, embedding_store=emb)
        conns = cd.discover_connections(e1.id, max_distance=3)
        for c in conns:
            assert isinstance(c.path, list)
            assert len(c.path) >= 2
            assert isinstance(c.strength, float)
            assert 0.0 <= c.strength <= 1.0
            assert isinstance(c.explanation, str)
            assert c.explanation

    def test_direct_connection_stronger_than_indirect(self, tmp_graph):
        kg, emb, e1, e2, e3, e4, e5 = tmp_graph
        cd = ConnectionDiscovery(graph=kg, embedding_store=emb)
        conns = cd.discover_connections(e1.id, max_distance=3)
        # Find direct vs indirect
        direct = [c for c in conns if c.path[-1] == e2.id]
        indirect = [c for c in conns if c.path[-1] == e4.id]
        assert direct
        assert indirect
        # Direct should have higher strength than indirect
        assert direct[0].strength > indirect[0].strength

    def test_max_distance_respected(self, tmp_graph):
        kg, emb, e1, *_ = tmp_graph
        cd = ConnectionDiscovery(graph=kg, embedding_store=emb)
        conns = cd.discover_connections(e1.id, max_distance=1)
        # Should only find direct neighbors
        assert all(len(c.path) <= 2 for c in conns)

    def test_explanation_contains_entity_names(self, tmp_graph):
        kg, emb, e1, e2, e3, e4, e5 = tmp_graph
        cd = ConnectionDiscovery(graph=kg, embedding_store=emb)
        conns = cd.discover_connections(e1.id, max_distance=3)
        for c in conns:
            end_entity = kg.get_entity(c.path[-1])
            assert end_entity is not None
            assert end_entity.name in c.explanation

    def test_no_duplicate_targets(self, tmp_graph):
        kg, emb, e1, *_ = tmp_graph
        cd = ConnectionDiscovery(graph=kg, embedding_store=emb)
        conns = cd.discover_connections(e1.id, max_distance=3)
        targets = [c.path[-1] for c in conns]
        assert len(targets) == len(set(targets))

    def test_start_entity_not_in_results(self, tmp_graph):
        kg, emb, e1, *_ = tmp_graph
        cd = ConnectionDiscovery(graph=kg, embedding_store=emb)
        conns = cd.discover_connections(e1.id, max_distance=3)
        for c in conns:
            assert c.path[-1] != e1.id
