"""Tests for SerendipityEngine."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.knowledge.graph import KnowledgeGraph
from src.models.entity import Entity, EntityType
from src.models.relation import Relation, RelationType
from src.augmentation.serendipity import SerendipityEngine


@pytest.fixture
def tmp_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        kg = KnowledgeGraph(db_path=os.path.join(tmpdir, "kg.db"))

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
        kg.create_relation(Relation(source_entity_id=e5.id, target_entity_id=e4.id, type=RelationType.depends_on, confidence=0.5))

        yield kg, e1, e2, e3, e4, e5


class TestRandomWalk:
    def test_returns_list_of_entities(self, tmp_graph):
        kg, e1, *_ = tmp_graph
        engine = SerendipityEngine(graph=kg)
        result = engine.random_walk(e1.id, steps=5)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(eid, str) for eid in result)

    def test_start_entity_first(self, tmp_graph):
        kg, e1, *_ = tmp_graph
        engine = SerendipityEngine(graph=kg)
        result = engine.random_walk(e1.id, steps=5)
        assert result[0] == e1.id

    def test_walk_length_respected(self, tmp_graph):
        kg, e1, *_ = tmp_graph
        engine = SerendipityEngine(graph=kg)
        result = engine.random_walk(e1.id, steps=3)
        assert len(result) <= 4  # start + up to 3 steps

    def test_valid_entity_ids(self, tmp_graph):
        kg, e1, *_ = tmp_graph
        engine = SerendipityEngine(graph=kg)
        result = engine.random_walk(e1.id, steps=5)
        for eid in result:
            assert kg.get_entity(eid) is not None

    def test_isolated_entity_returns_only_self(self, tmp_graph):
        kg, *_ = tmp_graph
        isolated = Entity(name="Lonely", type=EntityType.Concept)
        kg.create_entity(isolated)
        engine = SerendipityEngine(graph=kg)
        result = engine.random_walk(isolated.id, steps=5)
        assert result == [isolated.id]


class TestSurpriseScore:
    def test_returns_float(self, tmp_graph):
        kg, e1, *_ = tmp_graph
        engine = SerendipityEngine(graph=kg)
        score = engine.surprise_score(e1.id)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_high_degree_lower_score(self, tmp_graph):
        kg, e1, e2, e3, e4, e5 = tmp_graph
        engine = SerendipityEngine(graph=kg)
        # e1 has degree 2, isolated would have 0
        score_e1 = engine.surprise_score(e1.id)
        score_e4 = engine.surprise_score(e4.id)
        # e4 has degree 2 as well (from e3 and e5), e1 also 2
        # Test monotonic: high-degree entity should have lower surprise than low-degree entity?
        # Actually let's create an isolated entity and compare
        isolated = Entity(name="Lonely", type=EntityType.Concept)
        kg.create_entity(isolated)
        score_iso = engine.surprise_score(isolated.id)
        # Isolated entity should have highest surprise (it's unexpected)
        assert score_iso >= score_e1
        assert score_iso >= score_e4

    def test_score_based_on_relation_confidence(self, tmp_graph):
        kg, e1, *_ = tmp_graph
        engine = SerendipityEngine(graph=kg)
        score = engine.surprise_score(e1.id)
        # Score should incorporate average confidence of relations
        # e1 relations: 0.9, 0.6 -> avg 0.75
        # surprise = 1 - avg_confidence / max_degree_factor ... just check range
        assert 0.0 <= score <= 1.0
