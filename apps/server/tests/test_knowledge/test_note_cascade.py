"""Tests for the delete_note cross-store cascade (src/knowledge/note_cascade.py)."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.knowledge.note_cascade import (
    prune_cooccurrence_for_entities,
    purge_note_artifacts,
    remove_note_from_graph,
    remove_note_vectors,
)
from src.models.entity import Entity, EntityType
from src.models.relation import Relation, RelationType


@pytest.fixture
def graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        from src.knowledge.graph import KnowledgeGraph

        yield KnowledgeGraph(db_path=os.path.join(tmpdir, "kg.db"))


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        from src.knowledge.embeddings import EmbeddingStore

        yield EmbeddingStore(db_path=os.path.join(tmpdir, "chroma"))


def _entity(name, note_ids, properties=None):
    return Entity(
        name=name,
        type=EntityType.Concept,
        confidence=0.9,
        source_note_ids=list(note_ids),
        properties=properties or {},
    )


def _cooccurrence(graph, a_id, b_id):
    graph.create_relation(
        Relation(
            source_entity_id=a_id,
            target_entity_id=b_id,
            type=RelationType.related_to,
            confidence=0.5,
            properties={"co_occurrence": True},
        )
    )


# --------------------------------------------------------------------------- #
# remove_note_vectors
# --------------------------------------------------------------------------- #


class TestRemoveNoteVectors:
    def test_removes_all_chunks_for_note(self, store):
        store.add("note-1", "hello world", [0.1, 0.2, 0.3])
        store.add("note-2", "other", [0.4, 0.5, 0.6])
        remove_note_vectors("note-1", store)
        results = store.search([0.1, 0.2, 0.3])
        assert all(r["note_id"] != "note-1" for r in results)
        assert any(r["note_id"] == "note-2" for r in results)

    def test_nonexistent_note_is_noop(self, store):
        store.add("note-1", "hello", [0.1, 0.2, 0.3])
        remove_note_vectors("does-not-exist", store)
        assert store.count() >= 1


# --------------------------------------------------------------------------- #
# remove_note_from_graph
# --------------------------------------------------------------------------- #


class TestRemoveNoteFromGraph:
    def test_only_note_source_entity_is_deleted(self, graph):
        ent = _entity("Solo", ["n1"])
        graph.create_entity(ent)
        result = remove_note_from_graph("n1", graph)
        assert result["deleted"] == 1
        assert result["trimmed_ids"] == []
        assert graph.get_entity(ent.id) is None

    def test_entity_with_other_note_is_trimmed_and_kept(self, graph):
        ent = _entity("Shared", ["n1", "n2"])
        graph.create_entity(ent)
        result = remove_note_from_graph("n1", graph)
        assert result["deleted"] == 0
        assert result["trimmed_ids"] == [ent.id]
        survivor = graph.get_entity(ent.id)
        assert survivor is not None
        assert survivor.source_note_ids == ["n2"]

    def test_entity_with_memory_source_is_kept(self, graph):
        ent = _entity("FromMemory", ["n1"], properties={"source_memory_ids": ["m1"]})
        graph.create_entity(ent)
        result = remove_note_from_graph("n1", graph)
        assert result["deleted"] == 0
        survivor = graph.get_entity(ent.id)
        assert survivor is not None
        assert survivor.source_note_ids == []
        assert survivor.properties.get("source_memory_ids") == ["m1"]

    def test_entity_with_conversation_turn_source_is_kept(self, graph):
        ent = _entity("FromChat", ["n1"], properties={"source_conversation_turns": ["t1"]})
        graph.create_entity(ent)
        result = remove_note_from_graph("n1", graph)
        assert result["deleted"] == 0
        assert graph.get_entity(ent.id) is not None

    def test_no_entities_for_note_is_noop(self, graph):
        graph.create_entity(_entity("Unrelated", ["other"]))
        result = remove_note_from_graph("n1", graph)
        assert result == {"deleted": 0, "trimmed_ids": []}

    def test_deleting_entity_drops_its_edges(self, graph):
        a = _entity("A", ["n1"])
        b = _entity("B", ["n1"])
        graph.create_entity(a)
        graph.create_entity(b)
        _cooccurrence(graph, a.id, b.id)
        remove_note_from_graph("n1", graph)
        assert graph.get_entity(a.id) is None
        assert graph.get_entity(b.id) is None
        assert graph.count_relations() == 0


# --------------------------------------------------------------------------- #
# prune_cooccurrence_for_entities
# --------------------------------------------------------------------------- #


class TestPruneCooccurrence:
    def test_empty_intersection_edge_removed(self, graph):
        a = _entity("A", ["n2"])
        b = _entity("B", ["n3"])
        graph.create_entity(a)
        graph.create_entity(b)
        _cooccurrence(graph, a.id, b.id)
        removed = prune_cooccurrence_for_entities([a.id, b.id], graph)
        assert removed == 1
        assert graph.count_relations() == 0

    def test_shared_note_edge_kept(self, graph):
        a = _entity("A", ["n2"])
        b = _entity("B", ["n2"])
        graph.create_entity(a)
        graph.create_entity(b)
        _cooccurrence(graph, a.id, b.id)
        removed = prune_cooccurrence_for_entities([a.id, b.id], graph)
        assert removed == 0
        assert graph.count_relations() == 1

    def test_predicate_edge_survives_cooccurrence_removal(self, graph):
        a = _entity("A", ["n2"])
        b = _entity("B", ["n3"])
        graph.create_entity(a)
        graph.create_entity(b)
        _cooccurrence(graph, a.id, b.id)
        graph.create_relation(
            Relation(
                source_entity_id=a.id,
                target_entity_id=b.id,
                type=RelationType.related_to,
                confidence=0.8,
                properties={"predicate": "works_with", "source_memory_ids": ["m1"]},
            )
        )
        removed = prune_cooccurrence_for_entities([a.id, b.id], graph)
        assert removed == 1
        survivors = [
            r
            for r in graph.get_relations(a.id)
            if r.source_entity_id == a.id and r.target_entity_id == b.id
        ]
        assert len(survivors) == 1
        assert survivors[0].properties.get("predicate") == "works_with"

    def test_duplicate_cooccurrence_edges_all_removed(self, graph):
        a = _entity("A", ["n2"])
        b = _entity("B", ["n3"])
        graph.create_entity(a)
        graph.create_entity(b)
        _cooccurrence(graph, a.id, b.id)
        _cooccurrence(graph, a.id, b.id)
        removed = prune_cooccurrence_for_entities([a.id, b.id], graph)
        assert removed == 2
        assert graph.count_relations() == 0


# --------------------------------------------------------------------------- #
# purge_note_artifacts
# --------------------------------------------------------------------------- #


class TestPurgeNoteArtifacts:
    def test_end_to_end_clears_vectors_and_graph(self, graph, store):
        store.add("n1", "hello world", [0.1, 0.2, 0.3])
        a = _entity("A", ["n1"])
        b = _entity("B", ["n1"])
        graph.create_entity(a)
        graph.create_entity(b)
        _cooccurrence(graph, a.id, b.id)

        result = purge_note_artifacts("n1", embedding_store=store, graph=graph)

        assert result["vectors"] is True
        assert result["entities_deleted"] == 2
        assert store.count() == 0
        assert graph.count_entities() == 0
        assert graph.count_relations() == 0

    def test_both_stores_none_is_noop(self):
        result = purge_note_artifacts("n1", embedding_store=None, graph=None)
        assert result["vectors"] is False
        assert result["entities_deleted"] == 0
