"""Tests for hidden-notes consolidation: similarity, clustering, merge, CRUD.

Uses real Kuzu DB and ChromaDB in temp dirs, following the test_pipeline.py pattern.
LLMProvider is mocked.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.graph import KnowledgeGraph
from src.llm.base import LLMProvider
from src.models.note import Note
from src.storage.note_store import NoteStore


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        yield {
            "notes_dir": str(base / "notes"),
            "kuzu": str(base / "kuzu.db"),
            "chroma": str(base / "chroma"),
        }


@pytest.fixture
def note_store(tmp_dirs):
    return NoteStore(tmp_dirs["notes_dir"])


@pytest.fixture
def knowledge_graph(tmp_dirs):
    return KnowledgeGraph(db_path=tmp_dirs["kuzu"])


@pytest.fixture
def embedding_store(tmp_dirs):
    return EmbeddingStore(db_path=tmp_dirs["chroma"])


@pytest.fixture
def mock_llm_provider():
    provider = AsyncMock(spec=LLMProvider)
    provider.complete = AsyncMock(return_value="[]")
    provider.embed = AsyncMock(return_value=[0.1] * 384)
    provider.is_available = MagicMock(return_value=True)
    return provider


@pytest.fixture
def mock_cost_tracker():
    tracker = MagicMock()
    tracker.is_exhausted = MagicMock(return_value=False)
    tracker.remaining = MagicMock(return_value=10.0)
    tracker.daily_budget = 5.0
    return tracker


@pytest.fixture
def app_state(
    note_store,
    knowledge_graph,
    embedding_store,
    mock_llm_provider,
    mock_cost_tracker,
):
    state = SimpleNamespace(
        note_store=note_store,
        knowledge_graph=knowledge_graph,
        embedding_store=embedding_store,
        llm_provider=mock_llm_provider,
        embedding_provider=mock_llm_provider,
        cost_tracker=mock_cost_tracker,
        hybrid_fusion=None,
    )
    return SimpleNamespace(state=state)


@pytest.fixture
def entity_extraction_response():
    return json.dumps(
        [
            {
                "name": "Python",
                "type": "Concept",
                "aliases": ["Python language"],
                "confidence": 0.95,
            },
            {
                "name": "Guido van Rossum",
                "type": "Person",
                "aliases": ["Guido"],
                "confidence": 0.90,
            },
        ]
    )


def _make_hidden_note(
    nid: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    links: list[str] | None = None,
) -> Note:
    frontmatter: dict[str, list[str]] = {}
    if tags:
        frontmatter["tags"] = tags
    return Note(
        id=nid,
        title=title,
        content=content,
        frontmatter=frontmatter,
        links=links or [],
        path=f"/notes/hidden/{nid}.md",
        source_type="hidden",
    )


def _make_visible_note(nid: str, title: str, content: str) -> Note:
    return Note(
        id=nid,
        title=title,
        content=content,
        path=f"/notes/{nid}.md",
        source_type="note",
    )


def _make_merge_response(title: str, content: str) -> str:
    return json.dumps({"title": title, "content": content})


# --------------------------------------------------------------------------- #
# Tests: fetch_candidates
# --------------------------------------------------------------------------- #


class TestFetchCandidates:
    def test_fetch_candidates_only_hidden(self, note_store):
        note_store.create(_make_hidden_note("h1", "Hidden 1", "content 1"))
        note_store.create(_make_visible_note("v1", "Visible 1", "content 2"))
        note_store.create(_make_hidden_note("h2", "Hidden 2", "content 3"))

        from src.processing.consolidation import fetch_candidates

        result = fetch_candidates(note_store, limit=100)
        ids = {n.id for n in result}
        assert "h1" in ids
        assert "h2" in ids
        assert "v1" not in ids

    def test_fetch_candidates_respects_limit(self, note_store):
        for i in range(10):
            note_store.create(_make_hidden_note(f"h{i}", f"Note {i}", f"content {i}"))

        from src.processing.consolidation import fetch_candidates

        result = fetch_candidates(note_store, limit=5)
        assert len(result) <= 5


# --------------------------------------------------------------------------- #
# Tests: build_entity_map
# --------------------------------------------------------------------------- #


class TestBuildEntityMap:
    def test_build_entity_map_single_scan(self, knowledge_graph):
        from src.models.entity import Entity, EntityType

        e1 = Entity(name="Python", type=EntityType.Concept, source_note_ids=["note-a", "note-b"])
        e2 = Entity(name="Rust", type=EntityType.Concept, source_note_ids=["note-b"])
        knowledge_graph.create_entity(e1)
        knowledge_graph.create_entity(e2)

        from src.processing.consolidation import build_entity_map

        entity_map = build_entity_map(knowledge_graph)
        assert "note-a" in entity_map
        assert "note-b" in entity_map
        assert entity_map["note-a"] == {e1.id}
        assert entity_map["note-b"] == {e1.id, e2.id}


# --------------------------------------------------------------------------- #
# Tests: detect_similar_pairs
# --------------------------------------------------------------------------- #


class TestDetectSimilarPairs:
    def test_detect_similar_pairs_high_similarity(self):
        from src.processing.consolidation import detect_similar_pairs

        n1 = _make_hidden_note("n1", "A", "x")
        n2 = _make_hidden_note("n2", "B", "x")
        emb = {"n1": [1.0, 0.0], "n2": [1.0, 0.0]}
        entity_map = {"n1": {"e1"}, "n2": {"e1"}}
        pairs = detect_similar_pairs([n1, n2], emb, entity_map, threshold=0.85)
        assert len(pairs) == 1
        assert pairs[0][2] >= 0.85

    def test_detect_similar_pairs_low_similarity(self):
        from src.processing.consolidation import detect_similar_pairs

        n1 = _make_hidden_note("n1", "A", "x")
        n2 = _make_hidden_note("n2", "B", "y")
        emb = {"n1": [1.0, 0.0], "n2": [0.0, 1.0]}
        entity_map = {"n1": {"e1"}, "n2": {"e1"}}
        pairs = detect_similar_pairs([n1, n2], emb, entity_map, threshold=0.85)
        assert len(pairs) == 0

    def test_detect_similar_pairs_no_shared_entity(self):
        from src.processing.consolidation import detect_similar_pairs

        n1 = _make_hidden_note("n1", "A", "x")
        n2 = _make_hidden_note("n2", "B", "x")
        emb = {"n1": [1.0, 0.0], "n2": [1.0, 0.0]}
        entity_map = {"n1": {"e1"}, "n2": {"e2"}}
        pairs = detect_similar_pairs([n1, n2], emb, entity_map, threshold=0.85)
        assert len(pairs) == 0

    def test_detect_similar_pairs_missing_embedding(self):
        from src.processing.consolidation import detect_similar_pairs

        n1 = _make_hidden_note("n1", "A", "x")
        n2 = _make_hidden_note("n2", "B", "x")
        n3 = _make_hidden_note("n3", "C", "x")
        emb = {"n1": [1.0, 0.0], "n2": [1.0, 0.0]}
        entity_map = {"n1": {"e1"}, "n2": {"e1"}, "n3": {"e1"}}
        pairs = detect_similar_pairs([n1, n2, n3], emb, entity_map, threshold=0.85)
        pair_ids = {(p[0].id, p[1].id) for p in pairs}
        assert ("n1", "n2") in pair_ids or ("n2", "n1") in pair_ids
        for p in pairs:
            assert "n3" not in (p[0].id, p[1].id)


# --------------------------------------------------------------------------- #
# Tests: cluster_pairs
# --------------------------------------------------------------------------- #


class TestClusterPairs:
    def test_cluster_connected_components(self):
        from src.processing.consolidation import cluster_pairs

        n1 = _make_hidden_note("n1", "A", "x")
        n2 = _make_hidden_note("n2", "B", "x")
        n3 = _make_hidden_note("n3", "C", "x")
        notes = [n1, n2, n3]
        pairs = [
            (n1, n2, 0.9),
            (n2, n3, 0.88),
        ]
        clusters = cluster_pairs(pairs, notes, min_size=2, cap=5)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3
        cluster_ids = {n.id for n in clusters[0]}
        assert cluster_ids == {"n1", "n2", "n3"}

    def test_cluster_min_size_filter(self):
        from src.processing.consolidation import cluster_pairs

        n1 = _make_hidden_note("n1", "A", "x")
        n2 = _make_hidden_note("n2", "B", "x")
        n3 = _make_hidden_note("n3", "C", "x")
        notes = [n1, n2, n3]
        pairs = [(n1, n2, 0.9)]
        clusters = cluster_pairs(pairs, notes, min_size=2, cap=5)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2
        assert "n3" not in {n.id for n in clusters[0]}

    def test_cluster_size_cap_split(self):
        from src.processing.consolidation import cluster_pairs

        notes = [_make_hidden_note(f"n{i}", f"Note {i}", "x") for i in range(8)]
        pairs = []
        for i in range(len(notes) - 1):
            pairs.append((notes[i], notes[i + 1], 0.9 - i * 0.001))

        clusters = cluster_pairs(pairs, notes, min_size=2, cap=3)
        assert len(clusters) >= 2
        for cluster in clusters:
            assert len(cluster) <= 3


# --------------------------------------------------------------------------- #
# Tests: merge_cluster
# --------------------------------------------------------------------------- #


class TestMergeCluster:
    async def test_merge_cluster_valid_output(self, app_state, mock_llm_provider):
        from src.processing.consolidation import merge_cluster

        n1 = _make_hidden_note("n1", "Python Basics", "Python is great.", tags=["lang"])
        n2 = _make_hidden_note("n2", "Python Intro", "Python is a language.", tags=["intro"])
        mock_llm_provider.complete = AsyncMock(
            return_value=_make_merge_response("Merged Title", "Merged content")
        )
        result = await merge_cluster(app_state, [n1, n2])
        assert result is not None
        assert result.title == "Merged Title"
        assert result.content == "Merged content"
        assert result.source_type == "hidden"

    async def test_merge_cluster_invalid_json(self, app_state, mock_llm_provider):
        from src.processing.consolidation import merge_cluster

        n1 = _make_hidden_note("n1", "A", "content a")
        n2 = _make_hidden_note("n2", "B", "content b")
        mock_llm_provider.complete = AsyncMock(return_value="this is not json at all")
        result = await merge_cluster(app_state, [n1, n2])
        assert result is None

    async def test_merge_cluster_empty_content(self, app_state, mock_llm_provider):
        from src.processing.consolidation import merge_cluster

        n1 = _make_hidden_note("n1", "A", "content a")
        mock_llm_provider.complete = AsyncMock(return_value=_make_merge_response("Has Title", ""))
        result = await merge_cluster(app_state, [n1])
        assert result is None


# --------------------------------------------------------------------------- #
# Tests: create + delete order
# --------------------------------------------------------------------------- #


class TestCreateDeleteOrder:
    def test_create_then_delete_order(self, note_store, embedding_store, knowledge_graph):
        from src.processing.consolidation import create_merged_note, delete_originals

        merged = _make_hidden_note("merged-1", "Merged", "merged content")
        orig1 = _make_hidden_note("orig-1", "Original 1", "content 1")
        orig2 = _make_hidden_note("orig-2", "Original 2", "content 2")
        note_store.create(orig1)
        note_store.create(orig2)

        create_merged_note(note_store, merged)
        assert note_store.get("merged-1") is not None
        assert note_store.get("orig-1") is not None
        assert note_store.get("orig-2") is not None

        delete_originals(note_store, embedding_store, knowledge_graph, [orig1, orig2])
        assert note_store.get("orig-1") is None
        assert note_store.get("orig-2") is None
        assert note_store.get("merged-1") is not None

    def test_delete_originals_purges_artifacts(
        self, note_store, embedding_store, knowledge_graph, mock_llm_provider
    ):
        from src.processing.consolidation import delete_originals

        n1 = _make_hidden_note("del-1", "Note 1", "content for deletion")
        n2 = _make_hidden_note("del-2", "Note 2", "content for deletion")
        note_store.create(n1)
        note_store.create(n2)
        embedding_store.add(n1.id, n1.content, [0.1] * 384)
        embedding_store.add(n2.id, n2.content, [0.2] * 384)
        assert embedding_store.get_embeddings([n1.id, n2.id]) != {}

        delete_originals(note_store, embedding_store, knowledge_graph, [n1, n2])
        assert embedding_store.get_embeddings([n1.id, n2.id]) == {}
        assert note_store.get(n1.id) is None
        assert note_store.get(n2.id) is None


# --------------------------------------------------------------------------- #
# Tests: reprocess_merged_note
# --------------------------------------------------------------------------- #


class TestReprocessMergedNote:
    async def test_reprocess_merged_note(
        self, app_state, mock_llm_provider, embedding_store, entity_extraction_response
    ):
        from src.processing.consolidation import reprocess_merged_note

        merged = _make_hidden_note("merged-rep", "Merged", "Python and Guido van Rossum.")
        mock_llm_provider.complete = AsyncMock(return_value=entity_extraction_response)
        mock_llm_provider.embed = AsyncMock(return_value=[0.5] * 384)

        await reprocess_merged_note(app_state, merged)
        assert embedding_store.get_embeddings([merged.id]) != {}


# --------------------------------------------------------------------------- #
# Tests: run_consolidation end-to-end
# --------------------------------------------------------------------------- #


class TestRunConsolidationEndToEnd:
    async def test_run_consolidation_end_to_end(
        self, app_state, note_store, embedding_store, knowledge_graph, mock_llm_provider
    ):
        from src.processing.consolidation import run_consolidation

        n1 = _make_hidden_note("c1", "Python Basics", "Python is a language by Guido van Rossum.")
        n2 = _make_hidden_note("c2", "Python Intro", "Python is a language by Guido van Rossum.")
        n3 = _make_hidden_note("c3", "Rust Notes", "Rust is a systems programming language.")
        note_store.create(n1)
        note_store.create(n2)
        note_store.create(n3)

        emb_dim = 384
        embedding_store.add(n1.id, n1.content, [1.0] + [0.0] * (emb_dim - 1))
        embedding_store.add(n2.id, n2.content, [1.0] + [0.0] * (emb_dim - 1))
        embedding_store.add(n3.id, n3.content, [0.0, 1.0] + [0.0] * (emb_dim - 2))

        from src.models.entity import Entity, EntityType

        shared_entity = Entity(
            name="Python", type=EntityType.Concept, source_note_ids=[n1.id, n2.id]
        )
        knowledge_graph.create_entity(shared_entity)
        rust_entity = Entity(name="Rust", type=EntityType.Concept, source_note_ids=[n3.id])
        knowledge_graph.create_entity(rust_entity)

        merge_json = _make_merge_response(
            "Consolidated Python Notes",
            "Python is a language created by Guido van Rossum.",
        )
        entity_json = json.dumps(
            [{"name": "Python", "type": "Concept", "aliases": [], "confidence": 0.95}]
        )

        call_count = [0]

        async def _side_complete(prompt, system="", **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return merge_json
            return entity_json

        mock_llm_provider.complete = AsyncMock(side_effect=_side_complete)
        mock_llm_provider.embed = AsyncMock(return_value=[0.3] * 384)

        result = await run_consolidation(app_state)
        assert result["candidates"] == 3
        assert result["clusters"] >= 1
        assert result["merged"] >= 1
        assert result["deleted"] >= 2

        hidden_notes = note_store.list(source_type="hidden", limit=100)
        titles = {n.title for n in hidden_notes}
        assert "Consolidated Python Notes" in titles
        assert "Rust Notes" in titles

        assert note_store.get("c1") is None
        assert note_store.get("c2") is None
        assert note_store.get("c3") is not None

    async def test_run_consolidation_no_candidates(self, app_state):
        from src.processing.consolidation import run_consolidation

        result = await run_consolidation(app_state)
        assert result["candidates"] == 0
        assert result["clusters"] == 0
        assert result["merged"] == 0

    async def test_run_consolidation_no_embeddings(self, app_state, note_store, embedding_store):
        from src.processing.consolidation import run_consolidation

        n1 = _make_hidden_note("ne1", "Note 1", "content 1")
        n2 = _make_hidden_note("ne2", "Note 2", "content 2")
        note_store.create(n1)
        note_store.create(n2)
        assert embedding_store.get_embeddings([n1.id, n2.id]) == {}

        result = await run_consolidation(app_state)
        assert result["candidates"] == 2
        assert result["clusters"] == 0
        assert result["merged"] == 0
        assert note_store.get("ne1") is not None
        assert note_store.get("ne2") is not None

    async def test_run_consolidation_budget_exceeded_mid_run(
        self, app_state, note_store, embedding_store, knowledge_graph, mock_llm_provider
    ):
        from src.llm.router import BudgetExceededError
        from src.processing.consolidation import run_consolidation

        cluster_a = [
            _make_hidden_note(f"ba{i}", f"Note A{i}", "same content here") for i in range(2)
        ]
        cluster_b = [
            _make_hidden_note(f"bb{i}", f"Note B{i}", "same content here") for i in range(2)
        ]
        for n in cluster_a + cluster_b:
            note_store.create(n)
            embedding_store.add(n.id, n.content, [1.0] * 384)

        from src.models.entity import Entity, EntityType

        e_a = Entity(
            name="EntityA",
            type=EntityType.Concept,
            source_note_ids=[cluster_a[0].id, cluster_a[1].id],
        )
        e_b = Entity(
            name="EntityB",
            type=EntityType.Concept,
            source_note_ids=[cluster_b[0].id, cluster_b[1].id],
        )
        knowledge_graph.create_entity(e_a)
        knowledge_graph.create_entity(e_b)

        merge_call_count = [0]

        async def _side_complete(prompt, system="", **kw):
            if system and "consolidation assistant" in system:
                merge_call_count[0] += 1
                if merge_call_count[0] >= 2:
                    raise BudgetExceededError(daily_budget=5.0, today_cost=5.0)
                return _make_merge_response("Merged A", "merged content A")
            return json.dumps(
                [{"name": "EntityA", "type": "Concept", "aliases": [], "confidence": 0.9}]
            )

        mock_llm_provider.complete = AsyncMock(side_effect=_side_complete)
        mock_llm_provider.embed = AsyncMock(return_value=[0.1] * 384)

        result = await run_consolidation(app_state)
        assert result["merged"] <= 1

        all_originals = cluster_a + cluster_b
        unmerged_originals = [n for n in all_originals if note_store.get(n.id) is not None]
        assert len(unmerged_originals) == 2

    async def test_run_consolidation_cluster_failure_continues(
        self, app_state, note_store, embedding_store, knowledge_graph, mock_llm_provider
    ):
        from src.processing.consolidation import run_consolidation

        fa = [_make_hidden_note(f"fa{i}", f"Fail A{i}", "same content here") for i in range(2)]
        fb = [_make_hidden_note(f"fb{i}", f"Fail B{i}", "same content here") for i in range(2)]
        for n in fa + fb:
            note_store.create(n)
            embedding_store.add(n.id, n.content, [1.0] * 384)

        from src.models.entity import Entity, EntityType

        e_a = Entity(name="FailA", type=EntityType.Concept, source_note_ids=[fa[0].id, fa[1].id])
        e_b = Entity(name="FailB", type=EntityType.Concept, source_note_ids=[fb[0].id, fb[1].id])
        knowledge_graph.create_entity(e_a)
        knowledge_graph.create_entity(e_b)

        merge_call_count = [0]

        async def _side_complete(prompt, system="", **kw):
            if system and "consolidation assistant" in system:
                merge_call_count[0] += 1
                if merge_call_count[0] == 1:
                    raise RuntimeError("boom on first cluster")
                return _make_merge_response("Merged B", "merged content B")
            return json.dumps(
                [{"name": "FailB", "type": "Concept", "aliases": [], "confidence": 0.9}]
            )

        mock_llm_provider.complete = AsyncMock(side_effect=_side_complete)
        mock_llm_provider.embed = AsyncMock(return_value=[0.1] * 384)

        result = await run_consolidation(app_state)
        assert result["merged"] == 1
        assert result["skipped"] >= 1

        all_originals = fa + fb
        surviving = [n for n in all_originals if note_store.get(n.id) is not None]
        deleted = [n for n in all_originals if note_store.get(n.id) is None]
        assert len(surviving) == 2
        assert len(deleted) == 2


# --------------------------------------------------------------------------- #
# Tests: visible notes safety + idempotency + orphaned artifacts
# --------------------------------------------------------------------------- #


class TestSafetyInvariants:
    async def test_visible_notes_never_consolidated(
        self, app_state, note_store, embedding_store, knowledge_graph, mock_llm_provider
    ):
        from src.processing.consolidation import run_consolidation

        vis = _make_visible_note("vis-1", "Visible", "same content here")
        hid = _make_hidden_note("hid-1", "Hidden 1", "same content here")
        hid2 = _make_hidden_note("hid-2", "Hidden 2", "same content here")
        note_store.create(vis)
        note_store.create(hid)
        note_store.create(hid2)
        embedding_store.add(vis.id, vis.content, [1.0] * 384)
        embedding_store.add(hid.id, hid.content, [1.0] * 384)
        embedding_store.add(hid2.id, hid2.content, [1.0] * 384)

        from src.models.entity import Entity, EntityType

        e = Entity(
            name="SharedEntity",
            type=EntityType.Concept,
            source_note_ids=[vis.id, hid.id, hid2.id],
        )
        knowledge_graph.create_entity(e)

        mock_llm_provider.complete = AsyncMock(
            return_value=_make_merge_response("Merged Hidden", "merged content")
        )
        mock_llm_provider.embed = AsyncMock(return_value=[0.1] * 384)

        await run_consolidation(app_state)
        assert note_store.get("vis-1") is not None
        assert note_store.get("vis-1").source_type == "note"

    async def test_idempotent_across_runs(
        self, app_state, note_store, embedding_store, knowledge_graph, mock_llm_provider
    ):
        from src.processing.consolidation import run_consolidation

        n1 = _make_hidden_note("idem-1", "Dup A", "same content here")
        n2 = _make_hidden_note("idem-2", "Dup B", "same content here")
        note_store.create(n1)
        note_store.create(n2)
        embedding_store.add(n1.id, n1.content, [1.0] * 384)
        embedding_store.add(n2.id, n2.content, [1.0] * 384)

        from src.models.entity import Entity, EntityType

        e = Entity(name="IdemEntity", type=EntityType.Concept, source_note_ids=[n1.id, n2.id])
        knowledge_graph.create_entity(e)

        merge_json = _make_merge_response("Merged Idem", "merged content")
        entity_json = json.dumps(
            [{"name": "IdemEntity", "type": "Concept", "aliases": [], "confidence": 0.9}]
        )
        call_count = [0]

        async def _side_complete(prompt, system="", **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return merge_json
            return entity_json

        mock_llm_provider.complete = AsyncMock(side_effect=_side_complete)
        mock_llm_provider.embed = AsyncMock(return_value=[0.1] * 384)

        result1 = await run_consolidation(app_state)
        assert result1["merged"] >= 1

        result2 = await run_consolidation(app_state)
        assert result2["merged"] == 0

        hidden_notes = note_store.list(source_type="hidden", limit=100)
        assert len(hidden_notes) >= 1
        assert note_store.get("idem-1") is None
        assert note_store.get("idem-2") is None

    async def test_no_orphaned_artifacts(
        self, app_state, note_store, embedding_store, knowledge_graph, mock_llm_provider
    ):
        from src.processing.consolidation import run_consolidation

        n1 = _make_hidden_note("orph-1", "Orphan A", "same content here")
        n2 = _make_hidden_note("orph-2", "Orphan B", "same content here")
        note_store.create(n1)
        note_store.create(n2)
        embedding_store.add(n1.id, n1.content, [1.0] * 384)
        embedding_store.add(n2.id, n2.content, [1.0] * 384)

        from src.models.entity import Entity, EntityType

        e = Entity(name="OrphEntity", type=EntityType.Concept, source_note_ids=[n1.id, n2.id])
        knowledge_graph.create_entity(e)

        merge_json = _make_merge_response("Merged Orphan", "merged content")
        entity_json = json.dumps(
            [{"name": "NewEntity", "type": "Concept", "aliases": [], "confidence": 0.9}]
        )
        call_count = [0]

        async def _side_complete(prompt, system="", **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return merge_json
            return entity_json

        mock_llm_provider.complete = AsyncMock(side_effect=_side_complete)
        mock_llm_provider.embed = AsyncMock(return_value=[0.1] * 384)

        await run_consolidation(app_state)

        assert embedding_store.get_embeddings([n1.id]) == {}
        assert embedding_store.get_embeddings([n2.id]) == {}

        remaining_entities = knowledge_graph.find_entities(limit=100_000)
        for ent in remaining_entities:
            assert n1.id not in ent.source_note_ids
            assert n2.id not in ent.source_note_ids
