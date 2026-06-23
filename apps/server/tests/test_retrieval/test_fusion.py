"""Tests for HybridFusion."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.retrieval.fusion import HybridFusion
from src.retrieval.search_result import SearchResult


def _make_results(
    source: str, note_ids: list[str], scores: list[float] | None = None
) -> list[SearchResult]:
    """Helper to build a list of SearchResult for a given source."""
    if scores is None:
        scores = [1.0 - i * 0.1 for i in range(len(note_ids))]
    return [SearchResult(note_id=nid, score=sc, source=source) for nid, sc in zip(note_ids, scores)]


@pytest.fixture
def mock_dense():
    m = MagicMock()
    m.search = AsyncMock(return_value=_make_results("dense", ["d1", "d2", "d3"]))
    return m


@pytest.fixture
def mock_sparse():
    m = MagicMock()
    m.search = MagicMock(return_value=_make_results("sparse", ["s1", "s2", "s3"]))
    return m


@pytest.fixture
def mock_graph():
    m = MagicMock()
    m.search = MagicMock(return_value=_make_results("graph", ["g1", "g2", "g3"]))
    return m


@pytest.fixture
def fusion(mock_dense, mock_sparse, mock_graph):
    return HybridFusion(dense=mock_dense, sparse=mock_sparse, graph=mock_graph)


@pytest.mark.asyncio
async def test_fuse_combines_results_from_all_three(fusion):
    """_fuse should combine results from dense, sparse, and graph retrievers."""
    results = {
        "dense": _make_results("dense", ["d1", "d2"]),
        "sparse": _make_results("sparse", ["s1", "s2"]),
        "graph": _make_results("graph", ["g1", "g2"]),
    }
    fused = fusion._fuse(results)
    note_ids = {r.note_id for r in fused}
    assert note_ids == {"d1", "d2", "s1", "s2", "g1", "g2"}


@pytest.mark.asyncio
async def test_fuse_deduplicates_by_note_id(fusion):
    """_fuse should deduplicate by note_id, keeping the highest fused score."""
    results = {
        "dense": _make_results("dense", ["n1", "n2"]),
        "sparse": _make_results("sparse", ["n1", "n3"]),
        "graph": _make_results("graph", ["n1", "n4"]),
    }
    fused = fusion._fuse(results)
    note_ids = [r.note_id for r in fused]
    # n1 appears 3 times in input but should appear once in output
    assert note_ids.count("n1") == 1
    assert set(note_ids) == {"n1", "n2", "n3", "n4"}


@pytest.mark.asyncio
async def test_fuse_sorts_by_fused_score_descending(fusion):
    """_fuse should sort results by fused score in descending order."""
    results = {
        "dense": _make_results("dense", ["a", "b", "c"]),
        "sparse": _make_results("sparse", ["d", "e", "f"]),
        "graph": _make_results("graph", ["g", "h", "i"]),
    }
    fused = fusion._fuse(results)
    scores = [r.score for r in fused]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_custom_weights():
    """Custom weights should affect the fused scores."""
    dense = MagicMock()
    dense.search = AsyncMock(return_value=[])
    sparse = MagicMock()
    sparse.search = MagicMock(return_value=[])
    graph = MagicMock()
    graph.search = MagicMock(return_value=[])

    fusion_default = HybridFusion(dense=dense, sparse=sparse, graph=graph)
    fusion_custom = HybridFusion(
        dense=dense,
        sparse=sparse,
        graph=graph,
        weights={"dense": 0.7, "sparse": 0.2, "graph": 0.1},
    )

    results = {
        "dense": _make_results("dense", ["n1"]),
        "sparse": _make_results("sparse", ["n2"]),
        "graph": _make_results("graph", ["n3"]),
    }

    default_fused = fusion_default._fuse(results)
    custom_fused = fusion_custom._fuse(results)

    # With default weights (0.4, 0.3, 0.3), dense rank-0 gets 0.4/(0+60) ≈ 0.00667
    # With custom weights (0.7, 0.2, 0.1), dense rank-0 gets 0.7/(0+60) ≈ 0.01167
    # So n1 should have a higher score with custom weights
    default_n1 = next(r for r in default_fused if r.note_id == "n1")
    custom_n1 = next(r for r in custom_fused if r.note_id == "n1")
    assert custom_n1.score > default_n1.score


@pytest.mark.asyncio
async def test_empty_results_from_all_returns_empty(fusion):
    """_fuse should return an empty list when all retrievers return empty."""
    results = {
        "dense": [],
        "sparse": [],
        "graph": [],
    }
    fused = fusion._fuse(results)
    assert fused == []


@pytest.mark.asyncio
async def test_search_runs_all_retrievers_and_returns_top(
    fusion, mock_dense, mock_sparse, mock_graph
):
    """search() should run all retrievers and return fused results up to limit."""
    result = await fusion.search("test query", limit=5)
    mock_dense.search.assert_called_once_with("test query", limit=20)
    mock_sparse.search.assert_called_once_with("test query", limit=20)
    mock_graph.search.assert_called_once_with("test query", limit=20)
    assert isinstance(result, list)
    assert len(result) <= 5


@pytest.mark.asyncio
async def test_fuse_accumulates_cross_retriever_consensus(fusion):
    """A note found by all three retrievers must outscore one found by only one.

    RRF rewards cross-retriever consensus by SUMMING the per-retriever
    contributions rather than taking the max.
    """
    # A is found at rank 0 by sparse (0.3) AND graph (0.3); B at rank 0 by the
    # higher-weighted dense (0.4) only.
    #   MAX fusion  -> A=0.3/60, B=0.4/60  => B wins (wrong).
    #   SUM fusion  -> A=0.6/60, B=0.4/60  => A wins (consensus rewarded).
    results = {
        "dense": _make_results("dense", ["B"]),
        "sparse": _make_results("sparse", ["A"]),
        "graph": _make_results("graph", ["A"]),
    }
    fused = fusion._fuse(results)
    score_a = next(r.score for r in fused if r.note_id == "A")
    score_b = next(r.score for r in fused if r.note_id == "B")

    assert score_a > score_b


@pytest.mark.asyncio
async def test_fuse_prefers_non_empty_snippet(fusion):
    """When one source has an empty snippet and another has content for the same
    note, the representative result must keep the non-empty snippet."""
    results = {
        # dense at rank 0 with an EMPTY snippet (stored first)
        "dense": [SearchResult(note_id="A", score=0.9, source="dense", snippet="")],
        # sparse at rank 0 with CONTENT (seen later)
        "sparse": [SearchResult(note_id="A", score=0.8, source="sparse", snippet="real content")],
    }
    fused = fusion._fuse(results)
    rep = next(r for r in fused if r.note_id == "A")

    assert rep.snippet == "real content"


@pytest.mark.asyncio
async def test_fusion_preserves_entity_source_type(fusion):
    """A synthetic entity result (source_type='entity') from the graph retriever
    must retain source_type='entity' after RRF fusion."""
    results = {
        "graph": [
            SearchResult(
                note_id="entity:abc-123",
                score=0.9,
                source="graph",
                snippet="Entity: Some Entity",
                source_type="entity",
            )
        ],
    }
    fused = fusion._fuse(results)
    assert len(fused) == 1
    assert fused[0].note_id == "entity:abc-123"
    assert fused[0].source_type == "entity"


@pytest.mark.asyncio
async def test_fusion_handles_mixed_note_and_entity_results(fusion):
    """Dense returns note results, graph returns both note and entity results.
    After fusion, both source_types must be present and correct."""
    results = {
        "dense": [
            SearchResult(note_id="note-1", score=0.9, source="dense", source_type="note"),
        ],
        "graph": [
            SearchResult(note_id="note-2", score=0.8, source="graph", source_type="note"),
            SearchResult(
                note_id="entity:xyz",
                score=0.85,
                source="graph",
                source_type="entity",
            ),
        ],
    }
    fused = fusion._fuse(results)
    by_id = {r.note_id: r for r in fused}

    assert "note-1" in by_id
    assert "note-2" in by_id
    assert "entity:xyz" in by_id

    assert by_id["note-1"].source_type == "note"
    assert by_id["note-2"].source_type == "note"
    assert by_id["entity:xyz"].source_type == "entity"


@pytest.mark.asyncio
async def test_fusion_entity_result_does_not_collide_with_note(fusion):
    """A note result with note_id='note-1' and an entity result with
    note_id='entity:abc' must both survive fusion as separate results."""
    results = {
        "dense": [SearchResult(note_id="note-1", score=0.9, source="dense")],
        "graph": [
            SearchResult(
                note_id="entity:abc",
                score=0.85,
                source="graph",
                source_type="entity",
            )
        ],
    }
    fused = fusion._fuse(results)
    note_ids = {r.note_id for r in fused}

    assert note_ids == {"note-1", "entity:abc"}
    assert len(fused) == 2
