"""Tests for SparseRetriever using Tantivy BM25."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.retrieval.search_result import SearchResult
from src.retrieval.sparse import SparseRetriever


@pytest.fixture
def notes_dir() -> str:
    """Create a temporary directory with sample .md files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Note about Python
        with open(os.path.join(tmpdir, "python_basics.md"), "w") as f:
            f.write(
                "# Python Basics\n\n"
                "Python is a programming language. "
                "Python supports object oriented programming and functional programming. "
                "Variables in Python are dynamically typed."
            )

        # Note about machine learning
        with open(os.path.join(tmpdir, "ml_intro.md"), "w") as f:
            f.write(
                "# Machine Learning Introduction\n\n"
                "Machine learning is a subset of artificial intelligence. "
                "Deep learning uses neural networks with many layers. "
                "Training data is used to fit models."
            )

        # Note about cooking
        with open(os.path.join(tmpdir, "cooking_tips.md"), "w") as f:
            f.write(
                "# Cooking Tips\n\n"
                "Always season your food with salt and pepper. "
                "A hot pan is essential for searing meat. "
                "Fresh herbs add great flavor to dishes."
            )

        yield tmpdir


def test_search_finds_relevant_note(notes_dir: str) -> None:
    retriever = SparseRetriever(notes_dir=notes_dir)
    results = retriever.search("python programming")

    assert len(results) > 0
    assert results[0].note_id == "python_basics"
    assert results[0].source == "sparse"
    assert results[0].score > 0


def test_search_respects_limit(notes_dir: str) -> None:
    retriever = SparseRetriever(notes_dir=notes_dir)
    results = retriever.search("the and is", limit=2)

    assert len(results) <= 2


def test_search_returns_empty_for_no_matches(notes_dir: str) -> None:
    retriever = SparseRetriever(notes_dir=notes_dir)
    results = retriever.search("xyznonexistent12345")

    assert results == []


def test_search_results_are_sorted_by_score_desc(notes_dir: str) -> None:
    retriever = SparseRetriever(notes_dir=notes_dir)
    results = retriever.search("python programming language")

    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_search_returns_search_result_objects(notes_dir: str) -> None:
    retriever = SparseRetriever(notes_dir=notes_dir)
    results = retriever.search("python")

    assert all(isinstance(r, SearchResult) for r in results)


def test_search_returns_snippets(notes_dir: str) -> None:
    retriever = SparseRetriever(notes_dir=notes_dir)
    results = retriever.search("python programming")

    for r in results:
        assert isinstance(r.snippet, str)
        assert len(r.snippet) > 0


def test_search_across_multiple_files(notes_dir: str) -> None:
    retriever = SparseRetriever(notes_dir=notes_dir)
    results = retriever.search("and")

    # Common word should match many docs, limit=20 by default
    assert len(results) >= 1


def test_empty_notes_dir() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        retriever = SparseRetriever(notes_dir=tmpdir)
        results = retriever.search("anything")
        assert results == []


# ---------------------------------------------------------------------------
# Index caching — build once, rebuild only when dirty.
# ---------------------------------------------------------------------------


def test_index_built_once_across_repeated_searches(notes_dir: str, monkeypatch) -> None:
    """Repeated searches reuse the cached index (build runs exactly once)."""
    from src.retrieval import sparse as sparse_mod

    calls = {"count": 0}
    original = sparse_mod._build_index

    def _counting(directory):
        calls["count"] += 1
        return original(directory)

    monkeypatch.setattr(sparse_mod, "_build_index", _counting)

    retriever = SparseRetriever(notes_dir=notes_dir)
    retriever.search("python")
    retriever.search("machine learning")
    retriever.search("cooking")

    assert calls["count"] == 1


def test_mark_dirty_triggers_rebuild(notes_dir: str, monkeypatch) -> None:
    """After mark_dirty(), the next search rebuilds the index."""
    from src.retrieval import sparse as sparse_mod

    calls = {"count": 0}
    original = sparse_mod._build_index

    def _counting(directory):
        calls["count"] += 1
        return original(directory)

    monkeypatch.setattr(sparse_mod, "_build_index", _counting)

    retriever = SparseRetriever(notes_dir=notes_dir)
    retriever.search("python")
    assert calls["count"] == 1

    retriever.mark_dirty()
    retriever.search("python")
    assert calls["count"] == 2


def test_new_note_searchable_after_mark_dirty(notes_dir: str) -> None:
    """A note added after the first search is found once the index is invalidated."""
    retriever = SparseRetriever(notes_dir=notes_dir)
    # Prime the cache.
    retriever.search("python")

    new_id = "11111111-1111-1111-1111-111111111111"
    with open(os.path.join(notes_dir, f"{new_id}.md"), "w") as f:
        f.write("# Kubernetes\n\nKubernetes orchestrates containers with zorptastic scaling.")

    # Stale cache: the new token is not yet visible.
    assert retriever.search("zorptastic") == []

    retriever.mark_dirty()
    results = retriever.search("zorptastic")
    assert len(results) > 0
    assert results[0].note_id == new_id


def test_invalidate_is_alias_for_mark_dirty(notes_dir: str, monkeypatch) -> None:
    """invalidate() behaves the same as mark_dirty()."""
    from src.retrieval import sparse as sparse_mod

    calls = {"count": 0}
    original = sparse_mod._build_index

    def _counting(directory):
        calls["count"] += 1
        return original(directory)

    monkeypatch.setattr(sparse_mod, "_build_index", _counting)

    retriever = SparseRetriever(notes_dir=notes_dir)
    retriever.search("python")
    retriever.invalidate()
    retriever.search("python")

    assert calls["count"] == 2


def test_note_id_is_bare_stem(notes_dir: str) -> None:
    """note_id is the bare file stem (UUID), not a path."""
    retriever = SparseRetriever(notes_dir=notes_dir)
    results = retriever.search("python programming")

    assert results[0].note_id == "python_basics"


def test_concurrent_searches_do_not_error(notes_dir: str) -> None:
    """Concurrent searches across threads complete without raising."""
    from concurrent.futures import ThreadPoolExecutor

    retriever = SparseRetriever(notes_dir=notes_dir)

    def _run(_i):
        return retriever.search("python programming language")

    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(_run, range(32)))

    assert all(isinstance(o, list) for o in outputs)
    assert any(len(o) > 0 for o in outputs)
