"""FR-B2 verification: SparseRetriever rebuild coalescing + staleness behavior.

Pins the existing semantics of ``src/retrieval/sparse.py``:

- ``mark_dirty()`` only flags the cache; the actual Tantivy rebuild is deferred
  to the next ``search()``, so a burst of ``mark_dirty()`` calls (e.g. one per
  fact extraction in a turn batch) triggers at most ONE ``_build_index`` run
  (SC-5: <=1 rebuild per turn batch).
- A rebuild on the next search makes newly written note files visible, so a
  dirty retriever never serves a stale index.
- The double-checked rebuild under ``self._lock`` builds exactly once even
  when concurrent searches race on a cold cache.

Test-only task: no source changes accompany this file.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.retrieval.sparse import SparseRetriever


def _write_base_notes(notes_dir) -> None:
    """Create the 3-note corpus used across tests (same shape as test_sparse.py)."""
    (notes_dir / "python_basics.md").write_text(
        "# Python Basics\n\n"
        "Python is a programming language. "
        "Python supports object oriented programming and functional programming. "
        "Variables in Python are dynamically typed.",
        encoding="utf-8",
    )
    (notes_dir / "ml_intro.md").write_text(
        "# Machine Learning Introduction\n\n"
        "Machine learning is a subset of artificial intelligence. "
        "Deep learning uses neural networks with many layers. "
        "Training data is used to fit models.",
        encoding="utf-8",
    )
    (notes_dir / "cooking_tips.md").write_text(
        "# Cooking Tips\n\n"
        "Always season your food with salt and pepper. "
        "A hot pan is essential for searing meat. "
        "Fresh herbs add great flavor to dishes.",
        encoding="utf-8",
    )


def _counting_build_index(monkeypatch, calls: dict[str, int]):
    """Wrap the real ``_build_index`` in a counter; returns the wrapper via fixture-style setattr."""
    from src.retrieval import sparse as sparse_mod

    original = sparse_mod._build_index

    def _counting(directory):
        calls["count"] += 1
        return original(directory)

    monkeypatch.setattr(sparse_mod, "_build_index", _counting)


def test_mark_dirty_burst_coalesces_into_single_rebuild(tmp_path, monkeypatch) -> None:
    """N consecutive mark_dirty() calls trigger exactly ONE rebuild on the next search."""
    _write_base_notes(tmp_path)
    calls = {"count": 0}
    _counting_build_index(monkeypatch, calls)

    retriever = SparseRetriever(notes_dir=str(tmp_path))
    for _ in range(5):
        retriever.mark_dirty()

    results = retriever.search("python programming")
    assert len(results) > 0
    assert results[0].note_id == "python_basics"
    assert calls["count"] == 1

    # Subsequent searches on the clean cache rebuild nothing.
    retriever.search("machine learning")
    retriever.search("cooking")
    assert calls["count"] == 1


def test_rebuild_on_next_search_reflects_new_note(tmp_path) -> None:
    """After mark_dirty(), the next search sees a note file written since the last build."""
    _write_base_notes(tmp_path)

    retriever = SparseRetriever(notes_dir=str(tmp_path))
    retriever.mark_dirty()
    retriever.search("python")  # prime the cache

    new_id = "11111111-1111-1111-1111-111111111111"
    (tmp_path / f"{new_id}.md").write_text(
        "# Kubernetes\n\nKubernetes orchestrates containers with zorptastic scaling.",
        encoding="utf-8",
    )

    # Cached index predates the write: the unique token is not visible yet.
    assert retriever.search("zorptastic") == []

    retriever.mark_dirty()
    results = retriever.search("zorptastic")
    assert len(results) > 0
    assert results[0].note_id == new_id


def test_concurrent_searches_on_cold_cache_build_once(tmp_path, monkeypatch) -> None:
    """Racing searches on a fresh retriever build exactly once and agree on results."""
    _write_base_notes(tmp_path)
    calls = {"count": 0}
    _counting_build_index(monkeypatch, calls)

    retriever = SparseRetriever(notes_dir=str(tmp_path))  # cold: index not built yet

    with ThreadPoolExecutor(max_workers=5) as pool:
        outputs = list(pool.map(lambda _: retriever.search("python programming"), range(5)))

    assert all(isinstance(o, list) for o in outputs)
    assert all(len(o) > 0 for o in outputs)
    assert {o[0].note_id for o in outputs} == {"python_basics"}
    assert calls["count"] == 1
