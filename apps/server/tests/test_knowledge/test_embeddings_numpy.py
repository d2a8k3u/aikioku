"""Parity tests for the numpy-backed EmbeddingStore (RAM optimization, Package C).

These tests pin the EmbeddingStore contract — chunking semantics,
dedup-by-note_id max scoring, persistence format, crash recovery, and
thread safety — so the chroma and numpy backends are interchangeable.
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from src.knowledge.embeddings import EmbeddingStore

_DIM = 4


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "emb")


@pytest.fixture
def store(db_path):
    return EmbeddingStore(db_path=db_path, dimension=_DIM)


def _chunky_note_text(n_chunks: int = 2) -> str:
    # Each chunk ~2000 chars of sentence-like text so _semantic_chunks splits.
    sentence = "The quick brown fox jumps over the lazy dog. "
    chunk = sentence * 60  # ~2700 chars
    return (chunk + "\n") * (n_chunks - 1) + chunk


# --------------------------------------------------------------------------- #
# Chunking semantics
# --------------------------------------------------------------------------- #


class TestChunking:
    def test_multi_chunk_note_creates_multiple_docs_with_same_embedding(self, store):
        text = _chunky_note_text(2)
        store.add("note-1", text, [0.1, 0.2, 0.3, 0.4])
        assert store.count() >= 2

        embs = store.get_embeddings(["note-1"])
        assert embs["note-1"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_short_text_is_single_doc(self, store):
        store.add("note-1", "Short text.", [0.1, 0.2, 0.3, 0.4])
        assert store.count() == 1

    def test_add_document_does_not_chunk(self, store):
        store.add_document(
            "d1",
            "Long text. " * 1000,
            [0.1, 0.2, 0.3, 0.4],
            metadata={"note_id": "n1"},
        )
        assert store.count() == 1

    def test_add_document_adds_text_to_metadata(self, store):
        store.add_document("d1", "hello", [0.1, 0.2, 0.3, 0.4], metadata={"note_id": "n1"})
        with open(os.path.join(store._dir, "docs.json")) as f:
            docs = json.load(f)
        assert docs[0]["metadata"]["text"] == "hello"


# --------------------------------------------------------------------------- #
# Search: cosine parity, dedup, limit
# --------------------------------------------------------------------------- #


class TestSearch:
    def test_cosine_scores_and_ordering(self, store):
        store.add("note-a", "a", [1.0, 0.0, 0.0, 0.0])
        store.add("note-b", "b", [0.5, 0.5, 0.0, 0.0])
        results = store.search([1.0, 0.0, 0.0, 0.0], limit=2)
        assert [r["note_id"] for r in results] == ["note-a", "note-b"]
        assert results[0]["score"] == pytest.approx(1.0)
        assert results[1]["score"] == pytest.approx(0.5 / (0.5**2 + 0.5**2) ** 0.5)

    def test_dedup_by_note_id_keeps_max_score(self, store):
        store.add_document("n1#c0#sw0", "bad window", [0.0, 1.0, 0.0, 0.0], {"note_id": "n1"})
        store.add_document("n1#c0#sw1", "best window", [0.99, 0.01, 0.0, 0.0], {"note_id": "n1"})
        results = store.search([1.0, 0.0, 0.0, 0.0], limit=5)
        assert len(results) == 1
        assert results[0]["note_id"] == "n1"
        assert results[0]["text"] == "best window"
        assert results[0]["score"] == pytest.approx(0.99995, abs=1e-3)

    def test_score_exact_cosine_value(self, store):
        store.add("n1", "x", [0.0, 0.0, 3.0, 4.0])
        results = store.search([0.0, 0.0, 1.0, 0.0], limit=1)
        assert results[0]["score"] == pytest.approx(3.0 / 5.0)

    def test_negative_cosine_docs_are_skipped(self, store):
        store.add("neg", "negative", [-1.0, 0.0, 0.0, 0.0])
        store.add("pos", "positive", [1.0, 0.0, 0.0, 0.0])
        results = store.search([1.0, 0.0, 0.0, 0.0], limit=10)
        assert [r["note_id"] for r in results] == ["pos"]

    def test_zero_vector_query_returns_empty(self, store):
        store.add("n1", "x", [1.0, 0.0, 0.0, 0.0])
        assert store.search([0.0, 0.0, 0.0, 0.0]) == []

    def test_empty_store_returns_empty(self, store):
        assert store.search([1.0, 0.0, 0.0, 0.0]) == []

    def test_limit_applies_to_notes_not_docs(self, store):
        for i in range(6):
            store.add(f"n{i}", "content", [float(i) / 10.0, 0.5, 0.5, 0.5])
        results = store.search([0.5, 0.5, 0.5, 0.5], limit=3)
        assert len(results) == 3

    def test_score_is_float_and_non_negative(self, store):
        store.add("n1", "x", [1.0, 0.0, 0.0, 0.0])
        results = store.search([0.0, 1.0, 0.0, 0.0], limit=1)
        assert results == []
        store.add("n2", "y", [1.0, 0.0, 0.0, 0.0])
        results = store.search([1.0, 0.0, 0.0, 0.0], limit=1)
        assert isinstance(results[0]["score"], float)
        assert results[0]["score"] >= 0.0


# --------------------------------------------------------------------------- #
# Delete / count / get_embeddings
# --------------------------------------------------------------------------- #


class TestDeleteCountGet:
    def test_delete_removes_all_chunks_of_note(self, store):
        store.add("note-1", _chunky_note_text(2), [0.1, 0.2, 0.3, 0.4])
        store.add("note-2", "other", [0.5, 0.5, 0.5, 0.5])
        n_chunks = store.count() - 1
        assert n_chunks >= 2

        store.delete("note-1")
        assert store.count() == 1

        results = store.search([0.1, 0.2, 0.3, 0.4], limit=10)
        assert [r["note_id"] for r in results] == ["note-2"]

    def test_delete_nonexistent_does_not_raise(self, store):
        store.delete("missing")
        assert store.count() == 0

    def test_count_reports_docs(self, store):
        assert store.count() == 0
        store.add_document("d1", "t", [1.0, 0.0, 0.0, 0.0], {"note_id": "n1"})
        store.add_document("d2", "t", [0.0, 1.0, 0.0, 0.0], {"note_id": "n1"})
        assert store.count() == 2

    def test_get_embeddings_first_doc_per_note(self, store):
        store.add_document("n1#0", "first", [1.0, 0.0, 0.0, 0.0], {"note_id": "n1"})
        store.add_document("n1#1", "second", [0.0, 1.0, 0.0, 0.0], {"note_id": "n1"})
        store.add_document("n2#0", "other", [0.0, 0.0, 1.0, 0.0], {"note_id": "n2"})
        embs = store.get_embeddings(["n1", "n2", "missing"])
        assert set(embs.keys()) == {"n1", "n2"}
        assert embs["n1"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
        assert embs["n2"] == pytest.approx([0.0, 0.0, 1.0, 0.0])


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_persistence_files_created(self, store):
        store.add("n1", "hello", [0.1, 0.2, 0.3, 0.4])
        assert os.path.isfile(os.path.join(store._dir, "vectors.npz"))
        assert os.path.isfile(os.path.join(store._dir, "docs.json"))

    def test_round_trip_new_instance_identical_search(self, db_path):
        s1 = EmbeddingStore(db_path=db_path, dimension=_DIM)
        s1.add("n1", "alpha", [1.0, 0.0, 0.0, 0.0])
        s1.add_document("n2#c0", "beta", [0.9, 0.1, 0.0, 0.0], {"note_id": "n2"})
        before = s1.search([1.0, 0.0, 0.0, 0.0], limit=10)

        s2 = EmbeddingStore(db_path=db_path, dimension=_DIM)
        assert s2.count() == s1.count()
        assert s2.search([1.0, 0.0, 0.0, 0.0], limit=10) == before
        assert s2.get_embeddings(["n1"])["n1"] == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_collection_isolation(self, db_path):
        a = EmbeddingStore(db_path=db_path, collection_name="a", dimension=_DIM)
        b = EmbeddingStore(db_path=db_path, collection_name="b", dimension=_DIM)
        a.add("n1", "in a", [1.0, 0.0, 0.0, 0.0])
        assert b.count() == 0

    def test_orphan_tmp_promoted_on_load(self, db_path):
        s = EmbeddingStore(db_path=db_path, dimension=_DIM)
        s.add("n1", "hello", [0.1, 0.2, 0.3, 0.4])

        # Simulate a crash between tmp write and replace: only finals removed.
        tmp_vec = os.path.join(s._dir, "vectors.tmp")
        tmp_docs = os.path.join(s._dir, "docs.tmp")
        os.replace(os.path.join(s._dir, "vectors.npz"), tmp_vec)
        os.replace(os.path.join(s._dir, "docs.json"), tmp_docs)

        s2 = EmbeddingStore(db_path=db_path, dimension=_DIM)
        assert s2.count() == 1
        assert os.path.isfile(os.path.join(s._dir, "vectors.npz"))
        assert not os.path.exists(tmp_vec)
        assert not os.path.exists(tmp_docs)

    def test_stale_tmp_discarded_when_finals_valid(self, db_path):
        s = EmbeddingStore(db_path=db_path, dimension=_DIM)
        s.add("n1", "hello", [0.1, 0.2, 0.3, 0.4])
        with open(os.path.join(s._dir, "vectors.tmp"), "wb") as f:
            f.write(b"junk")
        with open(os.path.join(s._dir, "docs.tmp"), "w") as f:
            f.write("junk")

        s2 = EmbeddingStore(db_path=db_path, dimension=_DIM)
        assert s2.count() == 1
        assert not os.path.exists(os.path.join(s._dir, "vectors.tmp"))
        assert not os.path.exists(os.path.join(s._dir, "docs.tmp"))

    def test_delete_self_removes_dir_and_empties_store(self, db_path):
        s = EmbeddingStore(db_path=db_path, dimension=_DIM)
        s.add("n1", "hello", [0.1, 0.2, 0.3, 0.4])
        coll_dir = s._dir
        assert os.path.isdir(coll_dir)

        s.delete_self()
        assert not os.path.exists(coll_dir)
        assert s.count() == 0
        assert s.search([0.1, 0.2, 0.3, 0.4]) == []

        # A fresh instance over the same path starts empty.
        s2 = EmbeddingStore(db_path=db_path, dimension=_DIM)
        assert s2.count() == 0

    def test_reopening_with_different_dimension_adopts_stored(self, db_path):
        s1 = EmbeddingStore(db_path=db_path, dimension=3)
        s1.add("n1", "hello", [0.1, 0.2, 0.3])
        assert s1.count() == 1

        s2 = EmbeddingStore(db_path=db_path, dimension=8)
        assert s2.count() == 1
        assert s2._dimension == 3
        results = s2.search([0.1, 0.2, 0.3], limit=1)
        assert results[0]["note_id"] == "n1"

    def test_delete_self_clears_shared_state(self, db_path):
        # Mirrors the no-wipe contract: two stores over one collection must
        # not double-report after one deletes itself.
        s1 = EmbeddingStore(db_path=db_path, dimension=_DIM)
        s2 = EmbeddingStore(db_path=db_path, dimension=_DIM)
        assert s1._matrix is s2._matrix
        s1.add("n1", "x", [1.0, 0.0, 0.0, 0.0])
        assert s2.count() == 1
        s1.delete_self()
        assert s2.count() == 0


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


class TestThreadSafety:
    def test_concurrent_searches_and_writer(self, db_path):
        store = EmbeddingStore(db_path=db_path, dimension=_DIM)
        for i in range(20):
            store.add_document(f"n{i}#c0", f"doc {i}", [1.0, 0.0, 0.0, 0.0], {"note_id": f"n{i}"})

        errors: list[BaseException] = []
        barrier = threading.Barrier(6)

        def _reader() -> None:
            barrier.wait()
            try:
                for _ in range(30):
                    results = store.search([1.0, 0.0, 0.0, 0.0], limit=5)
                    assert len(results) == 5
                    assert all(0.0 <= r["score"] <= 1.0 for r in results)
            except BaseException as e:  # noqa: BLE001 - pragma: no cover (failure plumbing)
                errors.append(e)

        def _writer() -> None:
            barrier.wait()
            try:
                for i in range(10):
                    store.add_document(
                        f"w{i}#c0", f"writer {i}", [1.0, 0.0, 0.0, 0.0], {"note_id": f"w{i}"}
                    )
            except BaseException as e:  # noqa: BLE001 - pragma: no cover (failure plumbing)
                errors.append(e)

        threads = [threading.Thread(target=_reader) for _ in range(5)] + [
            threading.Thread(target=_writer)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == []
        assert store.count() == 30
        assert len(store.search([1.0, 0.0, 0.0, 0.0], limit=50)) == 30


# --------------------------------------------------------------------------- #
# State management helpers
# --------------------------------------------------------------------------- #


class TestStateManagement:
    def test_mutation_does_not_break_prior_snapshot(self, store):
        # Appends may only write into unused capacity rows; rows a prior
        # search snapshot can see must never change (design §4.3:
        # linearizable reads, writers never mutate visible data in place).
        store.add_document("d1", "one", [1.0, 0.0, 0.0, 0.0], {"note_id": "n1"})
        snapshot = store._matrix
        used_rows = snapshot[: store.count()].copy()
        store.add_document("d2", "two", [0.0, 1.0, 0.0, 0.0], {"note_id": "n2"})
        store.delete("n1")
        assert (snapshot[:1] == used_rows).all()
        assert store.count() == 1
