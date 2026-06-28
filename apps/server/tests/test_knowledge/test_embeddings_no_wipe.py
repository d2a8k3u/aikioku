"""EmbeddingStore must never wipe a collection on dimension mismatch.

A dimension/model change is handled by reembedding into a new fingerprinted
collection and atomically swapping (see src.knowledge.reembed), so an existing
collection under a fixed name is always adopted as-is, never silently dropped.
"""

from __future__ import annotations

import os
import tempfile

from src.knowledge.embeddings import EmbeddingStore


def test_dimension_mismatch_does_not_wipe():
    path = os.path.join(tempfile.mkdtemp(), "chroma")
    s = EmbeddingStore(db_path=path, collection_name="ctest", dimension=3)
    s.add("n1", "hello", [0.1, 0.2, 0.3])
    assert s.count() == 1

    # Reopen with a DIFFERENT dimension — data must survive (adopt, not wipe).
    s2 = EmbeddingStore(db_path=path, collection_name="ctest", dimension=8)
    assert s2.count() == 1
    assert s2.collection_name == "ctest"


def test_delete_self_removes_collection():
    path = os.path.join(tempfile.mkdtemp(), "chroma")
    s = EmbeddingStore(db_path=path, collection_name="ctest", dimension=3)
    s.add("n1", "hi", [0.1, 0.2, 0.3])
    s.delete_self()

    s2 = EmbeddingStore(db_path=path, collection_name="ctest", dimension=3)
    assert s2.count() == 0
