"""Integration test pinning the reembed swap contract (FR-C5.5).

The reembed pipeline (``src/knowledge/reembed``) rebuilds stale vectors into a
NEW fingerprint-named collection, atomically swaps the active collection in
``app_settings``/``app.state``, then drops the old collection. This test
exercises the exact public-API contract reembed relies on — constructor,
``add``, ``get_embeddings``, ``search``, ``delete_self`` — end-to-end against
the REAL numpy-backed EmbeddingStore, without any LLM. If the store rewrite
changes this contract, this test breaks.

Flow (mirrors ``_reembed_worker`` steps 2 + 5, minus the embedder):
1. Old collection ``notes__<oldfp>`` serving queries, seeded with notes.
2. New collection ``notes__<newfp>`` created via the public constructor.
3. Copy loop: vectors read from the old store via ``get_embeddings`` and
   written to the new store via ``add``.
4. Atomic swap (production swaps ``app.state`` + persists the collection
   name in ``app_settings`` — here we just hold the new store reference).
5. ``old.delete_self()`` — old collection emptied and its directory removed,
   new collection persists and serves both notes.
"""

from __future__ import annotations

import os

from src.knowledge.embeddings import EmbeddingStore

DIM = 4

# Note id -> (text, embedding). Vectors are not colinear so search ordering is
# deterministic: the query vector matches n1 exactly (score 1.0) and n2 at
# cosine 0.6.
NOTES: dict[str, tuple[str, list[float]]] = {
    "n1": ("Alpha note.", [1.0, 0.0, 0.0, 0.0]),
    "n2": ("Beta note.", [0.6, 0.8, 0.0, 0.0]),
}


def test_reembed_swap_contract_against_numpy_store(tmp_path):
    db_path = str(tmp_path)

    # 1. Old active collection, built under the previous fingerprint.
    old = EmbeddingStore(db_path, collection_name="notes__oldfp", dimension=DIM)
    for note_id, (text, embedding) in NOTES.items():
        old.add(note_id=note_id, text=text, embedding=embedding)
    assert old.count() == 2

    # 2. New collection under the target fingerprint — exactly what
    #    _reembed_worker constructs via the public API.
    new = EmbeddingStore(db_path, collection_name="notes__newfp", dimension=DIM)

    # 3. Copy loop: vectors come from the old store via get_embeddings
    #    (production re-embeds via the LLM provider; same store calls).
    copied = old.get_embeddings(list(NOTES))
    assert set(copied) == set(NOTES)
    for note_id, embedding in copied.items():
        new.add(note_id=note_id, text=NOTES[note_id][0], embedding=embedding)
    assert new.count() == 2

    # 4. Swap: production persists pointers in app_settings then swaps
    #    app.state references. Here the `new` reference stands in for the
    #    swapped-in store.

    # 5. Post-swap cleanup: drop the old collection.
    old.delete_self()

    # Old store is empty and its on-disk directory is gone...
    assert old.count() == 0
    assert not os.path.isdir(os.path.join(db_path, "notes__oldfp"))

    # ...while the new collection persists and serves both notes.
    assert new.count() == 2
    assert os.path.isdir(os.path.join(db_path, "notes__newfp"))
    results = new.search([1.0, 0.0, 0.0, 0.0], limit=10)
    assert {r["note_id"] for r in results} == {"n1", "n2"}
    assert results[0]["note_id"] == "n1"  # exact cosine match ranks first

    # The old store's data can't come back: its backing state was reset and
    # the swap contract guarantees no lingering on-disk artifacts.
    assert old.get_embeddings(["n1", "n2"]) == {}
