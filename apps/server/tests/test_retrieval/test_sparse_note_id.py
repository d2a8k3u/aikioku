"""Tests that SparseRetriever emits bare-UUID note_ids (no .md suffix).

Notes are stored by NoteStore as ``{UUID}.md``. The dense and graph
retrievers emit the bare UUID, so sparse must do the same or fusion never
matches across retrievers and ``.md``-suffixed citations fail to resolve via
the notes API (which parses note_id as a UUID).
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.retrieval.sparse import SparseRetriever


@pytest.fixture
def uuid_notes_dir(tmp_path) -> tuple[str, str]:
    """Create a notes dir with a single ``{uuid}.md`` file.

    Returns the directory path and the bare UUID.
    """
    note_uuid = str(uuid.uuid4())
    note_path = os.path.join(str(tmp_path), f"{note_uuid}.md")
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(
            "# Python\n\n"
            "Python is a programming language created by a Dutch engineer. "
            "It is widely used for scripting and data science."
        )
    return str(tmp_path), note_uuid


def test_search_emits_bare_uuid_note_id(uuid_notes_dir) -> None:
    notes_dir, note_uuid = uuid_notes_dir
    retriever = SparseRetriever(notes_dir=notes_dir)

    results = retriever.search("python programming")

    assert len(results) > 0
    for r in results:
        assert not r.note_id.endswith(".md")
    assert results[0].note_id == note_uuid
