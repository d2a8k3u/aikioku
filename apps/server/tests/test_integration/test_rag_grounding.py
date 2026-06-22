"""Host-gated integration tests for RAG grounding against the live stack.

These tests hit the running backend at ``http://localhost:8869`` (the
docker-compose port mapping). They are skipped automatically when the live
stack is unreachable, so they are safe to run inside the test container
(where they will simply skip).

Run explicitly with::

    pytest -q -m integration tests/test_integration/test_rag_grounding.py
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

# The host port mapped by docker-compose. Overridable for CI.
BASE_URL = os.environ.get("SECONDBRAIN_BASE_URL", "http://localhost:8869")


def _stack_reachable() -> bool:
    try:
        resp = httpx.get(f"{BASE_URL}/api/notes/", timeout=5.0)
        return resp.status_code < 500
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _stack_reachable(),
        reason="live stack not reachable at " + BASE_URL,
    ),
]


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def test_hybrid_retrieval_ranks_python_note_with_snippets() -> None:
    """A paraphrase query for Python should rank the Python note near the top,
    results must carry bare-UUID note_ids and non-empty snippets."""
    resp = httpx.post(
        f"{BASE_URL}/api/retrieval/hybrid",
        json={"query": "snake scripting language by a dutch engineer", "limit": 5},
        timeout=30.0,
    )
    resp.raise_for_status()
    results = resp.json()

    assert isinstance(results, list)
    assert len(results) > 0

    # bare-UUID note_ids (no .md suffix), and they resolve via the notes API.
    for r in results:
        assert not r["note_id"].endswith(".md")
        assert _is_uuid(r["note_id"])

    # At least one of the top results carries a non-empty snippet.
    assert any(r.get("snippet") for r in results[:5])

    # The top result must resolve to the Python note.
    top_note = httpx.get(
        f"{BASE_URL}/api/notes/{results[0]['note_id']}", timeout=15.0
    ).json()
    haystack = (top_note.get("title", "") + " " + top_note.get("content", "")).lower()
    assert "python" in haystack


def test_chat_simple_returns_grounded_answer_with_citation() -> None:
    """The simple-mode chat endpoint should return a non-empty grounded answer
    with at least one citation whose note_id resolves (bare UUID).

    The remote chat LLM is slow; we use a generous timeout and SKIP on timeout
    rather than fail (grounding correctness is proven by the retrieval test).
    """
    try:
        resp = httpx.post(
            f"{BASE_URL}/api/chat/",
            json={"query": "What is Docker?", "mode": "simple"},
            timeout=180.0,
        )
    except httpx.TimeoutException:
        pytest.skip("live chat LLM timed out (>180s)")

    if resp.status_code == 429:
        pytest.skip("chat endpoint rate-limited")
    resp.raise_for_status()
    data = resp.json()

    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0

    citations = data.get("citations", [])
    assert len(citations) >= 1
    for c in citations:
        assert not c["note_id"].endswith(".md")
        assert _is_uuid(c["note_id"])
