"""Integration tests for RAG grounding.

This file contains two categories of tests:

1. **Host-gated integration tests** — hit the running backend at
   ``http://localhost:8869`` (the docker-compose port mapping). They are
   skipped automatically when the live stack is unreachable, so they are safe
   to run inside the test container (where they will simply skip).

   Run explicitly with::

       pytest -q -m integration tests/test_integration/test_rag_grounding.py

2. **End-to-end RAG entity-retrieval tests** — use a real Kuzu graph +
   mock dense/sparse retrievers + mock LLM. These do NOT require the live
   stack and prove that conversation-derived (orphaned) entities are
   retrievable by the full RAG pipeline.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock

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


# Per-test skip for host-gated tests (instead of a module-level pytestmark so
# that the entity-retrieval tests below are NOT skipped when the stack is down).
_skip_if_stack_down = pytest.mark.skipif(
    not _stack_reachable(),
    reason="live stack not reachable at " + BASE_URL,
)


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


@pytest.mark.integration
@_skip_if_stack_down
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
    top_note = httpx.get(f"{BASE_URL}/api/notes/{results[0]['note_id']}", timeout=15.0).json()
    haystack = (top_note.get("title", "") + " " + top_note.get("content", "")).lower()
    assert "python" in haystack


@pytest.mark.integration
@_skip_if_stack_down
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


# ---------------------------------------------------------------------------
# End-to-end RAG entity-retrieval tests
# ---------------------------------------------------------------------------
# These tests do NOT require the live stack. They use a real Kuzu
# KnowledgeGraph (temp dir) + mock dense/sparse retrievers (return empty) +
# mock LLM + mock memory extractor, proving the full RAG pipeline retrieves
# orphaned conversation-derived entities.


@pytest.fixture
def tmp_db_path() -> str:
    """Provide a temporary file path for Kuzu DB, cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest.fixture
def empty_note_store(tmp_path):
    """Return a NoteStore backed by an empty temp directory."""
    from src.storage.note_store import NoteStore

    return NoteStore(str(tmp_path / "notes"))


def _make_mock_llm() -> AsyncMock:
    """Return a mock LLMProvider whose complete() returns a dummy string and
    captures the system prompt it was called with."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value="This is a dummy grounded response.")
    llm.is_available = MagicMock(return_value=True)
    return llm


def _make_mock_memory_extractor() -> MagicMock:
    """Return a mock MemoryExtractor whose extract_from_conversation returns []."""
    extractor = MagicMock()
    extractor.extract_from_conversation = AsyncMock(return_value=[])
    return extractor


def _make_empty_async_retriever() -> AsyncMock:
    """Return a mock dense retriever whose async search() returns an empty list.

    DenseRetriever.search is a coroutine; HybridFusion awaits it via
    _with_timeout. AsyncMock.search returns a coroutine that resolves to [].
    """
    retriever = AsyncMock()
    retriever.search = AsyncMock(return_value=[])
    return retriever


def _make_empty_sync_retriever() -> MagicMock:
    """Return a mock sparse retriever whose sync search() returns an empty list.

    SparseRetriever.search is synchronous; HybridFusion runs it in a thread
    executor. MagicMock.search returns [] directly.
    """
    retriever = MagicMock()
    retriever.search = MagicMock(return_value=[])
    return retriever


@pytest.mark.asyncio
async def test_rag_retrieves_conversation_derived_entity(tmp_db_path, empty_note_store):
    """End-to-end: orphaned entities (empty source_note_ids, from conversation)
    are retrievable by the full RAG pipeline and appear in the LLM system prompt
    and citations.

    Proves the Entity-RAG Retrieval Gap fix work together:
    - GraphRetriever produces synthetic SearchResults for orphaned entities.
    - HybridFusion preserves source_type="entity" through RRF fusion.
    - RAGGenerator handles entity-source results (skip note_store.get, [E{i}]
      labeling, entity citations).
    """
    from src.knowledge.graph import KnowledgeGraph
    from src.models.entity import Entity, EntityType
    from src.models.relation import Relation, RelationType
    from src.reasoning.rag import RAGGenerator
    from src.retrieval.fusion import HybridFusion
    from src.retrieval.graph_retrieval import GraphRetriever

    # --- Build a real KnowledgeGraph with two orphaned entities + a relation ---
    kg = KnowledgeGraph(db_path=tmp_db_path)

    active_interface = Entity(
        name="Active interface",
        type=EntityType.Project,
        source_note_ids=[],
        properties={"source_conversation_turns": ["turn-1"]},
        confidence=0.6,
    )
    fep = Entity(
        name="Free Energy Principle",
        type=EntityType.Concept,
        aliases=["FEP"],
        source_note_ids=[],
        confidence=1.0,
    )
    kg.create_entity(active_interface)
    kg.create_entity(fep)

    kg.create_relation(
        Relation(
            source_entity_id=active_interface.id,
            target_entity_id=fep.id,
            type=RelationType.related_to,
            confidence=0.7,
        )
    )

    # --- Wire up the RAG pipeline (real graph, mock everything else) ---
    graph_retriever = GraphRetriever(graph=kg)
    mock_dense = _make_empty_async_retriever()
    mock_sparse = _make_empty_sync_retriever()
    fusion = HybridFusion(dense=mock_dense, sparse=mock_sparse, graph=graph_retriever)

    mock_llm = _make_mock_llm()
    mock_extractor = _make_mock_memory_extractor()

    generator = RAGGenerator(
        fusion=fusion,
        llm_provider=mock_llm,
        memory_extractor=mock_extractor,
    )

    # --- Run generate ---
    # The natural-language query "What is Active interface and FEP?" is
    # tokenized by GraphRetriever into ["Active", "interface", "FEP"].
    # "Active" matches the entity by name (substring match).
    # "FEP" matches the entity "Free Energy Principle" via its alias.
    # The entity's graph snippet includes its relation to "Free Energy Principle",
    # so both names appear in the system prompt.
    result = await generator.generate("What is Active interface and FEP?", empty_note_store)

    # --- Assert: system prompt passed to the LLM contains both entity names ---
    # complete() is called with prompt= and system= kwargs.
    assert mock_llm.complete.await_count == 1
    call_kwargs = mock_llm.complete.call_args.kwargs
    system_prompt = call_kwargs.get("system", "")
    assert "Active interface" in system_prompt, (
        f"Expected 'Active interface' in system prompt, got:\n{system_prompt}"
    )
    assert "Free Energy Principle" in system_prompt, (
        f"Expected 'Free Energy Principle' in system prompt (via relation snippet), "
        f"got:\n{system_prompt}"
    )

    # --- Assert: citations include entity citations with required fields ---
    citations = result["citations"]
    assert len(citations) >= 1, f"Expected >= 1 citation, got {len(citations)}"

    entity_citations = [c for c in citations if c.get("source_type") == "entity"]
    assert len(entity_citations) >= 1, (
        f"Expected >= 1 entity citation, got {len(entity_citations)}: {citations}"
    )

    for c in entity_citations:
        assert "entity_id" in c, f"entity_id missing in entity citation: {c}"
        assert "entity_name" in c, f"entity_name missing in entity citation: {c}"
        assert "entity_type" in c, f"entity_type missing in entity citation: {c}"
        assert c["entity_type"] in ("Project", "Concept"), (
            f"Unexpected entity_type: {c['entity_type']}"
        )

    # The retrieved entity (Active interface) must appear in the citations.
    citation_names = {c.get("entity_name", "") for c in entity_citations}
    assert "Active interface" in citation_names, (
        f"'Active interface' not in entity citation names: {citation_names}"
    )


@pytest.mark.asyncio
async def test_rag_retrieves_entity_by_alias(tmp_db_path, empty_note_store):
    """End-to-end: alias-based entity retrieval works through the full RAG
    pipeline.

    An entity named "Free Energy Principle" with aliases ["FEP", "free energy
    principle"] is retrieved when the query mentions the alias "FEP", and the
    entity name (not the alias) appears in the system prompt and citations.
    """
    from src.knowledge.graph import KnowledgeGraph
    from src.models.entity import Entity, EntityType
    from src.reasoning.rag import RAGGenerator
    from src.retrieval.fusion import HybridFusion
    from src.retrieval.graph_retrieval import GraphRetriever

    # --- Build a real KnowledgeGraph with one alias-backed orphaned entity ---
    kg = KnowledgeGraph(db_path=tmp_db_path)

    fep = Entity(
        name="Free Energy Principle",
        type=EntityType.Concept,
        aliases=["FEP", "free energy principle"],
        source_note_ids=[],
        confidence=1.0,
    )
    kg.create_entity(fep)

    # --- Wire up the RAG pipeline ---
    graph_retriever = GraphRetriever(graph=kg)
    mock_dense = _make_empty_async_retriever()
    mock_sparse = _make_empty_sync_retriever()
    fusion = HybridFusion(dense=mock_dense, sparse=mock_sparse, graph=graph_retriever)

    mock_llm = _make_mock_llm()
    mock_extractor = _make_mock_memory_extractor()

    generator = RAGGenerator(
        fusion=fusion,
        llm_provider=mock_llm,
        memory_extractor=mock_extractor,
    )

    # --- Run generate with a natural-language query containing the alias ---
    # GraphRetriever tokenizes "What is FEP?" into ["FEP"], then searches
    # for "FEP" against entity names and aliases. The alias "FEP" matches
    # the entity "Free Energy Principle", and the full entity name then
    # appears in the system prompt.
    result = await generator.generate("What is FEP?", empty_note_store)

    # --- Assert: system prompt contains the entity NAME (found via alias) ---
    assert mock_llm.complete.await_count == 1
    call_kwargs = mock_llm.complete.call_args.kwargs
    system_prompt = call_kwargs.get("system", "")
    assert "Free Energy Principle" in system_prompt, (
        f"Expected 'Free Energy Principle' in system prompt (retrieved via alias "
        f"'FEP'), got:\n{system_prompt}"
    )

    # --- Assert: citations include the entity ---
    citations = result["citations"]
    entity_citations = [c for c in citations if c.get("source_type") == "entity"]
    assert len(entity_citations) >= 1, (
        f"Expected >= 1 entity citation, got {len(entity_citations)}: {citations}"
    )

    citation = entity_citations[0]
    assert citation["entity_name"] == "Free Energy Principle", (
        f"Expected entity_name 'Free Energy Principle', got '{citation['entity_name']}'"
    )
    assert citation["entity_type"] == "Concept", (
        f"Expected entity_type 'Concept', got '{citation['entity_type']}'"
    )
    assert "entity_id" in citation
