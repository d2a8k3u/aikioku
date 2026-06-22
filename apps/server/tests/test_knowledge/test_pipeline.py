"""Tests for Note -> KG entity extraction pipeline.

Uses real Kuzu DB and real ChromaDB in temp dirs. LLMProvider is mocked.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.graph import KnowledgeGraph
from src.llm.base import LLMProvider
from src.models.note import Note


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_dirs():
    """Create temporary directories for Kuzu DB and ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        yield {
            "kuzu": str(base / "kuzu.db"),
            "chroma": str(base / "chroma"),
        }


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
    provider.embed = AsyncMock(return_value=[0.0] * 384)
    provider.is_available = MagicMock(return_value=True)
    return provider


@pytest.fixture
def sample_note():
    return Note(
        id="note-001",
        title="Python and Guido",
        content=(
            "Python is a high-level programming language. "
            "Python was created by Guido van Rossum. "
            "Python supports multiple paradigms including OOP and functional programming."
        ),
        path="/notes/python.md",
    )


@pytest.fixture
def entity_extraction_response():
    """Mock LLM JSON response for entity extraction."""
    return json.dumps([
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
        {
            "name": "OOP",
            "type": "Concept",
            "aliases": ["Object-Oriented Programming"],
            "confidence": 0.85,
        },
    ])


# --------------------------------------------------------------------------- #
# Tests: extract_entities_from_note
# --------------------------------------------------------------------------- #


class TestExtractEntitiesFromNote:
    """Test that entities are extracted from note content via LLM."""

    @pytest.mark.asyncio
    async def test_extract_entities_from_note_returns_entities(
        self,
        knowledge_graph,
        mock_llm_provider,
        sample_note,
        entity_extraction_response,
    ):
        mock_llm_provider.complete.return_value = entity_extraction_response

        from src.knowledge.pipeline import extract_entities_from_note

        entities = await extract_entities_from_note(
            note=sample_note,
            llm_provider=mock_llm_provider,
            graph=knowledge_graph,
        )

        assert len(entities) == 3
        names = {e.name for e in entities}
        assert "Python" in names
        assert "Guido van Rossum" in names
        assert "OOP" in names

    @pytest.mark.asyncio
    async def test_extract_entities_stores_in_kg(
        self,
        knowledge_graph,
        mock_llm_provider,
        sample_note,
        entity_extraction_response,
    ):
        mock_llm_provider.complete.return_value = entity_extraction_response

        from src.knowledge.pipeline import extract_entities_from_note

        entities = await extract_entities_from_note(
            note=sample_note,
            llm_provider=mock_llm_provider,
            graph=knowledge_graph,
        )

        # After extraction, KG should contain the entities
        assert knowledge_graph.count_entities() == 3

        # Each entity should be fetchable
        for entity in entities:
            fetched = knowledge_graph.get_entity(entity.id)
            assert fetched is not None
            assert fetched.name == entity.name

    @pytest.mark.asyncio
    async def test_extract_entities_have_note_id_in_source(
        self,
        knowledge_graph,
        mock_llm_provider,
        sample_note,
        entity_extraction_response,
    ):
        mock_llm_provider.complete.return_value = entity_extraction_response

        from src.knowledge.pipeline import extract_entities_from_note

        entities = await extract_entities_from_note(
            note=sample_note,
            llm_provider=mock_llm_provider,
            graph=knowledge_graph,
        )

        for entity in entities:
            assert sample_note.id in entity.source_note_ids

    @pytest.mark.asyncio
    async def test_extract_entities_empty_response(
        self,
        knowledge_graph,
        mock_llm_provider,
        sample_note,
    ):
        mock_llm_provider.complete.return_value = "[]"

        from src.knowledge.pipeline import extract_entities_from_note

        entities = await extract_entities_from_note(
            note=sample_note,
            llm_provider=mock_llm_provider,
            graph=knowledge_graph,
        )

        assert entities == []
        assert knowledge_graph.count_entities() == 0

    @pytest.mark.asyncio
    async def test_extract_entities_from_fenced_json(
        self,
        knowledge_graph,
        mock_llm_provider,
        sample_note,
        entity_extraction_response,
    ):
        # Real LLMs wrap JSON in ```json fences; the pipeline must still parse it.
        mock_llm_provider.complete.return_value = (
            f"Here are the entities:\n```json\n{entity_extraction_response}\n```"
        )

        from src.knowledge.pipeline import extract_entities_from_note

        entities = await extract_entities_from_note(
            note=sample_note,
            llm_provider=mock_llm_provider,
            graph=knowledge_graph,
        )

        assert len(entities) == 3
        names = {e.name for e in entities}
        assert "Python" in names

    @pytest.mark.asyncio
    async def test_extract_entities_garbage_returns_empty(
        self,
        knowledge_graph,
        mock_llm_provider,
        sample_note,
    ):
        mock_llm_provider.complete.return_value = "I cannot extract any entities."

        from src.knowledge.pipeline import extract_entities_from_note

        entities = await extract_entities_from_note(
            note=sample_note,
            llm_provider=mock_llm_provider,
            graph=knowledge_graph,
        )

        assert entities == []
        assert knowledge_graph.count_entities() == 0

    @pytest.mark.asyncio
    async def test_extract_entities_creates_relations_between_cooccurring(
        self,
        knowledge_graph,
        mock_llm_provider,
        sample_note,
        entity_extraction_response,
    ):
        mock_llm_provider.complete.return_value = entity_extraction_response

        from src.knowledge.pipeline import extract_entities_from_note

        await extract_entities_from_note(
            note=sample_note,
            llm_provider=mock_llm_provider,
            graph=knowledge_graph,
        )

        # Relations should be created between co-occurring entities
        assert knowledge_graph.count_relations() > 0


# --------------------------------------------------------------------------- #
# Tests: store_note_embeddings
# --------------------------------------------------------------------------- #


class TestStoreNoteEmbeddings:
    """Test that note embeddings are stored after note creation."""

    @pytest.mark.asyncio
    async def test_embeddings_stored_after_note_creation(
        self,
        mock_llm_provider,
        embedding_store,
        sample_note,
    ):
        mock_llm_provider.embed = AsyncMock(
            return_value=[0.1] * 384
        )

        from src.knowledge.pipeline import store_note_embeddings

        await store_note_embeddings(
            note=sample_note,
            embedder=mock_llm_provider,
            embedding_store=embedding_store,
        )

        # Embedding should be stored
        assert embedding_store.count() >= 1

    @pytest.mark.asyncio
    async def test_embeddings_uses_note_content(
        self,
        mock_llm_provider,
        embedding_store,
        sample_note,
    ):
        mock_llm_provider.embed = AsyncMock(
            return_value=[0.2] * 384
        )

        from src.knowledge.pipeline import store_note_embeddings

        await store_note_embeddings(
            note=sample_note,
            embedder=mock_llm_provider,
            embedding_store=embedding_store,
        )

        # Verify embed was called with note content
        mock_llm_provider.embed.assert_called()
        call_args = mock_llm_provider.embed.call_args
        assert sample_note.content in call_args[0][0]
