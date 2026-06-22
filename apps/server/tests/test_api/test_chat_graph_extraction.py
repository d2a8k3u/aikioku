"""Chat turns extract entities + relations into the knowledge graph.

Covers the reusable text-based extraction core (``extract_entities_from_text``)
and the chat wiring (``_extract_graph_entities``) that feeds the graph from chat
turns the same way notes do — the graph is the project's source of truth.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.knowledge.graph import KnowledgeGraph
from src.llm.base import LLMProvider

_ENT_JSON = json.dumps([
    {"name": "Aiwen", "type": "Project", "aliases": [], "confidence": 0.9},
    {"name": "Apple Silicon", "type": "Concept", "aliases": [], "confidence": 0.8},
])


@pytest.fixture
def graph():
    with tempfile.TemporaryDirectory() as d:
        yield KnowledgeGraph(db_path=str(Path(d) / "kuzu.db"))


@pytest.fixture
def mock_llm():
    p = AsyncMock(spec=LLMProvider)
    p.complete = AsyncMock(return_value="[]")
    return p


# --- text-based extraction core --------------------------------------------------

async def test_extract_from_text_note_records_source_note_id(graph, mock_llm):
    mock_llm.complete.return_value = _ENT_JSON
    from src.knowledge.pipeline import extract_entities_from_text

    ents = await extract_entities_from_text(
        text="Aiwen runs on Apple Silicon",
        source_id="note-9",
        llm_provider=mock_llm,
        graph=graph,
    )

    assert len(ents) == 2
    assert graph.count_entities() == 2
    assert all("note-9" in e.source_note_ids for e in ents)


async def test_extract_from_text_chat_keeps_turn_id_out_of_note_ids(graph, mock_llm):
    """Chat provenance must NOT pollute source_note_ids (else it becomes a broken
    /notes/<turn_id> citation). It goes into properties instead."""
    mock_llm.complete.return_value = _ENT_JSON
    from src.knowledge.pipeline import extract_entities_from_text

    ents = await extract_entities_from_text(
        text="Aiwen on Apple Silicon",
        source_id="turn-7",
        llm_provider=mock_llm,
        graph=graph,
        source_is_note=False,
    )

    assert len(ents) == 2
    assert graph.count_entities() == 2
    assert all("turn-7" not in e.source_note_ids for e in ents)
    assert all(e.source_note_ids == [] for e in ents)
    assert all("turn-7" in e.properties.get("source_conversation_turns", []) for e in ents)


async def test_extract_from_text_empty_skips_llm(graph, mock_llm):
    from src.knowledge.pipeline import extract_entities_from_text

    ents = await extract_entities_from_text(
        text="   ", source_id="t", llm_provider=mock_llm, graph=graph
    )

    assert ents == []
    mock_llm.complete.assert_not_called()


async def test_extract_from_note_delegates_to_text(graph, mock_llm):
    mock_llm.complete.return_value = _ENT_JSON
    from src.knowledge.pipeline import extract_entities_from_note
    from src.models.note import Note

    note = Note(id="note-x", title="t", content="Aiwen on Apple Silicon", path="/n.md")
    ents = await extract_entities_from_note(note=note, llm_provider=mock_llm, graph=graph)

    assert len(ents) == 2
    assert all("note-x" in e.source_note_ids for e in ents)


# --- chat wiring -----------------------------------------------------------------

async def test_chat_schedules_graph_extraction(monkeypatch):
    from src.api import chat

    fake = AsyncMock(return_value=[])
    monkeypatch.setattr("src.knowledge.pipeline.extract_entities_from_text", fake)
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        llm_provider=object(), knowledge_graph=object())))

    chat._extract_graph_entities(req, "turn-1", "who is aiwen", "aiwen is a project")
    await asyncio.sleep(0.05)  # let the fire-and-forget task run

    fake.assert_awaited_once()
    kwargs = fake.call_args.kwargs
    assert kwargs["source_id"] == "turn-1"
    assert kwargs.get("source_is_note") is False
    assert "who is aiwen" in kwargs["text"] and "aiwen is a project" in kwargs["text"]


async def test_chat_extraction_skips_when_unconfigured(monkeypatch):
    from src.api import chat

    fake = AsyncMock(return_value=[])
    monkeypatch.setattr("src.knowledge.pipeline.extract_entities_from_text", fake)
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        llm_provider=None, knowledge_graph=None)))

    chat._extract_graph_entities(req, "turn-2", "q", "a")
    await asyncio.sleep(0.05)

    fake.assert_not_called()
