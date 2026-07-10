"""Chat fact extraction creates hidden notes from factual user statements.

Mirrors the _extract_graph_entities test pattern: monkeypatch config flags and
LLM/store calls, use SimpleNamespace for lightweight app.state, AsyncMock for
async functions, and asyncio.sleep to let fire-and-forget tasks run.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

_FACT_JSON = json.dumps(
    {
        "contains_facts": True,
        "facts": [
            {"title": "User's wife name", "content": "My wife's name is May"},
        ],
    }
)

_NO_FACTS_JSON = json.dumps(
    {"contains_facts": False, "facts": []}
)

_MULTI_FACT_JSON = json.dumps(
    {
        "contains_facts": True,
        "facts": [
            {"title": "Wife name", "content": "My wife's name is May"},
            {"title": "Workplace", "content": "I work at Acme Corp"},
            {"title": "Birthday", "content": "My birthday is March 15th"},
        ],
    }
)

_MALFORMED_FACTS_JSON = json.dumps(
    {
        "contains_facts": True,
        "facts": [
            {"title": "Good fact", "content": "I live in Tokyo"},
            {"title": "", "content": "Missing title"},
            {"title": "Missing content", "content": ""},
            {"title": "No content key"},
            {"content": "No title key"},
            "not-a-dict",
        ],
    }
)


_SENTINEL = object()


def _make_request(llm=_SENTINEL, store=_SENTINEL):
    """Build a lightweight request with app.state for fire-and-forget tests.

    Pass llm=None or store=None to test the early-return guard for missing
    providers. Omit them to get default non-None placeholder objects.
    """
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                llm_provider=object() if llm is _SENTINEL else llm,
                note_store=object() if store is _SENTINEL else store,
            )
        )
    )


def _make_llm(return_value=_FACT_JSON):
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=return_value)
    return llm


def _make_store():
    store = MagicMock()
    store.create = MagicMock(return_value=SimpleNamespace(id="note-1"))
    return store


# --- _extract_facts unit tests -------------------------------------------------


async def test_extract_facts_factual_turn_creates_note(monkeypatch):
    """A turn with a clear fact creates a hidden note with correct provenance."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    llm = _make_llm(_FACT_JSON)
    store = _make_store()
    req = _make_request(llm=llm, store=store)

    with patch("src.api.notes._extract_and_store_entities") as fake_ent, \
         patch("src.api.notes._invalidate_sparse_index") as fake_inv:
        chat._extract_facts(req, "turn-1", "My wife's name is May", "That's nice!")
        await asyncio.sleep(0.05)

    store.create.assert_called_once()
    created_note = store.create.call_args[0][0]
    assert created_note.source_type == "hidden"
    assert created_note.frontmatter["extracted_from_turn"] == "turn-1"
    assert "auto-extracted" in created_note.frontmatter["tags"]
    assert created_note.title == "User's wife name"
    fake_ent.assert_called_once()
    fake_inv.assert_called_once()


async def test_extract_facts_question_no_note(monkeypatch):
    """A question turn should not create notes (LLM returns contains_facts=false)."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    llm = _make_llm(_NO_FACTS_JSON)
    store = _make_store()
    req = _make_request(llm=llm, store=store)

    chat._extract_facts(req, "turn-2", "What's my wife's name?", "I'm not sure.")
    await asyncio.sleep(0.05)

    store.create.assert_not_called()


async def test_extract_facts_greeting_no_note(monkeypatch):
    """A greeting turn should not create notes."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    llm = _make_llm(_NO_FACTS_JSON)
    store = _make_store()
    req = _make_request(llm=llm, store=store)

    chat._extract_facts(req, "turn-3", "Hello!", "Hi there!")
    await asyncio.sleep(0.05)

    store.create.assert_not_called()


async def test_extract_facts_empty_response(monkeypatch):
    """LLM returns empty JSON dict — no note created, no crash."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    llm = _make_llm("{}")
    store = _make_store()
    req = _make_request(llm=llm, store=store)

    chat._extract_facts(req, "turn-4", "My name is David", "Hello David!")
    await asyncio.sleep(0.05)

    store.create.assert_not_called()


async def test_extract_facts_parse_error(monkeypatch):
    """LLM returns unparseable garbage — no crash, no note."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    llm = _make_llm("this is not json at all {{{{")
    store = _make_store()
    req = _make_request(llm=llm, store=store)

    chat._extract_facts(req, "turn-5", "My name is David", "Hello David!")
    await asyncio.sleep(0.05)

    store.create.assert_not_called()


async def test_extract_facts_timeout(monkeypatch):
    """LLM times out — no note, no crash."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    llm = _make_llm()
    llm.complete = AsyncMock(side_effect=asyncio.TimeoutError())
    store = _make_store()
    req = _make_request(llm=llm, store=store)

    chat._extract_facts(req, "turn-6", "My name is David", "Hello David!")
    await asyncio.sleep(0.05)

    store.create.assert_not_called()


async def test_extract_facts_budget_exceeded(monkeypatch):
    """BudgetExceededError is caught — no note, no crash."""
    from src.api import chat
    from src.llm.router import BudgetExceededError

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    llm = _make_llm()
    llm.complete = AsyncMock(side_effect=BudgetExceededError(5.0, 10.0))
    store = _make_store()
    req = _make_request(llm=llm, store=store)

    chat._extract_facts(req, "turn-7", "My name is David", "Hello David!")
    await asyncio.sleep(0.05)

    store.create.assert_not_called()


async def test_extract_facts_multiple_facts(monkeypatch):
    """Multiple facts in one turn create multiple notes."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    llm = _make_llm(_MULTI_FACT_JSON)
    store = _make_store()
    req = _make_request(llm=llm, store=store)

    with patch("src.api.notes._extract_and_store_entities"), \
         patch("src.api.notes._invalidate_sparse_index"):
        chat._extract_facts(req, "turn-8", "My wife is May, I work at Acme, birthday March 15", "Great!")
        await asyncio.sleep(0.05)

    assert store.create.call_count == 3
    for call in store.create.call_args_list:
        note = call[0][0]
        assert note.source_type == "hidden"
        assert note.frontmatter["extracted_from_turn"] == "turn-8"


async def test_extract_facts_no_llm_provider(monkeypatch):
    """llm is None — returns early, no crash."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    store = _make_store()
    req = _make_request(llm=None, store=store)

    chat._extract_facts(req, "turn-9", "My name is David", "Hello!")
    await asyncio.sleep(0.05)

    store.create.assert_not_called()


async def test_extract_facts_no_note_store(monkeypatch):
    """store is None — returns early, no crash."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    llm = _make_llm()
    req = _make_request(llm=llm, store=None)

    chat._extract_facts(req, "turn-10", "My name is David", "Hello!")
    await asyncio.sleep(0.05)

    llm.complete.assert_not_called()


async def test_extract_facts_skips_malformed_facts(monkeypatch):
    """Only well-formed facts (with both title and content) create notes."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    llm = _make_llm(_MALFORMED_FACTS_JSON)
    store = _make_store()
    req = _make_request(llm=llm, store=store)

    with patch("src.api.notes._extract_and_store_entities"), \
         patch("src.api.notes._invalidate_sparse_index"):
        chat._extract_facts(req, "turn-11", "stuff", "reply")
        await asyncio.sleep(0.05)

    # Only the first fact (Good fact) is well-formed
    assert store.create.call_count == 1
    note = store.create.call_args_list[0][0][0]
    assert note.title == "Good fact"
    assert note.content == "I live in Tokyo"


# --- _capture_turn integration tests -------------------------------------------


async def test_capture_turn_calls_extract_facts(monkeypatch):
    """_capture_turn calls _extract_facts when the flag is on."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: True)
    monkeypatch.setattr("src.runtime_config.auto_entity_extraction", lambda: False)

    with patch("src.api.conversations.update_message"), \
         patch("src.api.websocket.get_broadcaster", return_value=None), \
         patch.object(chat, "_embed_turn", new=AsyncMock()) as fake_embed, \
         patch.object(chat, "_extract_facts") as fake_facts:
        fake_embed.return_value = None
        req = _make_request()

        await chat._capture_turn(
            req, "user-1", "My wife's name is May", "That's nice!",
            "turn-12", [], [],
        )
        await asyncio.sleep(0.05)

    fake_facts.assert_called_once_with(req, "turn-12", "My wife's name is May", "That's nice!")


async def test_capture_turn_no_fact_extraction_when_disabled(monkeypatch):
    """_capture_turn does not call _extract_facts when the flag is off."""
    from src.api import chat

    monkeypatch.setattr("src.runtime_config.auto_fact_extraction", lambda: False)
    monkeypatch.setattr("src.runtime_config.auto_entity_extraction", lambda: False)

    with patch("src.api.conversations.update_message"), \
         patch("src.api.websocket.get_broadcaster", return_value=None), \
         patch.object(chat, "_embed_turn", new=AsyncMock()), \
         patch.object(chat, "_extract_facts") as fake_facts:
        req = _make_request()

        await chat._capture_turn(
            req, "user-1", "My wife's name is May", "That's nice!",
            "turn-13", [], [],
        )
        await asyncio.sleep(0.05)

    fake_facts.assert_not_called()