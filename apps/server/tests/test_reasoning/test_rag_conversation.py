"""RAGGenerator: conversation recall + history injection into the system prompt."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def fusion():
    f = AsyncMock()
    f.search.return_value = []  # no note hits, isolate conversation behaviour
    return f


@pytest.mark.asyncio
async def test_history_injected_into_system_prompt(fusion):
    from src.reasoning.rag import RAGGenerator

    rag = RAGGenerator(fusion=fusion, llm_provider=AsyncMock(), memory_extractor=AsyncMock())
    history = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]

    system_prompt, _ = await rag.build_context(
        "what was my previous question", note_store=MagicMock(), history=history
    )

    assert "Recent conversation" in system_prompt
    assert "previous question" in system_prompt


@pytest.mark.asyncio
async def test_conversation_recall_injected(fusion):
    from src.reasoning.rag import RAGGenerator

    conv_retriever = AsyncMock()
    conv_retriever.search.return_value = [
        {
            "note_id": "turn-1",
            "text": "[2026-03-01] user: about taxes\nassistant: deadline is April",
            "score": 0.9,
        }
    ]
    rag = RAGGenerator(
        fusion=fusion,
        llm_provider=AsyncMock(),
        memory_extractor=AsyncMock(),
        conversation_retriever=conv_retriever,
    )

    system_prompt, _ = await rag.build_context("when did I ask about taxes", note_store=MagicMock())

    conv_retriever.search.assert_awaited_once()
    assert "Relevant past conversations" in system_prompt
    assert "about taxes" in system_prompt
    assert "2026-03-01" in system_prompt


@pytest.mark.asyncio
async def test_no_conversation_retriever_is_safe(fusion):
    from src.reasoning.rag import RAGGenerator

    rag = RAGGenerator(fusion=fusion, llm_provider=AsyncMock(), memory_extractor=AsyncMock())
    system_prompt, citations = await rag.build_context("hello", note_store=MagicMock())
    assert "Relevant past conversations" not in system_prompt
    assert citations == []
