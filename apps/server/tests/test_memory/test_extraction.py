"""Tests for MemoryExtractor."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.events import EventBus
from src.llm.base import LLMProvider
from src.models.memory import Memory
from src.models.note import Note


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    """Return a mock LLMProvider."""
    provider = AsyncMock(spec=LLMProvider)
    provider.complete = AsyncMock()
    return provider


@pytest.fixture
def mock_event_bus(tmp_path) -> EventBus:
    """Return an EventBus backed by a temp database."""
    return EventBus(db_path=str(tmp_path / "events.db"))


@pytest.fixture
def sample_note() -> Note:
    """Return a sample Note for testing."""
    return Note(
        title="Python Basics",
        content="Python is a programming language. Python was created by Guido van Rossum.",
        path="/notes/python-basics.md",
    )


class TestExtractFromNote:
    """Test extract_from_note returns Memory objects with correct fields."""

    @pytest.mark.asyncio
    async def test_extract_from_note_returns_memories(
        self, mock_llm_provider, mock_event_bus, sample_note
    ):
        from src.memory.extraction import MemoryExtractor

        mock_response = json.dumps(
            [
                {
                    "subject": "Python",
                    "predicate": "is_a",
                    "object": "programming language",
                    "confidence": 0.95,
                },
                {
                    "subject": "Python",
                    "predicate": "created_by",
                    "object": "Guido van Rossum",
                    "confidence": 0.90,
                },
            ]
        )
        mock_llm_provider.complete.return_value = mock_response

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)
        memories = await extractor.extract_from_note(sample_note)

        assert len(memories) == 2
        assert all(isinstance(m, Memory) for m in memories)
        assert memories[0].subject == "Python"
        assert memories[0].predicate == "is_a"
        assert memories[0].object == "programming language"
        assert memories[0].confidence == 0.95
        assert memories[0].source == sample_note.id

    @pytest.mark.asyncio
    async def test_extract_from_note_calls_llm_with_note_content(
        self, mock_llm_provider, mock_event_bus, sample_note
    ):
        from src.memory.extraction import MemoryExtractor

        mock_llm_provider.complete.return_value = "[]"

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)
        await extractor.extract_from_note(sample_note)

        mock_llm_provider.complete.assert_called_once()
        call_kwargs = mock_llm_provider.complete.call_args
        assert sample_note.content in call_kwargs.kwargs.get("prompt", "")


class TestExtractFromConversation:
    """Test extract_from_conversation returns memories."""

    @pytest.mark.asyncio
    async def test_extract_from_conversation_returns_memories(
        self, mock_llm_provider, mock_event_bus
    ):
        from src.memory.extraction import MemoryExtractor

        messages = [
            {"role": "user", "content": "Tell me about machine learning."},
            {
                "role": "assistant",
                "content": "Machine learning is a subset of AI that uses algorithms.",
            },
        ]

        mock_response = json.dumps(
            [
                {
                    "subject": "Machine learning",
                    "predicate": "is_a",
                    "object": "subset of AI",
                    "confidence": 0.88,
                },
            ]
        )
        mock_llm_provider.complete.return_value = mock_response

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)
        memories = await extractor.extract_from_conversation(messages)

        assert len(memories) == 1
        assert memories[0].subject == "Machine learning"
        assert memories[0].predicate == "is_a"
        assert memories[0].object == "subset of AI"
        assert memories[0].confidence == 0.88

    @pytest.mark.asyncio
    async def test_extract_from_conversation_formats_messages(
        self, mock_llm_provider, mock_event_bus
    ):
        from src.memory.extraction import MemoryExtractor

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        mock_llm_provider.complete.return_value = "[]"

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)
        await extractor.extract_from_conversation(messages)

        mock_llm_provider.complete.assert_called_once()
        call_kwargs = mock_llm_provider.complete.call_args
        prompt = call_kwargs.kwargs.get("prompt", "")
        assert "user" in prompt.lower() or "Hello" in prompt


class TestExtractFromText:
    """Test extract_from_text parses free text into triples tagged with source."""

    @pytest.mark.asyncio
    async def test_extract_from_text_returns_memories(self, mock_llm_provider, mock_event_bus):
        from src.memory.extraction import MemoryExtractor

        mock_llm_provider.complete.return_value = json.dumps(
            [
                {"subject": "Alice", "predicate": "works_at", "object": "Acme", "confidence": 0.9},
            ]
        )

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)
        memories = await extractor.extract_from_text("Alice works at Acme", source="user")

        assert len(memories) == 1
        assert memories[0].subject == "Alice"
        assert memories[0].object == "Acme"
        assert memories[0].source == "user"

    @pytest.mark.asyncio
    async def test_extract_from_text_passes_text_to_llm(self, mock_llm_provider, mock_event_bus):
        from src.memory.extraction import MemoryExtractor

        mock_llm_provider.complete.return_value = "[]"

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)
        await extractor.extract_from_text("Bob lives in NYC")

        mock_llm_provider.complete.assert_called_once()
        assert "Bob lives in NYC" in mock_llm_provider.complete.call_args.kwargs.get("prompt", "")


class TestParseMemories:
    """Test _parse_memories handles valid and invalid JSON."""

    def test_parse_memories_valid_json(self, mock_llm_provider, mock_event_bus):
        from src.memory.extraction import MemoryExtractor

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)

        response = json.dumps(
            [
                {
                    "subject": "Alice",
                    "predicate": "knows",
                    "object": "Bob",
                    "confidence": 0.75,
                },
            ]
        )

        memories = extractor._parse_memories(response, source="test-source")

        assert len(memories) == 1
        assert memories[0].subject == "Alice"
        assert memories[0].predicate == "knows"
        assert memories[0].object == "Bob"
        assert memories[0].confidence == 0.75
        assert memories[0].source == "test-source"

    def test_parse_memories_invalid_json_returns_empty(self, mock_llm_provider, mock_event_bus):
        from src.memory.extraction import MemoryExtractor

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)

        memories = extractor._parse_memories("not valid json", source="test-source")

        assert memories == []

    def test_parse_memories_empty_array(self, mock_llm_provider, mock_event_bus):
        from src.memory.extraction import MemoryExtractor

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)

        memories = extractor._parse_memories("[]", source="test-source")

        assert memories == []

    def test_parse_memories_fenced_json(self, mock_llm_provider, mock_event_bus):
        from src.memory.extraction import MemoryExtractor

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)

        inner = json.dumps(
            [
                {
                    "subject": "Alice",
                    "predicate": "knows",
                    "object": "Bob",
                    "confidence": 0.75,
                },
            ]
        )
        response = f"Here are the memories:\n```json\n{inner}\n```"

        memories = extractor._parse_memories(response, source="test-source")

        assert len(memories) == 1
        assert memories[0].subject == "Alice"
        assert memories[0].object == "Bob"

    def test_parse_memories_seeds_vitality_with_confidence(self, mock_llm_provider, mock_event_bus):
        """A freshly-extracted, confident memory should start 'alive'.

        vitality_score must seed from confidence so stage_forgetting tiers it
        sensibly instead of dropping it to warm/cold by default.
        """
        from src.memory.extraction import MemoryExtractor

        extractor = MemoryExtractor(llm_provider=mock_llm_provider, event_bus=mock_event_bus)

        response = json.dumps(
            [
                {
                    "subject": "Alice",
                    "predicate": "knows",
                    "object": "Bob",
                    "confidence": 0.82,
                },
            ]
        )

        memories = extractor._parse_memories(response, source="test-source")

        assert len(memories) == 1
        assert memories[0].vitality_score == memories[0].confidence == 0.82
