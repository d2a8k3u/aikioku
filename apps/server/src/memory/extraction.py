"""Memory extraction via LLM."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from src.events import EventBus
from src.llm.base import LLMProvider
from src.llm.json_parse import LLMOutputParseError, parse_llm_json
from src.models.memory import Memory
from src.models.note import Note


def build_episodic_memory(question: str, created: datetime | None = None) -> Memory:
    """Build a deterministic episodic memory recording that the user asked a question.

    Uses NO LLM call: the question itself is the most faithful, searchable record
    of *what* and *when* the user asked, and building it deterministically avoids
    doubling the per-turn LLM load (the codebase is deliberately careful about
    fanning out extra LLM calls). The structured ``user asked_about <question>``
    triple flows into consolidation, ``/api/memory`` and the knowledge graph.

    Args:
        question: The user's question text.
        created: The turn timestamp; defaults to now when omitted.

    Returns:
        A high-confidence episodic ``Memory`` with ``source="conversation"``.
    """
    text = (question or "").strip()
    obj = text if text else "(empty question)"
    memory = Memory(
        subject="user",
        predicate="asked_about",
        object=obj,
        confidence=0.9,
        source="conversation",
        vitality_score=0.9,
    )
    if created is not None:
        memory.created = created
        memory.modified = created
    return memory


class MemoryExtractor:
    """Extracts structured memory triples from notes and conversations using an LLM."""

    def __init__(self, llm_provider: LLMProvider, event_bus: EventBus) -> None:
        self._llm = llm_provider
        self._event_bus = event_bus

    async def extract_from_note(self, note: Note) -> list[Memory]:
        """Extract memories from a note's content."""
        prompt = self._build_extraction_prompt(note.content)
        response = await self._llm.complete(prompt=prompt)
        return self._parse_memories(response, source=note.id)

    async def extract_from_text(self, text: str, source: str = "user") -> list[Memory]:
        """Extract memories from arbitrary free text (no Note object).

        Used by the direct ``POST /api/memory/`` create path: the caller supplies
        a sentence and the LLM parses it into one or more subject-predicate-object
        triples, tagged with ``source``.
        """
        prompt = self._build_extraction_prompt(text)
        response = await self._llm.complete(prompt=prompt)
        return self._parse_memories(response, source=source)

    async def extract_from_conversation(self, messages: list[dict[str, Any]]) -> list[Memory]:
        """Extract memories from a conversation."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        prompt = self._build_extraction_prompt(text)
        response = await self._llm.complete(prompt=prompt)
        return self._parse_memories(response, source="conversation")

    def _build_extraction_prompt(self, text: str) -> str:
        """Build a prompt that instructs the LLM to extract subject-predicate-object triples."""
        return (
            "Extract ALL factual memories from the following text as a JSON array of objects.\n"
            "Each object must have keys: subject, predicate, object, confidence.\n"
            "Confidence is a float between 0.0 and 1.0.\n"
            "Do NOT summarize, omit, or truncate any facts. Include every distinct\n"
            "subject-predicate-object triple you can identify, no matter how minor.\n"
            "Return only the JSON array, nothing else.\n\n"
            f"Text:\n{text}"
        )

    def _parse_memories(self, response: str, source: str) -> list[Memory]:
        """Parse the LLM response into a list of Memory objects."""
        try:
            data = cast("list[Any]", parse_llm_json(response, expect="list"))
        except LLMOutputParseError:
            return []

        memories = []
        for item in data:
            try:
                confidence = float(item["confidence"])
                memory = Memory(
                    subject=item["subject"],
                    predicate=item["predicate"],
                    object=item["object"],
                    confidence=confidence,
                    source=source,
                    # A freshly-extracted, confident memory starts "alive" so
                    # stage_forgetting tiers it sensibly instead of defaulting
                    # it to a low vitality.
                    vitality_score=confidence,
                )
                memories.append(memory)
            except (KeyError, TypeError, ValueError):
                continue

        return memories
