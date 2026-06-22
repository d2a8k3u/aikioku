"""Progressive Summarization module.

Generates multi-level summaries of notes:
- brief: 3-5 bullet points
- detailed: 1-2 paragraph summary
- one-liner: single sentence summary
"""
from __future__ import annotations

from src.llm.base import LLMProvider
from src.models.note import Note


class ProgressiveSummarizer:
    """Generates progressive (multi-level) summaries of notes using an LLM."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm = llm_provider

    async def summarize(self, note: Note) -> dict:
        """Return all three summary levels for a given note.

        Returns:
            dict with keys "brief", "detailed", "one-liner".
        """
        content = note.content
        return {
            "brief": await self._summarize_level(content, "brief"),
            "detailed": await self._summarize_level(content, "detailed"),
            "one-liner": await self._summarize_level(content, "one-liner"),
        }

    async def _summarize_level(self, content: str, level: str) -> str:
        """Call the LLM with a level-specific prompt and return the summary text."""
        system = self._build_prompt(content, level)
        response = await self.llm.complete(prompt=content, system=system)
        return response.strip()

    def _build_prompt(self, content: str, level: str) -> str:
        """Build a system prompt tailored to the requested summary level."""
        prompts = {
            "brief": (
                "You are a note summarizer. Summarize the following content into 3-5 bullet points. "
                "Each bullet should capture a key idea. Use concise language. "
                "Output only the bullet points, one per line, starting with '- '."
            ),
            "detailed": (
                "You are a note summarizer. Summarize the following content into 1-2 paragraphs. "
                "Capture the main ideas, important details, and overall narrative. "
                "Write in clear, flowing prose."
            ),
            "one-liner": (
                "You are a note summarizer. Summarize the following content in a single sentence. "
                "Capture the essence of the content in one clear, concise sentence. "
                "Output only the sentence."
            ),
        }
        return prompts[level]
