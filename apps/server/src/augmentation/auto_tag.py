"""Auto-tagging engine using LLM and rule-based heuristics."""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from src.llm.json_parse import LLMOutputParseError, parse_llm_json
from src.llm.long_text import chunk_for_llm

if TYPE_CHECKING:
    from src.llm.base import LLMProvider
    from src.models.note import Note

logger = logging.getLogger(__name__)


class AutoTagger:
    """Generates tags for notes using LLM suggestions and rule-based heuristics."""

    _SYSTEM_PROMPT = (
        "You are a tagging assistant. Given a note title and content, suggest 3-7 relevant tags. "
        "Return ONLY a JSON array of tag strings. Tags should be lowercase, single words or short "
        "hyphenated phrases. Do not include any other text."
    )

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self._llm = llm_provider

    def _rule_based_tags(self, note: Note) -> list[str]:
        """Extract tags from markdown #hashtag syntax and capitalized keywords."""
        tags: set[str] = set()
        # Markdown hashtags
        for match in re.findall(r"#([a-zA-Z0-9_\-]+)", note.content):
            tags.add(match.lower())
        # Capitalized keywords (potential named entities)
        for match in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", note.content):
            tags.add(match.lower())
        # Frontmatter tags
        frontmatter_tags = note.frontmatter.get("tags", [])
        if isinstance(frontmatter_tags, list):
            for t in frontmatter_tags:
                tags.add(str(t).lower())
        elif isinstance(frontmatter_tags, str):
            for t in frontmatter_tags.split(","):
                tags.add(t.strip().lower())
        return sorted(tags)

    async def generate_tags(self, note: Note) -> list[str]:
        """Generate tags for a note using LLM + rules, deduplicated and sorted."""
        rule_tags = self._rule_based_tags(note)
        if self._llm is None:
            return rule_tags
        # Tag every chunk of the full content and union the results — large notes
        # are never truncated, so tags reflect the whole note, not just its head.
        llm_tags: set[str] = set()
        for chunk in chunk_for_llm(note.content):
            prompt = f"Title: {note.title}\n\nContent:\n{chunk}\n\nSuggest tags as JSON array."
            try:
                response = await self._llm.complete(prompt=prompt, system=self._SYSTEM_PROMPT)
            except Exception:
                logger.warning("Auto-tag: LLM call failed for a chunk, skipping it")
                continue
            try:
                data = parse_llm_json(response, expect="list")
                llm_tags.update(str(t).lower() for t in data)
            except LLMOutputParseError as exc:
                logger.warning("Auto-tag: failed to parse LLM tags as JSON: %s", exc)
        combined = sorted(set(rule_tags) | llm_tags)
        return combined

    def suggest_tags_for_text(self, text: str) -> list[str]:
        """Suggest tags for arbitrary text using only rule-based heuristics."""
        tags: set[str] = set()
        for match in re.findall(r"#([a-zA-Z0-9_\-]+)", text):
            tags.add(match.lower())
        for match in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text):
            tags.add(match.lower())
        return sorted(tags)
