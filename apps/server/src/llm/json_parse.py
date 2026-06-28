"""Tolerant parsing of JSON returned by LLMs.

Real models often wrap JSON in ```json fences or surrounding prose, so a plain
``json.loads`` on the raw response throws. ``parse_llm_json`` strips fences and,
as a fallback, extracts the first balanced top-level array/object before parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

# Matches a leading ```json / ``` fence and the trailing ``` fence.
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)

_SNIPPET_LEN = 200


class LLMOutputParseError(ValueError):
    """Raised when LLM output cannot be parsed into the expected JSON type."""


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    if match:
        return match.group("body").strip()
    return text


def _type_matches(value: object, expect: str | None) -> bool:
    if expect is None:
        return isinstance(value, (list, dict))
    if expect == "list":
        return isinstance(value, list)
    if expect == "dict":
        return isinstance(value, dict)
    raise ValueError(f"Invalid expect value: {expect!r}")


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the substring from the first open_ch to the last close_ch, if any."""
    start = text.find(open_ch)
    end = text.rfind(close_ch)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_llm_json(text: str, expect: str | None = None) -> list[Any] | dict[str, Any]:
    """Parse JSON from a (possibly messy) LLM response.

    Args:
        text: The raw LLM response.
        expect: One of "list", "dict", or None. If set, only a result of that
            top-level type is accepted; otherwise both are accepted.

    Returns:
        The parsed list or dict.

    Raises:
        LLMOutputParseError: If nothing parses to the expected type.
    """
    if not isinstance(text, str):
        raise LLMOutputParseError(f"Expected str, got {type(text).__name__}")

    stripped = _strip_fence(text.strip())

    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)

    # Fallbacks: extract the first balanced top-level array and/or object.
    if expect in (None, "list"):
        array = _extract_balanced(stripped, "[", "]")
        if array is not None:
            candidates.append(array)
    if expect in (None, "dict"):
        obj = _extract_balanced(stripped, "{", "}")
        if obj is not None:
            candidates.append(obj)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if _type_matches(parsed, expect):
            return cast("list[Any] | dict[str, Any]", parsed)

    snippet = text.strip()[:_SNIPPET_LEN]
    expected_desc = expect if expect is not None else "list or dict"
    raise LLMOutputParseError(
        f"Could not parse LLM output as {expected_desc}. Snippet: {snippet!r}"
    )
