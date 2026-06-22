"""Prompt prefix caching for LLM providers.

Marks the static system prompt prefix as cacheable so providers can reuse
KV-cache states across turns. The system prompt is split at ``\\nContext:\\n``
(i.e. a line containing only ``Context:``) —
everything before this marker is static (same every turn), everything after
is dynamic (different notes each turn).

For OpenAI-compatible APIs (OpenRouter, Ollama Cloud), prompt caching works
by adding ``"cache_control": {"type": "ephemeral"}`` to content blocks in the
system message. The API caches KV states for marked blocks and reuses them on
subsequent requests with the same prefix.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Environment toggle
# ---------------------------------------------------------------------------

_MARKER = "\nContext:\n"


def _is_enabled() -> bool:
    """Check whether prompt caching is enabled via PROMPT_CACHE_ENABLED."""
    return os.environ.get("PROMPT_CACHE_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cacheable_system_prompt(full_system_prompt: str) -> list[dict]:
    """Split the system prompt at ``\\nContext:\\n`` and mark the static prefix as cacheable.

    Returns a list of content blocks suitable for an OpenAI-compatible
    system message ``"content"`` field.  The static prefix receives a
    ``cache_control`` marker so providers can reuse KV-cache states.

    If the marker is not found, the **entire** prompt is marked as cacheable
    (graceful degradation).
    """
    if _MARKER in full_system_prompt:
        idx = full_system_prompt.index(_MARKER)
        # Include the marker itself in the static prefix so the assembled
        # prompt is identical to the original.
        static_prefix = full_system_prompt[: idx + len(_MARKER)]
        dynamic_suffix = full_system_prompt[idx + len(_MARKER) :]
    else:
        # No marker found — cache the entire prompt.
        static_prefix = full_system_prompt
        dynamic_suffix = ""

    blocks: list[dict] = [
        {
            "type": "text",
            "text": static_prefix,
            "cache_control": {"type": "ephemeral"},
        },
    ]
    if dynamic_suffix:
        blocks.append(
            {
                "type": "text",
                "text": dynamic_suffix,
            }
        )
    return blocks


def build_messages_with_cache(system_prompt: str, user_prompt: str) -> list[dict]:
    """Build a messages list with prompt-caching support.

    When ``PROMPT_CACHE_ENABLED`` is true (the default), the system message
    uses cacheable content blocks.  When disabled, a plain-text system message
    is returned for full backward compatibility.

    Returns:
        A list of message dicts: ``[system_message, user_message]``.
    """
    if not _is_enabled():
        # Backward-compatible path: plain text system message.
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    # Cache-enabled path: content blocks with cache_control markers.
    system_content = build_cacheable_system_prompt(system_prompt)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]
