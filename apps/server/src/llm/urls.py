"""Base-URL normalization + endpoint joining for LLM providers.

Users enter a single base URL per provider; every call site builds endpoints
from it through here so the result is consistent and forgiving of pasted path
noise (trailing slashes, an accidental ``/api``, ``/v1``, or ``/api/v1``).

Dialects:
- ``"ollama"`` (local + remote): canonical base is the bare host root; native
  suffixes carry the full ``/api/...`` path.
- ``"openrouter"`` (OpenAI-compatible): canonical base is the host plus exactly
  one ``/api/v1``; suffixes are bare (``/models``, ``/chat/completions``).
- ``"openai"`` (OpenAI): canonical base is the host plus exactly one ``/v1``;
  suffixes are bare (``/embeddings``, ``/models``).
"""
from __future__ import annotations

OLLAMA_GENERATE = "/api/generate"
OLLAMA_EMBED = "/api/embed"          # remote (current Ollama shape)
OLLAMA_EMBEDDINGS = "/api/embeddings"  # local (legacy shape)
OLLAMA_TAGS = "/api/tags"
OPENROUTER_CHAT = "/chat/completions"
OPENROUTER_MODELS = "/models"
OPENAI_EMBEDDINGS = "/embeddings"
OPENAI_MODELS = "/models"

# Per-dialect path that the canonical base must end with (empty = bare host root).
_DIALECT_SUFFIX = {"ollama": "", "openrouter": "/api/v1", "openai": "/v1"}

# Trailing path segments a user may paste; stripped longest-first.
_NOISE = ("/api/v1", "/v1", "/api")


def _strip_noise(base: str) -> str:
    changed = True
    while changed:
        changed = False
        low = base.lower()
        for seg in _NOISE:
            if low.endswith(seg):
                base = base[: -len(seg)].rstrip("/")
                changed = True
                break
    return base


def normalize_base(raw: str, *, dialect: str) -> str:
    """Reduce a user-entered base URL to its canonical form for ``dialect``.

    Empty input is returned unchanged so callers keep their existing
    empty-base handling.
    """
    if not raw or not raw.strip():
        return raw
    base = _strip_noise(raw.strip().rstrip("/"))
    return base + _DIALECT_SUFFIX.get(dialect, "")


def join(base: str, suffix: str, *, dialect: str) -> str:
    """Normalize ``base`` for ``dialect`` and append ``suffix`` exactly once."""
    nb = normalize_base(base, dialect=dialect)
    if not nb:
        return nb
    seg = "/" + suffix.strip("/")
    if nb.lower().endswith(seg.lower()):
        return nb
    return nb + seg
