"""The semantic cache key + hit validation must be model-aware.

A model switch must never serve a previous model's cached answer, so the active
model is part of the cache key and is validated on hit (like mode/tone).
"""

from __future__ import annotations

import pytest

from src.cache import semantic_cache
from src.cache.semantic_cache import _build_cache_key, cache_get, cache_put


def test_cache_key_is_model_aware() -> None:
    a = _build_cache_key("q", "simple", "warm", "openrouter:model-a")
    b = _build_cache_key("q", "simple", "warm", "openrouter:model-b")
    same = _build_cache_key("q", "simple", "warm", "openrouter:model-a")
    assert a != b, "different models must produce different cache keys"
    assert a == same, "same inputs must be deterministic"


@pytest.mark.asyncio
async def test_get_put_accept_model_and_degrade_when_disabled(monkeypatch) -> None:
    """With the backend unavailable the cache is a graceful no-op — and both
    entry points take the model argument."""

    async def _disabled():
        return None

    monkeypatch.setattr(semantic_cache, "_init_cache", _disabled)

    assert await cache_get("q", "simple", "warm", "openrouter:model-a") is None
    # cache_put must not raise when the cache is disabled.
    await cache_put("q", "simple", "warm", "answer", [], [], "openrouter:model-a")
