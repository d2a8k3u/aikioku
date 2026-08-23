"""Semantic cache bounds: TTL on read + bounded eviction on write (FR-B1).

- TTL: entries older than ``_TTL`` seconds are treated as cache misses.
- Bound: ``cache_put`` prunes the oldest entries when the cache exceeds
  ``_MAX_ENTRIES`` (env ``SEMANTIC_CACHE_MAX_ENTRIES``, default 5000).

Tests run the real gptcache SSDataManager against an isolated cache DB in
``tmp_path``: the cache data dir is derived from ``os.path.expanduser``, so the
tests patch ``HOME`` before first cache init.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.cache import semantic_cache
from src.cache.semantic_cache import _MAX_ENTRIES, cache_get, cache_put


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path: Path, monkeypatch) -> Iterator[None]:
    """Point the semantic cache at a per-test data dir and reset module state."""
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "true")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SEMANTIC_CACHE_MAX_ENTRIES", "1000")

    semantic_cache._data_manager = None
    semantic_cache._onnx_embedding = None
    semantic_cache._similarity_eval = None
    semantic_cache._enabled = False

    yield

    semantic_cache._data_manager = None
    semantic_cache._onnx_embedding = None
    semantic_cache._similarity_eval = None
    semantic_cache._enabled = False


async def _drain_executor() -> None:
    """Run a no-op through the cache executor so queued pruning has completed."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(semantic_cache._executor, lambda: None)


def _question_count() -> int:
    dm = semantic_cache._data_manager
    return dm.s.count(is_all=True) if dm is not None else 0


async def _put(monkeypatch, query: str) -> None:
    await cache_put(query, "simple", "warm", f"answer-{query}", [], [], "openrouter:test-model")


@pytest.mark.asyncio
async def test_expired_entries_are_cache_misses(monkeypatch) -> None:
    """An entry older than _TTL seconds must never be served.

    Pruning runs on a background executor thread, so after the get returns a
    miss we drain the executor and assert the stale row is physically gone.
    """
    monkeypatch.setattr(semantic_cache, "_TTL", 0)

    await _put(monkeypatch, "expired entry")
    assert _question_count() == 1, "setup: entry must be stored"

    result = await cache_get("expired entry", "simple", "warm", "openrouter:test-model")
    assert result is None, "expired entry must be treated as a cache miss"

    await _drain_executor()
    assert _question_count() == 0, "expired entry must be purged"


@pytest.mark.asyncio
async def test_cache_is_bounded_to_max_entries(monkeypatch) -> None:
    """cache_put must evict the oldest entries once the bound is exceeded."""
    monkeypatch.setattr(semantic_cache, "_MAX_ENTRIES", 5)
    monkeypatch.setattr(semantic_cache, "_TTL", 3600)

    for i in range(20):
        await _put(monkeypatch, f"unique query {i}")

    await _drain_executor()
    assert _question_count() <= _MAX_ENTRIES, (
        f"cache must stay bounded to {_MAX_ENTRIES} entries "
        f"(found {_question_count()})"
    )

    # The most recent entries must survive eviction (oldest are evicted).
    assert await cache_get("unique query 19", "simple", "warm", "openrouter:test-model") is not None
