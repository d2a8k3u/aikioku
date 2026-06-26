"""Tests for LLMLingua-2 prompt compression and its fallbacks."""

from __future__ import annotations

from typing import Any

import pytest

from src.llm import compression
from src.llm.compression import compress_for_prompt


class _FakeCompressor:
    """Records the kwargs it was called with and returns a fixed result."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def compress_prompt(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"text": text, **kwargs})
        return self.result


@pytest.mark.unit
async def test_empty_text_returned_unchanged() -> None:
    assert await compress_for_prompt("") == ""
    assert await compress_for_prompt("   ") == ""


@pytest.mark.unit
async def test_short_text_short_circuits_without_compressor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Under the char budget (target*4) → no compressor is even consulted.
    monkeypatch.setattr(
        compression, "_get_compressor", lambda: pytest.fail("compressor must not load")
    )
    short = "a short note"
    assert await compress_for_prompt(short, target_tokens=2000) == short


@pytest.mark.unit
async def test_disabled_returns_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPRESSION_ENABLED", "false")
    text = "x" * 500
    assert await compress_for_prompt(text, target_tokens=10) == text


@pytest.mark.unit
async def test_unavailable_compressor_returns_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compression, "_get_compressor", lambda: None)
    text = "x" * 500
    assert await compress_for_prompt(text, target_tokens=10) == text


@pytest.mark.unit
async def test_compression_failure_falls_back_to_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def compress_prompt(self, *_a: Any, **_k: Any) -> dict[str, Any]:
            raise RuntimeError("model exploded")

    monkeypatch.setattr(compression, "_get_compressor", lambda: _Boom())
    text = "x" * 500
    assert await compress_for_prompt(text, target_tokens=10) == text


@pytest.mark.unit
async def test_successful_compression_uses_llmlingua2_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCompressor({"compressed_prompt": "  squeezed  "})
    monkeypatch.setattr(compression, "_get_compressor", lambda: fake)

    text = "y" * 500
    out = await compress_for_prompt(text, query="ignored", target_tokens=10)

    assert out == "squeezed"
    # Guard against the transformers-5.x regression: the call must use the
    # llmlingua-2 signature (target_token), never the removed gpt2-only kwargs.
    (call,) = fake.calls
    assert call["text"] == text
    assert call["target_token"] == 10
    assert "force_tokens" in call
    assert "return_compressed_text" not in call
    assert "condition_in_question" not in call


@pytest.mark.unit
async def test_empty_compressed_result_falls_back_to_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compression, "_get_compressor", lambda: _FakeCompressor({"compressed_prompt": "   "})
    )
    text = "z" * 500
    assert await compress_for_prompt(text, target_tokens=10) == text


@pytest.mark.integration
@pytest.mark.slow
async def test_real_llmlingua2_compresses_long_text() -> None:
    """Exercises the real model — downloads ~560MB on first run."""
    long = (
        "Konfigurace serveru a bezpečnostní detaily jsou důležité. "
        "Retrieval runs over the knowledge graph and the vector store. "
    ) * 70
    out = await compress_for_prompt(long, query="bezpečnost?", target_tokens=200)
    assert 0 < len(out) < len(long)
    assert out != long
