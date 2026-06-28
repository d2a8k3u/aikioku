"""Tests for non-truncating long-text processing helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock

from src.llm.long_text import chunk_for_llm, condense_for_prompt


class TestChunkForLLM:
    def test_empty_returns_no_chunks(self):
        assert chunk_for_llm("") == []
        assert chunk_for_llm("   ") == []

    def test_small_text_is_single_chunk(self):
        assert chunk_for_llm("One sentence.") == ["One sentence."]

    def test_long_text_splits_without_dropping_content(self):
        sentences = [f"Fact number {i} is true." for i in range(2000)]
        text = " ".join(sentences)
        chunks = chunk_for_llm(text, target_chars=500)
        assert len(chunks) > 1
        joined = " ".join(chunks)
        # No content lost: sentences from start, middle and end all survive.
        for i in (0, 999, 1999):
            assert f"Fact number {i} is true." in joined

    def test_oversized_single_sentence_kept_whole(self):
        sentence = "x" * 1000  # no boundary, longer than target — must not be cut
        assert chunk_for_llm(sentence, target_chars=100) == [sentence]


class TestCondenseForPrompt:
    async def test_short_text_returned_verbatim_without_llm_call(self):
        llm = AsyncMock()
        text = "Short note content."
        result = await condense_for_prompt(llm, text)
        assert result == text
        llm.complete.assert_not_awaited()

    async def test_oversized_text_is_folded(self):
        llm = AsyncMock()
        llm.complete.return_value = "SUMMARY"
        text = "A. " * 6000  # ~18k chars, well over the budget
        result = await condense_for_prompt(llm, text, target_chars=1000)
        assert llm.complete.await_count >= 1
        assert "SUMMARY" in result

    async def test_fold_failure_keeps_content_verbatim(self):
        llm = AsyncMock()
        llm.complete.side_effect = RuntimeError("boom")
        text = ("Sentence alpha. " * 100) + ("Sentence beta. " * 100)
        result = await condense_for_prompt(llm, text, target_chars=300)
        # No truncation: original content survives even when every fold call fails.
        assert "Sentence alpha." in result
        assert "Sentence beta." in result
