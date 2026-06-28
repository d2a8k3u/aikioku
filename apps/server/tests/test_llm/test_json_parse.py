"""Tests for the tolerant LLM JSON parser (src.llm.json_parse)."""

from __future__ import annotations

import pytest

from src.llm.json_parse import LLMOutputParseError, parse_llm_json


class TestParseLLMJson:
    """parse_llm_json tolerates fences/prose and enforces the expected type."""

    def test_plain_array(self):
        result = parse_llm_json('[{"a": 1}, {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_plain_object(self):
        result = parse_llm_json('{"a": 1, "b": 2}')
        assert result == {"a": 1, "b": 2}

    def test_json_fenced_array(self):
        text = '```json\n[{"type": "qa", "front": "Q", "back": "A"}]\n```'
        result = parse_llm_json(text, expect="list")
        assert result == [{"type": "qa", "front": "Q", "back": "A"}]

    def test_bare_fenced_array(self):
        text = "```\n[1, 2, 3]\n```"
        result = parse_llm_json(text, expect="list")
        assert result == [1, 2, 3]

    def test_prose_wrapped_array(self):
        text = 'Here are the cards: [{"front": "Q", "back": "A"}] Hope this helps!'
        result = parse_llm_json(text, expect="list")
        assert result == [{"front": "Q", "back": "A"}]

    def test_prose_wrapped_object(self):
        text = 'Sure, here is the result: {"a": 1} done.'
        result = parse_llm_json(text, expect="dict")
        assert result == {"a": 1}

    def test_object_when_expect_list_raises(self):
        with pytest.raises(LLMOutputParseError):
            parse_llm_json('{"a": 1}', expect="list")

    def test_array_when_expect_dict_raises(self):
        with pytest.raises(LLMOutputParseError):
            parse_llm_json("[1, 2, 3]", expect="dict")

    def test_garbage_raises(self):
        with pytest.raises(LLMOutputParseError):
            parse_llm_json("this is not json at all", expect="list")

    def test_empty_string_raises(self):
        with pytest.raises(LLMOutputParseError):
            parse_llm_json("", expect="list")

    def test_no_expect_accepts_either(self):
        assert parse_llm_json("[1, 2]") == [1, 2]
        assert parse_llm_json('{"x": 1}') == {"x": 1}

    def test_error_is_value_error_subclass(self):
        assert issubclass(LLMOutputParseError, ValueError)

    def test_error_contains_snippet(self):
        with pytest.raises(LLMOutputParseError) as exc_info:
            parse_llm_json("garbage text here", expect="list")
        assert "garbage" in str(exc_info.value)
