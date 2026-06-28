"""Tests for base-URL normalization + endpoint joining."""

from __future__ import annotations

import pytest

from src.llm import urls


class TestNormalizeOllama:
    @pytest.mark.parametrize(
        "raw",
        [
            "https://api.ollama.com",
            "https://api.ollama.com/",
            "https://api.ollama.com/api",
            "https://api.ollama.com/api/",
            "https://api.ollama.com/v1",
            "https://api.ollama.com/api/v1",
            "https://api.ollama.com/api/v1/",
        ],
    )
    def test_collapses_to_bare_root(self, raw):
        assert urls.normalize_base(raw, dialect="ollama") == "https://api.ollama.com"

    def test_local_root_unchanged(self):
        assert (
            urls.normalize_base("http://localhost:11434", dialect="ollama")
            == "http://localhost:11434"
        )

    def test_messy_slashes(self):
        assert urls.normalize_base("https://host//api//", dialect="ollama") == "https://host"

    def test_empty_passthrough(self):
        assert urls.normalize_base("", dialect="ollama") == ""


class TestNormalizeOpenRouter:
    @pytest.mark.parametrize(
        "raw",
        [
            "https://openrouter.ai",
            "https://openrouter.ai/",
            "https://openrouter.ai/api",
            "https://openrouter.ai/v1",
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/api/v1/",
        ],
    )
    def test_canonical_includes_api_v1_once(self, raw):
        assert urls.normalize_base(raw, dialect="openrouter") == "https://openrouter.ai/api/v1"


class TestNormalizeOpenAI:
    @pytest.mark.parametrize(
        "raw",
        [
            "https://api.openai.com",
            "https://api.openai.com/",
            "https://api.openai.com/v1",
            "https://api.openai.com/v1/",
            "https://api.openai.com/api/v1",
        ],
    )
    def test_canonical_includes_v1_once(self, raw):
        assert urls.normalize_base(raw, dialect="openai") == "https://api.openai.com/v1"

    def test_join_embeddings(self):
        assert (
            urls.join("https://api.openai.com", urls.OPENAI_EMBEDDINGS, dialect="openai")
            == "https://api.openai.com/v1/embeddings"
        )

    def test_join_models_from_v1_base(self):
        assert (
            urls.join("https://api.openai.com/v1", urls.OPENAI_MODELS, dialect="openai")
            == "https://api.openai.com/v1/models"
        )


class TestJoinOllama:
    @pytest.mark.parametrize(
        "raw",
        ["https://api.ollama.com", "https://api.ollama.com/api", "https://api.ollama.com/v1"],
    )
    def test_tags(self, raw):
        assert (
            urls.join(raw, urls.OLLAMA_TAGS, dialect="ollama") == "https://api.ollama.com/api/tags"
        )

    def test_generate(self):
        assert (
            urls.join("https://api.ollama.com/", urls.OLLAMA_GENERATE, dialect="ollama")
            == "https://api.ollama.com/api/generate"
        )

    def test_embed_messy_base_resolves_canonically(self):
        assert (
            urls.join("http://host.docker.internal:11434/api", urls.OLLAMA_EMBED, dialect="ollama")
            == "http://host.docker.internal:11434/api/embed"
        )

    def test_no_double_suffix(self):
        assert (
            urls.join("http://localhost:11434/api/tags", urls.OLLAMA_TAGS, dialect="ollama")
            == "http://localhost:11434/api/tags"
        )


class TestJoinOpenRouter:
    @pytest.mark.parametrize(
        "raw",
        ["https://openrouter.ai", "https://openrouter.ai/api/v1", "https://openrouter.ai/v1"],
    )
    def test_models(self, raw):
        assert (
            urls.join(raw, urls.OPENROUTER_MODELS, dialect="openrouter")
            == "https://openrouter.ai/api/v1/models"
        )

    def test_chat(self):
        assert (
            urls.join("https://openrouter.ai", urls.OPENROUTER_CHAT, dialect="openrouter")
            == "https://openrouter.ai/api/v1/chat/completions"
        )
