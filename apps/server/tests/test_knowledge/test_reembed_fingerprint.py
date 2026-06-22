"""Tests for the effective embedding fingerprint."""
from __future__ import annotations

from src import runtime_config
from src.knowledge.reembed_fingerprint import effective_embedding_fingerprint as fp


def _set(**kw) -> None:
    for k, v in kw.items():
        runtime_config.set_app_setting(k, str(v))


def _remote_baseline() -> None:
    _set(
        embedding_provider="ollama",
        ollama_embedding_model="m1",
        ollama_embedding_base_url="http://host:11434",
        embedding_dimension=1024,
        embedding_strict="true",
    )


def test_changes_on_dimension():
    _remote_baseline()
    before = fp()
    _set(embedding_dimension=512)
    assert fp() != before


def test_changes_on_remote_embedding_model():
    _remote_baseline()
    before = fp()
    _set(ollama_embedding_model="m2")
    assert fp() != before


def test_changes_on_remote_base_url():
    _remote_baseline()
    before = fp()
    _set(ollama_embedding_base_url="http://other:11434")
    assert fp() != before


def test_changes_on_provider_switch():
    _remote_baseline()
    before = fp()
    _set(
        embedding_provider="openai",
        openai_embedding_model="text-embedding-3-small",
        openai_base_url="https://api.openai.com",
    )
    assert fp() != before


def test_openai_changes_on_model():
    _set(
        embedding_provider="openai",
        openai_embedding_model="text-embedding-3-small",
        openai_base_url="https://api.openai.com",
        embedding_dimension=1024,
    )
    before = fp()
    _set(openai_embedding_model="text-embedding-3-large")
    assert fp() != before


def test_huggingface_changes_on_model():
    _set(embedding_provider="huggingface", hf_embedding_model="m1", embedding_dimension=1024)
    before = fp()
    _set(hf_embedding_model="m2")
    assert fp() != before


def test_stable_on_strict_toggle():
    _remote_baseline()
    before = fp()
    _set(embedding_strict="false")
    assert fp() == before
