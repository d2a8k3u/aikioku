"""Tests for the background reembed engine (src.knowledge.reembed)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src import runtime_config
from src.config import settings
from src.knowledge import reembed
from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.reembed_fingerprint import effective_embedding_fingerprint


class FakeLLM:
    def __init__(self, dim: int, fail: bool = False) -> None:
        self.dim = dim
        self.fail = fail

    async def embed(self, text: str):
        if self.fail:
            from src.llm.ollama_remote import EmbeddingUnavailableError

            raise EmbeddingUnavailableError("down")
        return [0.05] * self.dim


def _note(nid: str, content: str, modified: datetime | None = None):
    return SimpleNamespace(id=nid, content=content, modified=modified or datetime(2020, 1, 1))


def _make_app(llm: FakeLLM, notes: list) -> SimpleNamespace:
    base = os.path.dirname(settings.sqlite_db_path)
    old_notes = EmbeddingStore(
        os.path.join(base, "chroma"), collection_name="notes", dimension=llm.dim
    )
    old_convs = EmbeddingStore(
        os.path.join(base, "chroma_conversations"),
        collection_name="conversations",
        dimension=llm.dim,
    )
    return SimpleNamespace(
        state=SimpleNamespace(
            configured=True,
            llm_provider=llm,
            note_store=SimpleNamespace(list_all=lambda: list(notes)),
            embedding_store=old_notes,
            conversation_embedding_store=old_convs,
            _reembed_task=None,
        )
    )


@pytest.fixture(autouse=True)
def _reset_reembed_state(monkeypatch):
    # Keep the swap step from triggering a full runtime rebuild.
    monkeypatch.setattr("src.main.build_runtime", lambda app: None)
    runtime_config.set_app_setting("embedding_dimension", "8")
    reembed._reembed_state.update(
        state="idle",
        target_fp=None,
        processed_notes=0,
        total_notes=0,
        processed_convs=0,
        total_convs=0,
        error=None,
    )
    reembed._current_target_fp = None
    yield


async def test_worker_happy_path_builds_swaps_deletes(monkeypatch):
    llm = FakeLLM(dim=8)
    monkeypatch.setattr("src.llm.factory.build_embedding_provider", lambda *a, **k: llm)
    app = _make_app(llm, [_note("n1", "alpha"), _note("n2", "beta")])
    fp = "deadbeef00000001"
    reembed._current_target_fp = fp

    await reembed._reembed_worker(app, fp)

    assert reembed._reembed_state["state"] == "idle"
    assert reembed._reembed_state["processed_notes"] == 2
    assert app.state.embedding_store.collection_name == f"notes__{fp}"
    assert app.state.embedding_store.count() >= 2
    assert runtime_config.get_app_setting("embedding_active_fingerprint") == fp
    assert runtime_config.get_app_setting("embedding_notes_collection") == f"notes__{fp}"


def test_as_aware_utc_coerces_naive_and_passes_aware_through():
    naive = datetime(2020, 1, 1)
    aware = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert reembed._as_aware_utc(naive).tzinfo is timezone.utc
    assert reembed._as_aware_utc(aware) is aware


async def test_worker_catchup_tzaware_modified_does_not_crash(monkeypatch):
    # A note whose `modified` is tz-aware lands in the catch-up window; the
    # comparison against the (now aware) start time must not raise TypeError.
    llm = FakeLLM(dim=8)
    monkeypatch.setattr("src.llm.factory.build_embedding_provider", lambda *a, **k: llm)
    future_aware = datetime(2999, 1, 1, tzinfo=timezone.utc)
    app = _make_app(llm, [_note("n1", "alpha", modified=future_aware)])
    fp = "deadbeef00000002"
    reembed._current_target_fp = fp

    await reembed._reembed_worker(app, fp)

    assert reembed._reembed_state["state"] == "idle"
    assert reembed._reembed_state["error"] is None


async def test_worker_probe_wrong_dim_keeps_old(monkeypatch):
    llm = FakeLLM(dim=4)  # target dimension is 8 -> probe length mismatch
    monkeypatch.setattr("src.llm.factory.build_embedding_provider", lambda *a, **k: llm)
    app = _make_app(llm, [_note("n1", "alpha")])
    old = app.state.embedding_store
    reembed._current_target_fp = "fp_wrongdim"

    await reembed._reembed_worker(app, "fp_wrongdim")

    assert reembed._reembed_state["state"] == "failed"
    assert app.state.embedding_store is old  # never swapped


async def test_worker_provider_down_keeps_old(monkeypatch):
    llm = FakeLLM(dim=8, fail=True)
    monkeypatch.setattr("src.llm.factory.build_embedding_provider", lambda *a, **k: llm)
    app = _make_app(llm, [_note("n1", "alpha")])
    old = app.state.embedding_store
    reembed._current_target_fp = "fp_down"

    await reembed._reembed_worker(app, "fp_down")

    assert reembed._reembed_state["state"] == "failed"
    assert app.state.embedding_store is old


async def test_worker_superseded_does_not_swap(monkeypatch):
    llm = FakeLLM(dim=8)
    monkeypatch.setattr("src.llm.factory.build_embedding_provider", lambda *a, **k: llm)
    app = _make_app(llm, [_note("n1", "alpha")])
    old = app.state.embedding_store
    # A newer target is current -> the run for the older fp must bail before swap.
    reembed._current_target_fp = "NEWER"

    await reembed._reembed_worker(app, "OLDER")

    assert app.state.embedding_store is old


def test_current_fingerprint_mismatch():
    runtime_config.set_app_setting("llm_provider", "ollama_remote")
    runtime_config.set_app_setting("ollama_embedding_model", "m1")
    cur = effective_embedding_fingerprint()
    runtime_config.set_app_setting("embedding_active_fingerprint", cur)
    mism, got = reembed.current_fingerprint_mismatch()
    assert not mism and got == cur

    runtime_config.set_app_setting("ollama_embedding_model", "m2")
    mism2, _ = reembed.current_fingerprint_mismatch()
    assert mism2


def test_maybe_schedule_on_mismatch(monkeypatch):
    captured = {}
    monkeypatch.setattr(reembed, "schedule_reembed", lambda app, fp: captured.update(fp=fp))
    runtime_config.set_app_setting("embedding_active_fingerprint", "OLD_FP")
    app = SimpleNamespace(state=SimpleNamespace(configured=True, llm_provider=object()))
    assert reembed.maybe_schedule_reembed(app) is True
    assert "fp" in captured


def test_maybe_schedule_noop_when_matching(monkeypatch):
    def _boom(app, fp):
        raise AssertionError("should not schedule when fingerprint matches and idle")

    monkeypatch.setattr(reembed, "schedule_reembed", _boom)
    cur = effective_embedding_fingerprint()
    runtime_config.set_app_setting("embedding_active_fingerprint", cur)
    runtime_config.set_app_setting("reembed_status", '{"state": "idle"}')
    app = SimpleNamespace(state=SimpleNamespace(configured=True, llm_provider=object()))
    assert reembed.maybe_schedule_reembed(app) is False


def test_maybe_schedule_noop_when_unconfigured(monkeypatch):
    monkeypatch.setattr(reembed, "schedule_reembed", lambda app, fp: pytest.fail("unconfigured"))
    app = SimpleNamespace(state=SimpleNamespace(configured=False, llm_provider=None))
    assert reembed.maybe_schedule_reembed(app) is False
