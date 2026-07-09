"""Tests for the consolidation worker loop — guard checks, interval logic, resilience."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.processing.consolidation_worker import _consolidation_worker, _interval_elapsed

# Sentinel raised inside the mocked ``asyncio.sleep`` to break the infinite
# worker loop after one iteration, without leaking StopIteration into the
# coroutine machinery (which Python wraps in a RuntimeError).
class _LoopBreak(Exception):
    pass


# --------------------------------------------------------------------------- #
# _interval_elapsed helper
# --------------------------------------------------------------------------- #


class TestIntervalElapsed:
    def test_missing_timestamp_returns_true(self):
        assert _interval_elapsed(None, 6.0) is True

    def test_recent_timestamp_returns_false(self):
        recent = datetime.utcnow().isoformat()
        assert _interval_elapsed(recent, 6.0) is False

    def test_old_timestamp_returns_true(self):
        old = (datetime.utcnow() - timedelta(hours=10)).isoformat()
        assert _interval_elapsed(old, 6.0) is True

    def test_unparseable_timestamp_returns_true(self):
        assert _interval_elapsed("not-a-timestamp", 6.0) is True

    def test_boundary_timestamp_returns_true(self):
        boundary = (datetime.utcnow() - timedelta(hours=6)).isoformat()
        assert _interval_elapsed(boundary, 6.0) is True


# --------------------------------------------------------------------------- #
# _consolidation_worker
# --------------------------------------------------------------------------- #


def _make_app() -> MagicMock:
    """A minimal FastAPI-like mock with a MagicMock state."""
    app = MagicMock()
    app.state = MagicMock()
    app.state.cost_tracker = MagicMock()
    app.state.cost_tracker.is_exhausted = MagicMock(return_value=False)
    app.state.cost_tracker.remaining = MagicMock(return_value=10.0)
    return app


async def _run_one_iteration(app, monkeypatch, **overrides):
    """Run the worker for exactly one loop iteration, returning call spies.

    Patches ``asyncio.sleep`` so the second call (after the initial settle
    delay) raises ``StopIteration`` — breaking the infinite loop after the
    first body runs.
    """
    import asyncio

    from src import runtime_config
    from src.processing import consolidation

    defaults = {
        "auto_consolidation": True,
        "is_configured": True,
        "interval_hours": 6.0,
        "last_run": None,
        "min_budget": 0.50,
    }
    defaults.update(overrides)

    monkeypatch.setattr(
        runtime_config, "auto_note_consolidation", lambda: defaults["auto_consolidation"]
    )
    monkeypatch.setattr(runtime_config, "is_configured", lambda: defaults["is_configured"])
    monkeypatch.setattr(
        runtime_config,
        "consolidation_interval_hours",
        lambda: defaults["interval_hours"],
    )
    monkeypatch.setattr(
        runtime_config,
        "get_app_setting",
        lambda key: defaults["last_run"] if key == "consolidation_last_run" else None,
    )
    monkeypatch.setattr(
        runtime_config,
        "consolidation_min_budget_remaining_usd",
        lambda: defaults["min_budget"],
    )
    set_calls: list[tuple[str, str]] = []
    settings_state: dict[str, str] = {}
    if defaults["last_run"] is not None:
        settings_state["consolidation_last_run"] = defaults["last_run"]

    def _fake_get(key: str):
        return settings_state.get(key)

    def _fake_set(key: str, value: str) -> None:
        settings_state[key] = value
        set_calls.append((key, value))

    monkeypatch.setattr(runtime_config, "get_app_setting", _fake_get)
    monkeypatch.setattr(runtime_config, "set_app_setting", _fake_set)

    run_consolidation_mock = AsyncMock(
        return_value={"candidates": 5, "clusters": 1, "merged": 1, "deleted": 2, "skipped": 0}
    )
    monkeypatch.setattr(consolidation, "run_consolidation", run_consolidation_mock)

    call_count = {"n": 0}

    async def _sleep(_seconds):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            raise _LoopBreak

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    try:
        await _consolidation_worker(app, poll_seconds=0)
    except _LoopBreak:
        pass

    return run_consolidation_mock, set_calls


@pytest.mark.asyncio
async def test_worker_flag_off_no_mutations(monkeypatch):
    app = _make_app()
    run_mock, set_calls = await _run_one_iteration(
        app, monkeypatch, auto_consolidation=False
    )
    run_mock.assert_not_called()
    assert set_calls == []


@pytest.mark.asyncio
async def test_worker_not_configured_skips(monkeypatch):
    app = _make_app()
    run_mock, set_calls = await _run_one_iteration(
        app, monkeypatch, is_configured=False
    )
    run_mock.assert_not_called()
    assert set_calls == []


@pytest.mark.asyncio
async def test_worker_interval_not_elapsed_skips(monkeypatch):
    app = _make_app()
    recent = datetime.utcnow().isoformat()
    run_mock, set_calls = await _run_one_iteration(
        app, monkeypatch, last_run=recent
    )
    run_mock.assert_not_called()
    assert set_calls == []


@pytest.mark.asyncio
async def test_worker_budget_exhausted_skips(monkeypatch):
    app = _make_app()
    app.state.cost_tracker.is_exhausted = MagicMock(return_value=True)
    run_mock, set_calls = await _run_one_iteration(app, monkeypatch)
    run_mock.assert_not_called()
    assert set_calls == []


@pytest.mark.asyncio
async def test_worker_insufficient_budget_skips(monkeypatch):
    app = _make_app()
    app.state.cost_tracker.remaining = MagicMock(return_value=0.10)
    run_mock, set_calls = await _run_one_iteration(
        app, monkeypatch, min_budget=0.50
    )
    run_mock.assert_not_called()
    assert set_calls == []


@pytest.mark.asyncio
async def test_worker_runs_when_all_conditions_met(monkeypatch):
    app = _make_app()
    run_mock, set_calls = await _run_one_iteration(app, monkeypatch)
    run_mock.assert_awaited_once_with(app)
    assert len(set_calls) == 1
    key, value = set_calls[0]
    assert key == "consolidation_last_run"
    datetime.fromisoformat(value)


@pytest.mark.asyncio
async def test_worker_updates_last_run_after_success(monkeypatch):
    app = _make_app()
    _, set_calls = await _run_one_iteration(app, monkeypatch)
    assert len(set_calls) == 1
    key, value = set_calls[0]
    assert key == "consolidation_last_run"
    parsed = datetime.fromisoformat(value)
    assert (datetime.utcnow() - parsed) < timedelta(minutes=1)


@pytest.mark.asyncio
async def test_worker_exception_does_not_crash_loop(monkeypatch):
    import asyncio

    from src import runtime_config
    from src.processing import consolidation

    monkeypatch.setattr(runtime_config, "auto_note_consolidation", lambda: True)
    monkeypatch.setattr(runtime_config, "is_configured", lambda: True)
    monkeypatch.setattr(
        runtime_config, "consolidation_interval_hours", lambda: 6.0
    )
    monkeypatch.setattr(
        runtime_config, "get_app_setting", lambda key: None
    )
    monkeypatch.setattr(
        runtime_config, "consolidation_min_budget_remaining_usd", lambda: 0.50
    )
    monkeypatch.setattr(runtime_config, "set_app_setting", lambda k, v: None)

    error = RuntimeError("boom")
    failing_mock = AsyncMock(side_effect=error)
    monkeypatch.setattr(consolidation, "run_consolidation", failing_mock)

    app = _make_app()
    call_count = {"n": 0}

    async def _sleep(_seconds):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            raise _LoopBreak

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    try:
        await _consolidation_worker(app, poll_seconds=0)
    except _LoopBreak:
        pass

    assert failing_mock.await_count >= 1