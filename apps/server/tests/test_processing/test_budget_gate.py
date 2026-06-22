"""Budget-state helpers, the deferred-work queue, and the gate/drain logic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import settings
from src.llm.router import CostRecord, CostTracker


def _tracker(tmp_path, budget: float) -> CostTracker:
    return CostTracker(str(tmp_path / "costs.db"), daily_budget_usd=budget)


def _spend(ct: CostTracker, amount: float) -> None:
    ct.record(CostRecord(provider="x", model="m", cost_usd=amount))


class TestCostTrackerBudgetState:
    def test_remaining_and_fraction(self, tmp_path):
        ct = _tracker(tmp_path, 10.0)
        assert ct.remaining() == 10.0
        assert ct.fraction() == 0.0
        _spend(ct, 4.0)
        assert ct.remaining() == 6.0
        assert ct.fraction() == pytest.approx(0.4)

    def test_is_exhausted_at_limit(self, tmp_path):
        ct = _tracker(tmp_path, 5.0)
        assert ct.is_exhausted() is False
        _spend(ct, 5.0)
        assert ct.is_exhausted() is True

    def test_warning_band_then_exhausted(self, tmp_path):
        ct = _tracker(tmp_path, 10.0)
        _spend(ct, 9.0)
        assert ct.is_warning(0.9) is True
        assert ct.is_exhausted() is False
        _spend(ct, 1.0)  # now exactly at the limit
        assert ct.is_warning(0.9) is False
        assert ct.is_exhausted() is True

    def test_zero_budget_means_unlimited(self, tmp_path):
        ct = _tracker(tmp_path, 0.0)
        _spend(ct, 100.0)
        assert ct.is_exhausted() is False
        assert ct.is_warning(0.9) is False
        assert ct.fraction() == 0.0
        assert ct.remaining() == 0.0


class TestPendingWork:
    def test_enqueue_count_list_delete(self):
        from src.processing import pending_work

        assert pending_work.count() == 0
        pending_work.enqueue("note_processing", "n1", {"note_id": "n1"})
        pending_work.enqueue("note_processing", "n2", {"note_id": "n2"})
        assert pending_work.count() == 2
        items = pending_work.list_pending()
        assert {i["entity_id"] for i in items} == {"n1", "n2"}
        assert items[0]["payload"] == {"note_id": "n1"}
        pending_work.delete(items[0]["id"])
        assert pending_work.count() == 1

    def test_enqueue_dedupes_on_key(self):
        from src.processing import pending_work

        pending_work.enqueue("note_processing", "n1", {"note_id": "n1", "v": 1})
        pending_work.enqueue("note_processing", "n1", {"note_id": "n1", "v": 2})
        assert pending_work.count() == 1
        assert pending_work.list_pending()[0]["payload"]["v"] == 2

    def test_bump_attempt_drops_at_cap(self):
        from src.processing import pending_work

        pending_work.enqueue("note_processing", "n1", {"note_id": "n1"})
        item_id = pending_work.list_pending()[0]["id"]
        for _ in range(4):
            pending_work.bump_attempt(item_id, "boom")
        assert pending_work.count() == 1  # 4 < cap of 5
        pending_work.bump_attempt(item_id, "boom")  # 5th attempt -> dropped
        assert pending_work.count() == 0


def _app(tracker: CostTracker) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(cost_tracker=tracker, _last_budget_state=None))


class TestGatedDrain:
    async def test_gated_runs_when_budget_ok(self):
        from src.processing import budget_gate, pending_work

        app = _app(CostTracker(settings.sqlite_db_path, daily_budget_usd=10.0))
        seen = {}

        async def runner(_app, payload):
            seen["payload"] = payload
            return ["ok"]

        budget_gate.RUNNERS["__test__"] = runner
        try:
            result = await budget_gate.gated(app, "__test__", "e1", {"x": 1})
        finally:
            budget_gate.RUNNERS.pop("__test__", None)
        assert result == ["ok"]
        assert seen["payload"] == {"x": 1}
        assert pending_work.count() == 0

    async def test_gated_enqueues_when_exhausted(self):
        from src.processing import budget_gate, pending_work

        ct = CostTracker(settings.sqlite_db_path, daily_budget_usd=5.0)
        _spend(ct, 5.0)
        app = _app(ct)
        calls = {"n": 0}

        async def runner(_app, payload):
            calls["n"] += 1

        budget_gate.RUNNERS["__test__"] = runner
        try:
            result = await budget_gate.gated(app, "__test__", "e1", {"x": 1})
        finally:
            budget_gate.RUNNERS.pop("__test__", None)
        assert result is None
        assert calls["n"] == 0
        assert pending_work.count() == 1

    async def test_drain_runs_queue_when_funded(self):
        from src.processing import budget_gate, pending_work

        app = _app(CostTracker(settings.sqlite_db_path, daily_budget_usd=10.0))
        pending_work.enqueue("__test__", "e1", {"x": 1})
        pending_work.enqueue("__test__", "e2", {"x": 2})
        seen: list[int] = []

        async def runner(_app, payload):
            seen.append(payload["x"])

        budget_gate.RUNNERS["__test__"] = runner
        try:
            processed = await budget_gate.drain(app)
        finally:
            budget_gate.RUNNERS.pop("__test__", None)
        assert processed == 2
        assert sorted(seen) == [1, 2]
        assert pending_work.count() == 0

    async def test_drain_stops_while_exhausted(self):
        from src.processing import budget_gate, pending_work

        ct = CostTracker(settings.sqlite_db_path, daily_budget_usd=5.0)
        _spend(ct, 5.0)
        app = _app(ct)
        pending_work.enqueue("__test__", "e1", {"x": 1})

        async def runner(_app, payload):
            raise AssertionError("runner must not fire while exhausted")

        budget_gate.RUNNERS["__test__"] = runner
        try:
            processed = await budget_gate.drain(app)
        finally:
            budget_gate.RUNNERS.pop("__test__", None)
        assert processed == 0
        assert pending_work.count() == 1
