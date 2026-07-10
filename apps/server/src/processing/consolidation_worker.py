"""Background worker loop for hidden-notes consolidation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import structlog
from fastapi import FastAPI

from src import runtime_config
from src.processing import consolidation

logger = structlog.get_logger()


def _interval_elapsed(last_run_str: str | None, interval_hours: float) -> bool:
    """Return True if the interval has elapsed since ``last_run_str``.

    A missing or unparseable timestamp means "never run", so the worker is
    immediately eligible.
    """
    if not last_run_str:
        return True
    try:
        last_run = datetime.fromisoformat(last_run_str)
    except (ValueError, TypeError):
        return True
    return datetime.utcnow() - last_run >= timedelta(hours=interval_hours)


async def _consolidation_worker(app: FastAPI, poll_seconds: int = 300) -> None:
    """Polling loop that runs note consolidation when all guards pass.

    Guards (checked in order, cheapest first):
      1. ``auto_note_consolidation`` config flag is on
      2. ``is_configured`` — the setup wizard has completed
      3. interval has elapsed since the last run
      4. a cost tracker exists, is not exhausted, and has enough remaining budget
    """
    await asyncio.sleep(30)
    while True:
        try:
            if not runtime_config.auto_note_consolidation():
                await asyncio.sleep(poll_seconds)
                continue
            if not runtime_config.is_configured():
                await asyncio.sleep(poll_seconds)
                continue
            last_run = runtime_config.get_app_setting("consolidation_last_run")
            interval = runtime_config.consolidation_interval_hours()
            if last_run and not _interval_elapsed(last_run, interval):
                await asyncio.sleep(poll_seconds)
                continue

            tracker = getattr(app.state, "cost_tracker", None)
            if tracker is None:
                await asyncio.sleep(poll_seconds)
                continue
            if tracker.is_exhausted():
                logger.info("consolidation.skipped", reason="budget_exhausted")
                await asyncio.sleep(poll_seconds)
                continue
            if tracker.remaining() < runtime_config.consolidation_min_budget_remaining_usd():
                logger.info(
                    "consolidation.skipped",
                    reason="insufficient_budget",
                    remaining=tracker.remaining(),
                )
                await asyncio.sleep(poll_seconds)
                continue

            result = await consolidation.run_consolidation(app)
            runtime_config.set_app_setting("consolidation_last_run", datetime.utcnow().isoformat())
            logger.info("consolidation.completed", **result)
        except Exception:
            logger.exception("consolidation_worker.failed")
        await asyncio.sleep(poll_seconds)
