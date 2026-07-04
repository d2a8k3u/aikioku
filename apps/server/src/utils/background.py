"""Background task utilities for FastAPI applications.

Provides a safe way to spawn fire-and-forget coroutines that won't be
silently garbage-collected by Python 3.12+'s weak task references.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from fastapi import FastAPI


def spawn_background(app: FastAPI, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Spawn a fire-and-forget background task with strong reference.

    Python 3.12+ holds only weak references to asyncio tasks, so
    ensure_future/create_task can be silently GC'd. This keeps a strong
    reference in app.state._bg_tasks and auto-removes on completion.
    """
    task = asyncio.ensure_future(coro)
    bg = getattr(app.state, "_bg_tasks", None)
    if bg is not None:
        bg.add(task)
        task.add_done_callback(bg.discard)
    return task
