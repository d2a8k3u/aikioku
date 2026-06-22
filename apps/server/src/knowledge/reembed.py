"""Background reembed engine.

When the effective embedding configuration changes (see ``reembed_fingerprint``),
every stored vector is stale. This rebuilds the notes and conversation vector
stores in the background with NO search blackout:

1. Validate the new embedding works (probe) BEFORE touching any data.
2. Build into NEW Chroma collections named by the target fingerprint, while the
   old collections keep serving queries.
3. Atomically swap ``app.state`` to the new stores (rebuilding the cached
   retrievers via ``build_runtime``), then delete the old collections.

The job is single-flight (a newer config supersedes a running one), survives
restarts (status + active fingerprint persisted in ``app_settings``), and never
deletes old data on failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from src import runtime_config
from src.config import settings
from src.knowledge.embeddings import EmbeddingStore
from src.knowledge.reembed_fingerprint import effective_embedding_fingerprint
from src.llm.ollama_remote import _deterministic_embedding

logger = logging.getLogger(__name__)

_REEMBED_CONCURRENCY = 4
_PROGRESS_EVERY = 10  # persist + emit progress every N processed items
_EMBED_RETRIES = 3  # attempts per item before counting a hard failure
_ABORT_AFTER_FAILURES = 25  # total hard failures => treat as a real outage, abort

# Module-level state (mirrors api/admin.py reextract precedent).
_reembed_state: dict = {
    "state": "idle",  # idle | running | failed
    "target_fp": None,
    "processed_notes": 0,
    "total_notes": 0,
    "processed_convs": 0,
    "total_convs": 0,
    "error": None,
}
_reembed_task: "asyncio.Task | None" = None
_current_target_fp: "str | None" = None


# --- public API -----------------------------------------------------------------


def get_status() -> dict:
    return dict(_reembed_state)


def current_fingerprint_mismatch() -> tuple[bool, str]:
    """Return (mismatch, current_fingerprint)."""
    current = effective_embedding_fingerprint()
    active = runtime_config.get_app_setting("embedding_active_fingerprint")
    return (active != current, current)


def schedule_reembed(app, target_fp: str) -> dict:
    """Single-flight + supersede entry point. Returns immediately."""
    global _reembed_task, _current_target_fp
    if _reembed_task is not None and not _reembed_task.done():
        if _current_target_fp == target_fp:
            return {"status": "already_running", **get_status()}
        _reembed_task.cancel()
    _current_target_fp = target_fp
    _reembed_task = asyncio.create_task(_reembed_worker(app, target_fp))
    app.state._reembed_task = _reembed_task
    return {"status": "started", "target_fp": target_fp}


def maybe_schedule_reembed(app) -> bool:
    """Schedule a reembed if configured and the stored vectors are stale.

    Triggers on a fingerprint mismatch, or to resume/retry an interrupted
    (``running``) or previously ``failed`` run. Safe to call on every settings
    change and at startup — a no-op when nothing needs rebuilding.
    """
    if not getattr(app.state, "configured", False):
        return False
    if getattr(app.state, "llm_provider", None) is None:
        return False
    mismatch, current_fp = current_fingerprint_mismatch()
    try:
        status = json.loads(runtime_config.get_app_setting("reembed_status") or "{}")
    except Exception:
        status = {}
    if mismatch or status.get("state") in ("running", "failed"):
        schedule_reembed(app, current_fp)
        return True
    return False


# --- internals ------------------------------------------------------------------


def _persist_status() -> None:
    runtime_config.set_app_setting("reembed_status", json.dumps(_reembed_state))


def _fail(app, msg: str) -> None:
    _reembed_state.update(state="failed", error=msg)
    _persist_status()
    logger.warning("reembed failed: %s", msg)
    _emit(app, "reembed.failed", {"target_fp": _reembed_state["target_fp"], "error": msg})


def _emit(app, event_type: str, data: dict) -> None:
    from src.api.websocket import get_broadcaster

    bc = get_broadcaster()
    if bc is None:
        return
    # broadcast() is async; fire-and-forget so the worker never blocks on a slow WS.
    asyncio.ensure_future(bc.broadcast(event_type, data))


def _progress_payload(target_fp: str) -> dict:
    return {
        "target_fp": target_fp,
        "processed_notes": _reembed_state["processed_notes"],
        "total_notes": _reembed_state["total_notes"],
        "processed_convs": _reembed_state["processed_convs"],
        "total_convs": _reembed_state["total_convs"],
    }


def _emb_paths() -> tuple[str, str]:
    base = os.path.dirname(settings.sqlite_db_path)
    return (os.path.join(base, "chroma"), os.path.join(base, "chroma_conversations"))


def _as_aware_utc(dt: datetime) -> datetime:
    """Coerce a possibly-naive datetime to aware UTC.

    Note timestamps may be naive or tz-aware depending on how the note was
    created, so the catch-up window comparison must normalize both sides or it
    raises ``TypeError`` on a naive/aware mix.
    """
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


async def _embed_with_retry(embedder, text: str) -> list[float]:
    """Embed with bounded retries to ride out a transient endpoint blip."""
    last: Exception | None = None
    for attempt in range(_EMBED_RETRIES):
        try:
            return await embedder.embed(text)
        except Exception as exc:  # noqa: BLE001
            last = exc
            await asyncio.sleep(0.5 * (attempt + 1))
    raise last if last is not None else RuntimeError("embed failed")


async def _embed_items(app, target_fp, embedder, store, items, counter) -> bool:
    """Embed ``(id, text)`` pairs into ``store`` with bounded concurrency.

    Idempotent per item (delete-then-add). Each item is retried; a handful of
    items that still fail are skipped (logged) so a transient blip never kills a
    near-complete run. Only a sustained outage (``_ABORT_AFTER_FAILURES`` total
    hard failures) aborts — leaving the old store untouched. Returns False if the
    run was superseded or aborted.
    """
    sem = asyncio.Semaphore(_REEMBED_CONCURRENCY)
    aborted = {"flag": False}
    failures = {"count": 0}

    async def _one(item) -> None:
        if aborted["flag"] or _current_target_fp != target_fp:
            return
        item_id, text = item
        async with sem:
            if aborted["flag"] or _current_target_fp != target_fp:
                return
            try:
                emb = await _embed_with_retry(embedder, text)
            except Exception as exc:  # noqa: BLE001 — already retried above
                failures["count"] += 1
                logger.warning(
                    "reembed embed failed for %s (%d total): %s", item_id, failures["count"], exc
                )
                if failures["count"] >= _ABORT_AFTER_FAILURES:
                    aborted["flag"] = True
                return
            try:
                # Synchronous Chroma writes — serialized by the event loop, so the
                # concurrent embeds above never overlap a write on one collection.
                store.delete(item_id)
                store.add(note_id=item_id, text=text, embedding=emb)
            except Exception as exc:  # noqa: BLE001
                logger.warning("reembed store write failed for %s: %s", item_id, exc)
                return
            if counter is not None:
                _reembed_state[counter] += 1
                if _reembed_state[counter] % _PROGRESS_EVERY == 0:
                    _persist_status()
                    _emit(app, "reembed.progress", _progress_payload(target_fp))

    await asyncio.gather(*[_one(i) for i in items])

    if aborted["flag"]:
        _fail(app, f"embedding provider unavailable mid-run ({failures['count']} failures)")
        return False
    if _current_target_fp != target_fp:
        return False
    if failures["count"]:
        logger.warning("reembed: %d item(s) skipped after retries", failures["count"])
    if counter is not None:
        _persist_status()
        _emit(app, "reembed.progress", _progress_payload(target_fp))
    return True


async def _reembed_worker(app, target_fp: str) -> None:
    dim = runtime_config.embedding_dimension()
    _reembed_state.update(
        state="running",
        target_fp=target_fp,
        error=None,
        processed_notes=0,
        total_notes=0,
        processed_convs=0,
        total_convs=0,
    )
    _persist_status()
    try:
        if getattr(app.state, "llm_provider", None) is None:
            _fail(app, "llm provider not configured")
            return
        # Use a dedicated, strict, single embedding provider — NOT the router —
        # so a bulk reembed never cascades through the circuit breaker into the
        # local-Ollama fallback (a different embedding space).
        from src.llm.factory import build_embedding_provider

        embedder = build_embedding_provider(strict=True)

        # 1. Validate-before-destroy probe (retried to tolerate a transient blip).
        try:
            probe = await _embed_with_retry(embedder, "reembed probe")
        except Exception as exc:  # noqa: BLE001
            _fail(app, f"embedding probe failed: {exc}")
            return
        if len(probe) != dim:
            _fail(app, f"probe dimension {len(probe)} != target {dim}")
            return
        if probe == _deterministic_embedding("reembed probe", dim):
            _fail(
                app,
                "embedding endpoint returned the deterministic fallback (real model unreachable)",
            )
            return

        emb_path, conv_emb_path = _emb_paths()
        notes_coll = f"notes__{target_fp}"
        conv_coll = f"conversations__{target_fp}"
        new_notes = EmbeddingStore(emb_path, collection_name=notes_coll, dimension=dim)
        new_convs = EmbeddingStore(conv_emb_path, collection_name=conv_coll, dimension=dim)

        started_at = datetime.now(timezone.utc)
        _emit(app, "reembed.started", {"target_fp": target_fp})

        # 2. Notes.
        note_store = app.state.note_store
        notes = note_store.list_all()
        _reembed_state["total_notes"] = len(notes)
        _persist_status()
        if not await _embed_items(
            app,
            target_fp,
            embedder,
            new_notes,
            [(n.id, n.content) for n in notes if n.content],
            counter="processed_notes",
        ):
            return

        # 3. Conversations.
        from src.api.conversations import iter_turns_for_reembed

        turns = iter_turns_for_reembed()
        conv_ids_done = {tid for tid, _ in turns}
        _reembed_state["total_convs"] = len(turns)
        _persist_status()
        if not await _embed_items(
            app,
            target_fp,
            embedder,
            new_convs,
            turns,
            counter="processed_convs",
        ):
            return

        # 4. Catch-up: notes created/edited and conversation turns added during the run.
        recent_notes = [
            (n.id, n.content)
            for n in note_store.list_all()
            if n.content and _as_aware_utc(n.modified) >= started_at
        ]
        if recent_notes and not await _embed_items(
            app,
            target_fp,
            embedder,
            new_notes,
            recent_notes,
            counter=None,
        ):
            return
        new_turns = [
            (tid, text) for tid, text in iter_turns_for_reembed() if tid not in conv_ids_done
        ]
        if new_turns and not await _embed_items(
            app,
            target_fp,
            embedder,
            new_convs,
            new_turns,
            counter=None,
        ):
            return

        if _current_target_fp != target_fp:
            return  # superseded right before swap

        # 5. Atomic swap. Persist pointers FIRST so a crash before delete resumes cleanly.
        old_notes = getattr(app.state, "embedding_store", None)
        old_convs = getattr(app.state, "conversation_embedding_store", None)
        runtime_config.set_app_setting("embedding_notes_collection", notes_coll)
        runtime_config.set_app_setting("embedding_conversations_collection", conv_coll)
        runtime_config.set_app_setting("embedding_active_fingerprint", target_fp)
        app.state.embedding_store = new_notes
        app.state.conversation_embedding_store = new_convs
        # Rebuild hybrid_fusion — DenseRetriever caches the store reference.
        from src.main import build_runtime

        build_runtime(app)
        # Only now drop the old collections (skip a same-fp resume where names match).
        for old, new_name in ((old_notes, notes_coll), (old_convs, conv_coll)):
            if old is not None and getattr(old, "collection_name", None) != new_name:
                old.delete_self()

        _reembed_state.update(state="idle", error=None)
        _persist_status()
        _emit(app, "reembed.complete", {"target_fp": target_fp})
        logger.info("reembed complete: %s", _progress_payload(target_fp))
    except asyncio.CancelledError:
        logger.info("reembed superseded/cancelled for %s", target_fp)
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(app, f"{type(exc).__name__}: {exc}")
        logger.exception("reembed failed")
