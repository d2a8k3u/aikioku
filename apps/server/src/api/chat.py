"""Chat API endpoint: RAG and multi-hop reasoning with SSE streaming.

Every turn is persisted to the DB (single continuous thread per user) and fed
into the memory pipeline:

- The incoming user message is stored synchronously before generation, and the
  user's recent history is loaded to ground the answer (so in-session recall like
  "what was my previous question?" works).
- A placeholder assistant message with ``in_progress=True`` is created at the start
  of the turn, then promoted to the final answer once generation completes. This
  means a page reload mid-generation shows the work-in-progress reply instead of
  an incomplete chat.
- After the answer is produced, ``_capture_turn`` runs in the background: it
  updates the placeholder with the final content, embeds the turn into the
  ``"conversations"`` vector store (cross-conversation recall), and persists
  extracted memories (factual triples + a deterministic episodic record).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from src.auth import UserInDB, require_auth
from src.cache.semantic_cache import cache_get, cache_put
from src.config import settings
from src.limiter import limiter
from src.storage.note_store import NoteStore
from src.api.websocket import get_broadcaster

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Number of recent messages loaded as short-term conversation context.
_HISTORY_LIMIT = 10

# Hard timeout (seconds) on context building before the stream starts.
# When exceeded, the stream begins with raw retrieval snippets and the
# condensed context is discarded — the user gets an answer, not silence.
_CONTEXT_BUILD_TIMEOUT_S = 10.0


def _persist_memories(memories: list) -> None:
    """Persist chat-extracted memories to SQLite (best-effort).

    A failure here (e.g. a locked DB) must never surface to the client, so all
    errors are swallowed and logged.
    """
    if not memories:
        return
    try:
        from src.api.memory import _store_memories

        _store_memories(memories)
    except Exception:
        logger.warning("Failed to persist chat memories.", exc_info=True)


class ChatRequest(BaseModel):
    query: str
    mode: str = "simple"
    tone: str = "warm"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("simple", "multi_hop"):
            raise ValueError("mode must be 'simple' or 'multi_hop'")
        return v

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        if v not in ("warm", "focused", "playful"):
            raise ValueError("tone must be 'warm', 'focused', or 'playful'")
        return v


class ChatResponse(BaseModel):
    response: str
    citations: list
    mode: str
    sub_questions: list = []


def _get_note_store(request: Request) -> NoteStore:
    """Get NoteStore from app state or create a new one."""
    store = getattr(request.app.state, "note_store", None)
    if store is None:
        store = NoteStore(settings.notes_dir)
        request.app.state.note_store = store
    return store


def _build_rag_generator(request: Request):
    """Return the singleton RAGGenerator from app.state."""
    return request.app.state.rag_generator


def _build_multi_hop_reasoner(request: Request):
    """Build and return a MultiHopReasoner instance using app.state singletons."""
    from src.memory.extraction import MemoryExtractor
    from src.reasoning.multi_hop import MultiHopReasoner

    llm = request.app.state.llm_provider
    graph = request.app.state.knowledge_graph
    memory_extractor = MemoryExtractor(llm, request.app.state.event_bus)
    rag = _build_rag_generator(request)
    return MultiHopReasoner(
        graph=graph,
        llm_provider=llm,
        rag=rag,
        memory_extractor=memory_extractor,
    )


# ------------------------------------------------------------------ persistence


async def _embed_turn(
    request: Request, turn_id: str, created: datetime, user_q: str, assistant_a: str
) -> None:
    """Embed a completed turn into the ``"conversations"`` vector store.

    The stored text is prefixed with the turn date so retrieved turns carry their
    timestamp. Best-effort: never raises.
    """
    store = getattr(request.app.state, "conversation_embedding_store", None)
    embedder = getattr(request.app.state, "embedding_provider", None)
    if store is None or embedder is None:
        return
    text = f"[{created.date().isoformat()}] user: {user_q}\nassistant: {assistant_a}"
    try:
        embedding = await embedder.embed(text)
        store.add(note_id=turn_id, text=text, embedding=embedding)
    except Exception:
        logger.warning("Failed to embed conversation turn.", exc_info=True)


async def _extract_factual(request: Request, user_q: str, assistant_a: str) -> list:
    """Extract factual memory triples from a turn (best-effort)."""
    llm = getattr(request.app.state, "llm_provider", None)
    if llm is None:
        return []
    try:
        from src.memory.extraction import MemoryExtractor

        extractor = MemoryExtractor(llm, getattr(request.app.state, "event_bus", None))
        return await extractor.extract_from_conversation(
            [
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": assistant_a},
            ]
        )
    except Exception:
        logger.warning("Factual memory extraction failed.", exc_info=True)
        return []


def _extract_graph_entities(request: Request, turn_id: str, user_q: str, assistant_a: str) -> None:
    """Extract entities + relations from a completed chat turn into the shared KG.

    The knowledge graph is the project's source of truth, so chat turns enrich it
    the same way notes do (see notes.py ``_extract_and_store_entities``). Runs
    fire-and-forget; failures never affect the chat. Uses the SHARED Kuzu handle
    (``app.state.knowledge_graph``) — a second handle would conflict (Kuzu is
    single-writer). The assistant id (``turn_id``) is recorded as provenance.
    """

    async def _do() -> None:
        try:
            from src.knowledge.pipeline import extract_entities_from_text

            llm = getattr(request.app.state, "llm_provider", None)
            graph = getattr(request.app.state, "knowledge_graph", None)
            if llm is None or graph is None:
                return
            text = f"{user_q}\n\n{assistant_a}".strip()
            entities = await extract_entities_from_text(
                text=text,
                source_id=turn_id,
                llm_provider=llm,
                graph=graph,
                source_is_note=False,
            )
            logger.info("Chat entity extraction: turn %s -> %d entities", turn_id, len(entities))
        except Exception:
            logger.warning("Chat entity extraction failed for turn %s.", turn_id, exc_info=True)

    asyncio.ensure_future(_do())


async def _capture_turn(
    request: Request,
    user_id: str,
    user_q: str,
    assistant_a: str,
    turn_id: str,
    citations: list,
    sub_questions: list,
    memories: list | None = None,
) -> None:
    """Persist + index a completed turn. Runs in the background; never fails the chat.

    Steps: promote the pre-created ``in_progress`` assistant placeholder to its
    final content, embed the turn for cross-conversation recall, and persist
    memories.

    Args:
        memories: Factual memories already extracted during generation (non-stream
            and multi-hop paths return them). When None (the simple streaming
            path, which generates no memories), factual memories are extracted
            here. A deterministic episodic memory is always added.
    """
    from src.api.conversations import update_message
    from src.memory.extraction import build_episodic_memory

    # Outer guard: this often runs fire-and-forget (streaming path), so an
    # unhandled error here would surface only as an orphaned-task warning.
    try:
        created = datetime.utcnow()

        # Promote the placeholder assistant message to final content.
        try:
            update_message(
                turn_id,
                content=assistant_a,
                citations=citations,
                sub_questions=sub_questions,
                in_progress=False,
            )
            # Broadcast so WebSocket-connected clients (including the chat page
            # after a mid-generation reload) can update the placeholder in-place
            # without polling.
            broadcaster = get_broadcaster()
            if broadcaster:
                await broadcaster.broadcast("chat.message_updated", {
                    "message_id": turn_id,
                    "content": assistant_a,
                    "citations": citations,
                    "sub_questions": sub_questions,
                    "in_progress": False,
                })
        except Exception:
            logger.warning("Failed to update assistant placeholder %s.", turn_id, exc_info=True)

        await _embed_turn(request, turn_id, created, user_q, assistant_a)
        _extract_graph_entities(request, turn_id, user_q, assistant_a)

        captured = list(memories) if memories else []
        if memories is None:
            captured.extend(await _extract_factual(request, user_q, assistant_a))
        try:
            captured.append(build_episodic_memory(user_q, created=created))
        except Exception:
            logger.warning("Failed to build episodic memory.", exc_info=True)
        _persist_memories(captured)
    except Exception:
        logger.warning("Failed to capture conversation turn.", exc_info=True)


def _store_user_message(user_id: str, query: str) -> None:
    """Persist the incoming user message synchronously (best-effort)."""
    from src.api.conversations import store_message
    from src.models.conversation import ConversationMessage

    try:
        store_message(ConversationMessage(user_id=user_id, role="user", content=query))
    except Exception:
        logger.warning("Failed to persist user message.", exc_info=True)


def _begin_turn(user_id: str, query: str) -> str:
    """Persist the user message and a placeholder assistant message for a new turn.

    Returns the placeholder assistant message id. If persistence fails, a fresh
    id is still returned so the rest of the pipeline can continue without an
    assistant message ever appearing in the history.
    """
    from src.api.conversations import store_message
    from src.models.conversation import ConversationMessage

    turn_id = str(uuid.uuid4())
    _store_user_message(user_id, query)
    try:
        store_message(
            ConversationMessage(
                id=turn_id,
                user_id=user_id,
                role="assistant",
                content="",
                in_progress=True,
            )
        )
    except Exception:
        logger.warning("Failed to create assistant placeholder %s.", turn_id, exc_info=True)
    return turn_id


def _load_history(user_id: str) -> list[dict]:
    """Load the user's recent history (chronological) for short-term grounding."""
    try:
        from src.api.conversations import recent_history

        return recent_history(user_id, limit=_HISTORY_LIMIT)
    except Exception:
        logger.warning("Failed to load conversation history.", exc_info=True)
        return []


# ------------------------------------------------------------------ endpoints


@router.post("/", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    background_tasks: BackgroundTasks,
    user: UserInDB = Depends(require_auth),
) -> dict:
    """Process a chat query using RAG or multi-hop reasoning.

    The turn is persisted and captured into memory in a background task (after
    the response is returned), so it never blocks the request and a persistence
    failure never fails the chat.
    """
    user_id = user.username
    note_store = _get_note_store(request)

    history = _load_history(user_id)
    turn_id = _begin_turn(user_id, body.query)

    if body.mode == "simple":
        rag = _build_rag_generator(request)
        result = await rag.generate(body.query, note_store, history=history)
    else:
        reasoner = _build_multi_hop_reasoner(request)
        result = await reasoner.reason(body.query, note_store, history=history)

    citations = result.get("citations", [])
    sub_questions = result.get("sub_questions", [])
    background_tasks.add_task(
        _capture_turn,
        request,
        user_id,
        body.query,
        result["response"],
        turn_id,
        citations,
        sub_questions,
        result.get("memories", []),
    )

    return {
        "response": result["response"],
        "citations": citations,
        "mode": body.mode,
        "sub_questions": sub_questions,
    }


@router.get("/stream")
@limiter.limit("10/minute")
async def chat_stream_get(
    request: Request,
    query: str,
    mode: str = "simple",
    tone: str = "warm",
    user: UserInDB = Depends(require_auth),
):
    """Stream a chat response via Server-Sent Events (GET)."""
    if mode not in ("simple", "multi_hop"):
        mode = "simple"
    if tone not in ("warm", "focused", "playful"):
        tone = "warm"
    return await _stream_chat(request, query, mode, user.username, tone)


@router.post("/stream")
@limiter.limit("10/minute")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    user: UserInDB = Depends(require_auth),
):
    """Stream a chat response via Server-Sent Events (POST)."""
    return await _stream_chat(request, body.query, body.mode, user.username, body.tone)


# Bound the number of characters per streamed chunk for already-materialized text
# (multi-hop synthesized answer) so the client receives progressive message events.
_STREAM_CHUNK_SIZE = 256


def _sse(event: str, data: dict | list) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Progressive context builder (used by simple-mode streaming) ──────────

async def _build_context_progressive(
    rag,
    query: str,
    note_store: NoteStore,
    history: list[dict] | None,
    pre_fetched: list | None = None,
) -> tuple[str, list[dict]]:
    """Build grounding context with a hard timeout.

    When ``pre_fetched`` is provided (fusion results from the citations step),
    the full build_context path reuses them instead of re-running fusion.

    When the full condensed context isn't ready within ``_CONTEXT_BUILD_TIMEOUT_S``,
    falls back to raw retrieval snippets so the stream can start immediately.
    The user gets an answer, not silence.

    Returns (system_prompt, citations).
    """
    try:
        return await asyncio.wait_for(
            rag.build_context(query, note_store, history=history),
            timeout=_CONTEXT_BUILD_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Context build timed out after %ds; falling back to raw snippets",
            _CONTEXT_BUILD_TIMEOUT_S,
        )
        # Fast path: retrieve only, skip condensation. Build a minimal
        # system prompt from raw snippets. Reuse pre_fetched if available.
        results = pre_fetched if pre_fetched is not None else await rag._fusion.search(query)
        context_entries: list[dict] = []
        for r in results[:5]:
            note = note_store.get(r.note_id)
            title = note.title if note else None
            raw = r.snippet or (note.content if note else "")
            content = raw
            context_entries.append({
                "note_id": r.note_id,
                "snippet": r.snippet or "",
                "content": content,
                "score": r.score,
                "title": title,
            })
        system_prompt = rag._build_system_prompt(context_entries, None, history)
        citations = rag._extract_citations("", context_entries)
        return system_prompt, citations


async def _stream_chat(
    request: Request, query: str, mode: str, user_id: str, tone: str = "warm"
) -> StreamingResponse:
    """Stream a chat response with progressive context building.

    A placeholder assistant message is created at the start of the turn, so a
    page reload mid-generation shows a work-in-progress reply instead of an
    incomplete chat. The placeholder is promoted to the final answer once the
    stream completes, and removed if the user explicitly stops generation.

    Simple mode: citations are sent IMMEDIATELY after retrieval. Context building
    runs with a hard timeout — if it exceeds 10s, the stream starts with raw
    snippets so the user sees an answer, not silence.

    Multi-hop mode: run the (now fast, concurrent) reasoner and stream its
    already-synthesized answer to the client without re-invoking the LLM.
    """

    note_store = _get_note_store(request)
    llm = request.app.state.llm_provider

    history = _load_history(user_id)
    turn_id = _begin_turn(user_id, query)

    # Broadcast buddy state: thinking
    broadcaster = get_broadcaster()
    if broadcaster:
        await broadcaster.broadcast("buddy.state", {"state": "thinking"})
        await broadcaster.broadcast("chat.streaming", {"active": True})

    # ── Semantic cache check ──────────────────────────────────────────────
    cached = await cache_get(query, mode, tone)
    if cached is not None:
        # Cache hit: stream the cached answer immediately in SSE format.
        # The turn placeholder is already created; we persist the cached
        # answer so history stays consistent.

        async def _cache_hit_stream():
            _completed = False
            try:
                answer = cached["response"]
                citations_data = cached.get("citations", [])
                sub_qs = cached.get("sub_questions", [])

                yield _sse(
                    "citations",
                    {"citations": citations_data, "sub_questions": sub_qs},
                )
                if answer:
                    for i in range(0, len(answer), _STREAM_CHUNK_SIZE):
                        yield _sse(
                            "message",
                            {"chunk": answer[i : i + _STREAM_CHUNK_SIZE]},
                        )
                else:
                    yield _sse("message", {"chunk": ""})

                asyncio.ensure_future(
                    _capture_turn(
                        request,
                        user_id,
                        query,
                        answer,
                        turn_id,
                        citations_data,
                        sub_qs,
                        None,
                    )
                )
                if broadcaster:
                    await broadcaster.broadcast("buddy.state", {"state": "listening"})
                    await broadcaster.broadcast("chat.streaming", {"active": False})
                yield "event: done\ndata: {}\n\n"
                _completed = True
            finally:
                if not _completed:
                    _delete_turn_placeholder(turn_id)
                    if broadcaster:
                        await broadcaster.broadcast("buddy.state", {"state": "listening"})
                        await broadcaster.broadcast("chat.streaming", {"active": False})

        return StreamingResponse(
            _cache_hit_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ── Cache miss: continue normal flow ─────────────────────────────────

    # Tone instruction prepended to system prompt
    _TONE_INSTRUCTIONS = {
        "warm": "Respond kindly and encouragingly. Use a friendly tone.",
        "focused": "Respond concisely and factually. Stick to the facts.",
        "playful": "Respond lightly and curiously. You can be playful.",
    }
    tone_prefix = _TONE_INSTRUCTIONS.get(tone, "")

    if mode == "simple":
        rag = _build_rag_generator(request)

        async def _event_stream():
            # Track whether the turn completed normally so the finally block
            # knows whether to clean up the placeholder.
            _completed = False
            try:
                # Step 1: Retrieve immediately, send citations right away.
                # The user sees note chips before the answer starts — instant feedback.
                results = await rag._fusion.search(query)
                raw_citations = rag._extract_citations("", [
                    {
                        "note_id": r.note_id,
                        "snippet": r.snippet,
                        "content": r.snippet,
                        "score": r.score,
                        "title": None,
                    }
                    for r in results[:5]
                ])
                yield _sse("citations", raw_citations)

                # Step 1.5: Stream raw note snippets immediately for instant preview.
                # The user sees content previews while LLMLingua compression runs.
                for r in results[:5]:
                    note = note_store.get(r.note_id)
                    if note and note.content:
                        snippet_text = note.content[:500]
                        yield _sse("snippet", {
                            "note_id": r.note_id,
                            "title": note.title,
                            "snippet": snippet_text,
                        })

                # Step 2: Build context with timeout, reusing pre-fetched results.
                # If slow, fall back to snippets.
                system_prompt, citations = await _build_context_progressive(
                    rag, query, note_store, history, pre_fetched=results
                )
                if tone_prefix:
                    system_prompt = tone_prefix + "\n\n" + system_prompt

                # Step 3: Stream the LLM answer with a total timeout guard.
                # The httpx client already has a 120s per-request timeout, but
                # an additional asyncio-level timeout ensures the generator
                # cannot hang indefinitely if the HTTP layer silently stalls.
                _STREAM_TOTAL_TIMEOUT_S = 120.0
                answer_parts: list[str] = []

                async def _stream_with_timeout():
                    """Iterate llm.stream() with a total timeout on the whole loop."""
                    stream = llm.stream(prompt=query, system=system_prompt)
                    deadline = asyncio.get_event_loop().time() + _STREAM_TOTAL_TIMEOUT_S
                    while True:
                        remaining = deadline - asyncio.get_event_loop().time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError("Stream total timeout exceeded")
                        try:
                            chunk = await asyncio.wait_for(stream.__anext__(), timeout=min(remaining, 30.0))
                            answer_parts.append(chunk)
                            yield _sse("message", {"chunk": chunk})
                        except StopAsyncIteration:
                            return
                        except asyncio.TimeoutError:
                            # Per-chunk timeout — the stream stalled. If we have
                            # partial content, stop gracefully; otherwise raise.
                            if answer_parts:
                                return
                            raise

                try:
                    async for event in _stream_with_timeout():
                        yield event
                except asyncio.TimeoutError:
                    if answer_parts:
                        pass  # partial answer is fine
                    else:
                        yield _sse("message", {"chunk": "[I'm taking too long. Please try a shorter question.]"})
                except asyncio.CancelledError:
                    # Client disconnected or stop requested — handled by finally.
                    raise
                except Exception:
                    # Degrade gracefully: emit the system prompt as a single chunk so
                    # the client receives something rather than an empty stream.
                    yield _sse("message", {"chunk": system_prompt})

                answer = "".join(answer_parts)
                # Simple streaming generates no memories; _capture_turn extracts them.
                asyncio.ensure_future(
                    _capture_turn(request, user_id, query, answer, turn_id, citations, [], None)
                )
                # Store in semantic cache for future hits (fire-and-forget).
                asyncio.ensure_future(
                    cache_put(query, mode, tone, answer, citations, [])
                )
                # Broadcast buddy state: listening
                if broadcaster:
                    await broadcaster.broadcast("buddy.state", {"state": "listening"})
                    await broadcaster.broadcast("chat.streaming", {"active": False})
                yield "event: done\ndata: {}\n\n"
                _completed = True
            finally:
                if not _completed:
                    # Turn did not complete — client disconnected, stop requested,
                    # or an unhandled error. Remove the placeholder so the chat
                    # doesn't show a stuck WIP bubble on reload.
                    _delete_turn_placeholder(turn_id)
                    if broadcaster:
                        await broadcaster.broadcast("buddy.state", {"state": "listening"})
                        await broadcaster.broadcast("chat.streaming", {"active": False})

    else:
        reasoner = _build_multi_hop_reasoner(request)
        result = await reasoner.reason(query, note_store, history=history)
        answer = result.get("response", "")
        citations = result.get("citations", [])
        sub_questions = result.get("sub_questions", [])
        memories = result.get("memories", [])

        async def _event_stream():
            _completed = False
            try:
                yield _sse(
                    "citations",
                    {"citations": citations, "sub_questions": sub_questions},
                )
                try:
                    if answer:
                        for i in range(0, len(answer), _STREAM_CHUNK_SIZE):
                            yield _sse(
                                "message",
                                {"chunk": answer[i : i + _STREAM_CHUNK_SIZE]},
                            )
                    else:
                        yield _sse("message", {"chunk": ""})
                except asyncio.CancelledError:
                    raise  # handled by finally
                except Exception:
                    yield _sse("message", {"chunk": answer})
                asyncio.ensure_future(
                    _capture_turn(
                        request,
                        user_id,
                        query,
                        answer,
                        turn_id,
                        citations,
                        sub_questions,
                        memories,
                    )
                )
                # Store in semantic cache for future hits (fire-and-forget).
                asyncio.ensure_future(
                    cache_put(query, mode, tone, answer, citations, sub_questions)
                )
                # Broadcast buddy state: listening
                if broadcaster:
                    await broadcaster.broadcast("buddy.state", {"state": "listening"})
                    await broadcaster.broadcast("chat.streaming", {"active": False})
                yield "event: done\ndata: {}\n\n"
                _completed = True
            finally:
                if not _completed:
                    _delete_turn_placeholder(turn_id)
                    if broadcaster:
                        await broadcaster.broadcast("buddy.state", {"state": "listening"})
                        await broadcaster.broadcast("chat.streaming", {"active": False})

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _delete_turn_placeholder(turn_id: str) -> None:
    """Remove an in-progress assistant placeholder when generation is cancelled."""
    try:
        from src.api.conversations import delete_message

        delete_message(turn_id)
    except Exception:
        logger.warning("Failed to delete assistant placeholder %s.", turn_id, exc_info=True)


# ------------------------------------------------------------------ stop generation

# Active generation tasks keyed by user_id for cancellation.
_active_generations: dict[str, "asyncio.Task"] = {}


@router.post("/stop")
async def stop_generation(
    request: Request,
    user: UserInDB = Depends(require_auth),
) -> dict:
    """Cancel the authenticated user's active chat generation."""

    task = _active_generations.pop(user.username, None)
    if task is not None and not task.done():
        task.cancel()
        broadcaster = get_broadcaster()
        if broadcaster:
            await broadcaster.broadcast("buddy.state", {"state": "idle"})
            await broadcaster.broadcast("chat.streaming", {"active": False})
        return {"status": "cancelled"}
    return {"status": "no_active_generation"}
