"""Chat API endpoint: RAG and multi-hop reasoning with SSE streaming.

Every turn is persisted to the DB (single continuous thread per user) and fed
into the indexing pipeline:

- The incoming user message is stored synchronously before generation, and the
  user's recent history is loaded to ground the answer (so in-session recall like
  "what was my previous question?" works).
- A placeholder assistant message with ``in_progress=True`` is created at the start
  of the turn, then promoted to the final answer once generation completes. This
  means a page reload mid-generation shows the work-in-progress reply instead of
  an incomplete chat.
- After the answer is produced, ``_capture_turn`` runs in the background: it
  updates the placeholder with the final content, embeds the turn into the
  ``"conversations"`` vector store (cross-conversation recall), and extracts
  knowledge-graph entities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from src import runtime_config
from src.api.websocket import get_broadcaster
from src.auth import CurrentUser
from src.cache.semantic_cache import cache_get, cache_put
from src.config import settings
from src.limiter import limiter
from src.storage.note_store import NoteStore
from src.utils.background import spawn_background as _spawn_background
from src.utils.clock import utc_now_naive

if TYPE_CHECKING:
    from src.reasoning.multi_hop import MultiHopReasoner
    from src.reasoning.rag import RAGGenerator
    from src.retrieval.search_result import SearchResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Number of recent messages loaded as short-term conversation context.
_HISTORY_LIMIT = 10

# Hard timeout (seconds) on context building before the stream starts.
# When exceeded, the stream begins with raw retrieval snippets and the
# condensed context is discarded — the user gets an answer, not silence.
_CONTEXT_BUILD_TIMEOUT_S = 10.0

# Timeout for the fact extraction LLM call. The prompt is simpler than entity
# extraction (which uses 60s) — a single classification + extraction over a
# small turn — so 30s is generous while preventing indefinite hangs.
LLM_FACT_EXTRACTION_TIMEOUT = 30

# Combined text above this many characters is condensed before the LLM call
# to stay within model context windows without truncating content.
_FACT_CHAR_BUDGET = 8000

_FACT_SYSTEM_PROMPT = (
    "You are a fact extractor for a personal knowledge system. Your job is to identify\n"
    "factual statements made by the USER in a chat conversation that are worth remembering\n"
    "long-term.\n\n"
    "Rules:\n"
    "1. Only extract facts stated by the USER — not information provided by the assistant\n"
    "   (the assistant's answer is already retrievable via the knowledge base).\n"
    "2. Only extract clear, factual statements. Skip:\n"
    '   - Opinions ("I think X is great")\n'
    '   - Questions ("What is Y?")\n'
    '   - Transient statements ("I\'m feeling tired today")\n'
    '   - Hypotheticals ("If I were to...")\n'
    '   - Commands/requests ("Tell me about Z")\n'
    "3. A factual statement is something that could be true or false and is worth\n"
    "   remembering about the user or their context. Examples:\n"
    '   - "My wife\'s name is May" → fact\n'
    '   - "I work at Acme Corp" → fact\n'
    '   - "I prefer Python over JavaScript" → fact\n'
    '   - "My birthday is March 15th" → fact\n'
    "4. Preserve the user's original language in the extracted note content.\n"
    "5. For each fact, provide a concise, descriptive title (max 80 chars) suitable\n"
    "   for search indexing.\n\n"
    "Respond with JSON only, no markdown fences:\n"
    '{"contains_facts": true/false, "facts": [{"title": "...", "content": "..."}]}\n\n'
    'If no facts are found, return: {"contains_facts": false, "facts": []}'
)


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
    citations: list[dict[str, Any]]
    mode: str
    sub_questions: list[str] = []


def _get_note_store(request: Request) -> NoteStore:
    """Get NoteStore from app state or create a new one."""
    store = getattr(request.app.state, "note_store", None)
    if store is None:
        store = NoteStore(settings.notes_dir)
        request.app.state.note_store = store
    return store


def _build_rag_generator(request: Request) -> RAGGenerator:
    """Return the singleton RAGGenerator from app.state."""
    return cast("RAGGenerator", request.app.state.rag_generator)


def _build_multi_hop_reasoner(request: Request) -> MultiHopReasoner:
    """Build and return a MultiHopReasoner instance using app.state singletons."""
    from src.reasoning.multi_hop import MultiHopReasoner

    llm = request.app.state.llm_provider
    graph = request.app.state.knowledge_graph
    rag = _build_rag_generator(request)
    return MultiHopReasoner(
        graph=graph,
        llm_provider=llm,
        rag=rag,
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

    _spawn_background(request.app, _do())


def _extract_facts(request: Request, turn_id: str, user_q: str, assistant_a: str) -> None:
    """Extract factual statements from a chat turn into hidden notes.

    Uses one LLM call to classify the turn and extract any user-stated facts.
    Each fact becomes a hidden note (source_type="hidden") so it is retrievable
    via the existing fusion path (dense + sparse + graph) without appearing in
    the user-facing notes API. Fire-and-forget: failures never affect the chat.
    """

    async def _do() -> None:
        try:
            from src.api.notes import _extract_and_store_entities, _invalidate_sparse_index
            from src.llm.json_parse import parse_llm_json
            from src.llm.long_text import condense_for_prompt
            from src.models.note import Note

            llm = getattr(request.app.state, "llm_provider", None)
            store = getattr(request.app.state, "note_store", None)
            if llm is None or store is None:
                return

            text = f"{user_q}\n\n{assistant_a}".strip()
            if len(text) > _FACT_CHAR_BUDGET:
                text = await condense_for_prompt(llm, text)

            prompt = (
                f"Conversation turn:\n\nUser: {user_q}\n\nAssistant: {assistant_a}\n\n"
                "Extract factual statements worth remembering. Return JSON with keys "
                '"contains_facts" and "facts".'
            )
            response = await asyncio.wait_for(
                llm.complete(prompt=prompt, system=_FACT_SYSTEM_PROMPT),
                timeout=LLM_FACT_EXTRACTION_TIMEOUT,
            )

            result = parse_llm_json(response, expect="dict")
            if not isinstance(result, dict):
                return
            if not result.get("contains_facts"):
                return
            facts = result.get("facts", [])
            if not isinstance(facts, list):
                return

            for fact in facts:
                if not isinstance(fact, dict):
                    continue
                title = str(fact.get("title", "")).strip()
                content = str(fact.get("content", "")).strip()
                if not title or not content:
                    continue
                note = Note(
                    title=title,
                    content=content,
                    path=f"/notes/hidden/{uuid.uuid4()}.md",
                    source_type="hidden",
                    frontmatter={
                        "extracted_from_turn": turn_id,
                        "tags": ["auto-extracted"],
                    },
                )
                created = store.create(note)
                _extract_and_store_entities(request, created)
                _invalidate_sparse_index(request)

            logger.info("Chat fact extraction: turn %s -> %d facts", turn_id, len(facts))
        except Exception:
            logger.warning("Chat fact extraction failed for turn %s.", turn_id, exc_info=True)

    _spawn_background(request.app, _do())


async def _capture_turn(
    request: Request,
    user_id: str,
    user_q: str,
    assistant_a: str,
    turn_id: str,
    citations: list[dict[str, Any]],
    sub_questions: list[str],
) -> None:
    """Persist + index a completed turn. Runs in the background; never fails the chat.

    Steps: promote the pre-created ``in_progress`` assistant placeholder to its
    final content, embed the turn for cross-conversation recall, and extract
    knowledge-graph entities.
    """
    from src.api.conversations import update_message

    # Outer guard: this often runs fire-and-forget (streaming path), so an
    # unhandled error here would surface only as an orphaned-task warning.
    try:
        created = utc_now_naive()

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
                await broadcaster.broadcast(
                    "chat.message_updated",
                    {
                        "message_id": turn_id,
                        "content": assistant_a,
                        "citations": citations,
                        "sub_questions": sub_questions,
                        "in_progress": False,
                    },
                )
        except Exception:
            logger.warning("Failed to update assistant placeholder %s.", turn_id, exc_info=True)

        await _embed_turn(request, turn_id, created, user_q, assistant_a)
        if runtime_config.auto_entity_extraction():
            _extract_graph_entities(request, turn_id, user_q, assistant_a)
        if runtime_config.auto_fact_extraction():
            _extract_facts(request, turn_id, user_q, assistant_a)
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


def _load_history(user_id: str) -> list[dict[str, Any]]:
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
    user: CurrentUser,
) -> dict[str, Any]:
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
    user: CurrentUser,
    mode: str = "simple",
    tone: str = "warm",
) -> StreamingResponse:
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
    user: CurrentUser,
) -> StreamingResponse:
    """Stream a chat response via Server-Sent Events (POST)."""
    return await _stream_chat(request, body.query, body.mode, user.username, body.tone)


# Bound the number of characters per streamed chunk for already-materialized text
# (multi-hop synthesized answer) so the client receives progressive message events.
_STREAM_CHUNK_SIZE = 256


def _sse(event: str, data: dict[str, Any] | list[Any]) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Progressive context builder (used by simple-mode streaming) ──────────


async def _build_context_progressive(
    rag: RAGGenerator,
    query: str,
    note_store: NoteStore,
    history: list[dict[str, Any]] | None,
    pre_fetched: list[SearchResult] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
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
    except TimeoutError:
        logger.warning(
            "Context build timed out after %ds; falling back to raw snippets",
            _CONTEXT_BUILD_TIMEOUT_S,
        )
        # Fast path: retrieve only, skip condensation. Build a minimal
        # system prompt from raw snippets. Reuse pre_fetched if available.
        results = pre_fetched if pre_fetched is not None else await rag._fusion.search(query)
        context_entries: list[dict[str, Any]] = []
        for r in results[:5]:
            note = note_store.get(r.note_id)
            title = note.title if note else None
            raw = r.snippet or (note.content if note else "")
            content = raw
            context_entries.append(
                {
                    "note_id": r.note_id,
                    "snippet": r.snippet or "",
                    "content": content,
                    "score": r.score,
                    "title": title,
                }
            )
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
    # Cache entries are model-specific: a different provider/model produces a
    # different answer, so the active model is validated on hit (see cache_get).
    model_id = f"{runtime_config.llm_provider()}:{runtime_config.llm_model()}"

    history = _load_history(user_id)
    turn_id = _begin_turn(user_id, query)

    # Broadcast buddy state: thinking
    broadcaster = get_broadcaster()
    if broadcaster:
        await broadcaster.broadcast("buddy.state", {"state": "thinking"})
        await broadcaster.broadcast("chat.streaming", {"active": True})

    # ── Semantic cache check ──────────────────────────────────────────────
    cached = await cache_get(query, mode, tone, model_id)
    if cached is not None:
        # Cache hit: stream the cached answer immediately in SSE format.
        # The turn placeholder is already created; we persist the cached
        # answer so history stays consistent.

        async def _cache_hit_stream() -> AsyncIterator[str]:
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

                _spawn_background(
                    request.app,
                    _capture_turn(
                        request,
                        user_id,
                        query,
                        answer,
                        turn_id,
                        citations_data,
                        sub_qs,
                    ),
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

        async def _event_stream() -> AsyncIterator[str]:
            # Track whether the turn completed normally so the finally block
            # knows whether to clean up the placeholder.
            _completed = False
            try:
                # Step 1: Retrieve immediately, send citations right away.
                # The user sees note chips before the answer starts — instant feedback.
                results = await rag._fusion.search(query)
                raw_citations = rag._extract_citations(
                    "",
                    [
                        {
                            "note_id": r.note_id,
                            "snippet": r.snippet,
                            "content": r.snippet,
                            "score": r.score,
                            "title": None,
                        }
                        for r in results[:5]
                    ],
                )
                yield _sse("citations", raw_citations)

                # Step 1.5: Stream raw note snippets immediately for instant preview.
                # The user sees content previews while LLMLingua compression runs.
                for r in results[:5]:
                    note = note_store.get(r.note_id)
                    if note and note.content:
                        snippet_text = note.content[:500]
                        yield _sse(
                            "snippet",
                            {
                                "note_id": r.note_id,
                                "title": note.title,
                                "snippet": snippet_text,
                            },
                        )

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

                async def _stream_with_timeout() -> AsyncIterator[str]:
                    """Iterate llm.stream() with a total timeout on the whole loop."""
                    stream = llm.stream(prompt=query, system=system_prompt)
                    deadline = asyncio.get_event_loop().time() + _STREAM_TOTAL_TIMEOUT_S
                    while True:
                        remaining = deadline - asyncio.get_event_loop().time()
                        if remaining <= 0:
                            raise TimeoutError("Stream total timeout exceeded")
                        try:
                            chunk = await asyncio.wait_for(
                                stream.__anext__(), timeout=min(remaining, 30.0)
                            )
                            answer_parts.append(chunk)
                            yield _sse("message", {"chunk": chunk})
                        except StopAsyncIteration:
                            return
                        except TimeoutError:
                            # Per-chunk timeout — the stream stalled. If we have
                            # partial content, stop gracefully; otherwise raise.
                            if answer_parts:
                                return
                            raise

                try:
                    async for event in _stream_with_timeout():
                        yield event
                except TimeoutError:
                    if answer_parts:
                        pass  # partial answer is fine
                    else:
                        yield _sse(
                            "message",
                            {"chunk": "[I'm taking too long. Please try a shorter question.]"},
                        )
                except asyncio.CancelledError:
                    # Client disconnected or stop requested — handled by finally.
                    raise
                except Exception:
                    # Degrade gracefully: emit the system prompt as a single chunk so
                    # the client receives something rather than an empty stream.
                    logger.exception("chat stream: greeting generation failed")
                    yield _sse("message", {"chunk": system_prompt})

                answer = "".join(answer_parts)
                _spawn_background(
                    request.app,
                    _capture_turn(request, user_id, query, answer, turn_id, citations, []),
                )
                # Store in semantic cache for future hits (fire-and-forget).
                _spawn_background(
                    request.app, cache_put(query, mode, tone, answer, citations, [], model_id)
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

        async def _event_stream() -> AsyncIterator[str]:
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
                    logger.exception("chat stream: chunked answer delivery failed")
                    yield _sse("message", {"chunk": answer})
                _spawn_background(
                    request.app,
                    _capture_turn(
                        request,
                        user_id,
                        query,
                        answer,
                        turn_id,
                        citations,
                        sub_questions,
                    ),
                )
                # Store in semantic cache for future hits (fire-and-forget).
                _spawn_background(
                    request.app,
                    cache_put(query, mode, tone, answer, citations, sub_questions, model_id),
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
