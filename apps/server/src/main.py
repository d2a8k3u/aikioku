"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send
import structlog

from src.llm.json_parse import LLMOutputParseError
from src.llm.ollama_remote import EmbeddingUnavailableError
from src.llm.router import BudgetExceededError

from src.config import settings
from src import runtime_config
from src.events import EventBus
from src.knowledge.graph import KnowledgeGraph
from src.limiter import limiter
from src.observability import configure_logging, metrics_middleware, render_metrics

# Configure logging before importing modules that may log at import time.
configure_logging(settings.log_level)
from src.api import (  # noqa: E402
    auth_router,
    chat_router,
    conversations_router,
    notes_router,
    search_router,
    import_export_router,
    review_router,
    graph_router,
    settings_router,
    tokens_router,
    setup_router,
    entities_router,
    stats_router,
    retrieval_router,
    serendipity_router,
    summarization_router,
    question_gen_router,
    connections_router,
    cognitive_state_router,
    git_sync_router,
    anomaly_router,
    auto_tag_router,
    plugin_router,
    buddy_router,
    websocket_router,
    admin_router,
    reembed_router,
    schema_router,
    budget_router,
)
from src.api.websocket import set_broadcaster  # noqa: E402
from src.mcp_server import build_mcp  # noqa: E402

logger = structlog.get_logger()


async def _budget_drain_worker(app: FastAPI, poll_seconds: int = 60) -> None:
    """Background task that drains deferred LLM work once the budget allows.

    Notes/memories ingested while the daily budget was exhausted are queued
    (``pending_llm_work``); this loop re-runs them after the budget resets (UTC
    midnight) or is raised in Settings. A short poll keeps the post-reset
    latency under a minute without busy-waiting.
    """
    import asyncio

    from src.processing import pending_work
    from src.processing.budget_gate import drain

    await asyncio.sleep(8)  # let startup settle
    while True:
        try:
            tracker = getattr(app.state, "cost_tracker", None)
            if tracker is not None and not tracker.is_exhausted() and pending_work.count() > 0:
                await drain(app)
        except Exception:
            logger.exception("budget_drain.failed")
        await asyncio.sleep(poll_seconds)


def build_runtime(app: FastAPI) -> None:
    """(Re)build the secret-dependent runtime: LLM provider + retrievers.

    Reads effective config + decrypted secrets from ``runtime_config``. Called at
    startup when the app is configured, and again after the setup wizard or a
    settings/secret change persists new values — so providers hot-reload with no
    restart. Requires ``embedding_store``/``knowledge_graph`` to already exist on
    ``app.state`` (path-only services initialised in ``lifespan``).
    """
    import asyncio

    old_llm = getattr(app.state, "llm_provider", None)
    old_emb = getattr(app.state, "embedding_provider", None)
    if old_llm is not None:
        try:
            asyncio.ensure_future(old_llm.aclose())
        except Exception:
            logger.warning("build_runtime.old_llm_close_failed", exc_info=True)
    if old_emb is not None and old_emb is not old_llm:
        try:
            asyncio.ensure_future(old_emb.aclose())
        except Exception:
            logger.warning("build_runtime.old_emb_close_failed", exc_info=True)

    from src.llm.factory import build_embedding_provider, build_llm_provider
    from src.llm.router import CostTracker

    cost_tracker = CostTracker(
        settings.sqlite_db_path,
        daily_budget_usd=runtime_config.llm_daily_budget_usd(),
    )
    app.state.cost_tracker = cost_tracker
    app.state.llm_provider = build_llm_provider(cost_tracker)
    # Dedicated embedder, independent of the chat provider. Used for ALL query +
    # document embedding so the vector space matches regardless of chat model.
    app.state.embedding_provider = build_embedding_provider()

    from src.retrieval.dense import DenseRetriever
    from src.retrieval.sparse import SparseRetriever
    from src.retrieval.graph_retrieval import GraphRetriever
    from src.retrieval.fusion import HybridFusion

    dense = DenseRetriever(app.state.embedding_store, app.state.embedding_provider)
    sparse = SparseRetriever(settings.notes_dir)
    graph = GraphRetriever(app.state.knowledge_graph)
    app.state.hybrid_fusion = HybridFusion(dense, sparse, graph)

    from src.reasoning.rag import RAGGenerator
    from src.retrieval.conversation_retrieval import ConversationRetriever

    conversation_retriever = (
        ConversationRetriever(
            app.state.conversation_embedding_store,
            app.state.embedding_provider,
        )
        if getattr(app.state, "conversation_embedding_store", None) is not None
        else None
    )
    app.state.rag_generator = RAGGenerator(
        fusion=app.state.hybrid_fusion,
        llm_provider=app.state.llm_provider,
        conversation_retriever=conversation_retriever,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and cleanup services."""
    # Startup
    app.state.event_bus = EventBus(settings.sqlite_db_path)
    set_broadcaster(app.state.event_bus)

    # Initialize KnowledgeGraph singleton
    import os

    kg_path = settings.sqlite_db_path.replace(".db", "_kg.db")
    os.makedirs(os.path.dirname(kg_path), exist_ok=True)
    app.state.knowledge_graph = KnowledgeGraph(kg_path)

    # Initialize EmbeddingStore singleton. The active collection name is persisted
    # (so a reembed can atomically swap to a new fingerprinted collection); it
    # falls back to the legacy "notes"/"conversations" names on first run.
    from src.knowledge.embeddings import EmbeddingStore

    emb_path = os.path.join(os.path.dirname(settings.sqlite_db_path), "chroma")
    os.makedirs(emb_path, exist_ok=True)
    notes_coll = runtime_config.get_app_setting("embedding_notes_collection") or "notes"
    app.state.embedding_store = EmbeddingStore(
        emb_path, collection_name=notes_coll, dimension=runtime_config.embedding_dimension()
    )

    # Initialize the conversation-turn EmbeddingStore singleton (cross-conversation
    # recall). A SEPARATE Chroma path from notes avoids two PersistentClients on
    # one path; the "conversations" collection keeps turn vectors out of the note
    # fusion path entirely.
    conv_emb_path = os.path.join(os.path.dirname(settings.sqlite_db_path), "chroma_conversations")
    os.makedirs(conv_emb_path, exist_ok=True)
    conv_coll = (
        runtime_config.get_app_setting("embedding_conversations_collection") or "conversations"
    )
    app.state.conversation_embedding_store = EmbeddingStore(
        conv_emb_path, collection_name=conv_coll, dimension=runtime_config.embedding_dimension()
    )

    # Initialize the shared NoteStore singleton and warm its metadata index.
    # reindex() rebuilds the index from the on-disk markdown notes so list/count
    # operations are served from SQLite (no full-corpus scan per request).
    from src.storage.note_store import NoteStore

    from src.scripts.migrate_memory_to_hidden import run_migration

    migration_result = run_migration(settings.notes_dir)
    if migration_result["migrated"] > 0:
        logger.info(
            "memory_migration.complete",
            migrated=migration_result["migrated"],
            skipped=migration_result["skipped"],
            failed=migration_result["failed"],
        )

    note_store = NoteStore(settings.notes_dir)
    try:
        indexed = note_store.reindex()
        logger.info("note_index.warmed", notes=indexed)
    except Exception:
        logger.exception("note_index.warm_failed")
    app.state.note_store = note_store

    # Build the secret-dependent runtime (LLM provider + retrievers) only once
    # the app has been configured via the setup wizard. Until then these stay
    # None and the frontend routes the user to /setup. build_runtime() is also
    # invoked by the setup/settings API to hot-reload without a restart.
    app.state.configured = runtime_config.is_configured()
    app.state.cost_tracker = None
    if app.state.configured:
        build_runtime(app)
    else:
        app.state.llm_provider = None
        app.state.hybrid_fusion = None
        logger.info("aikioku.unconfigured", detail="awaiting setup wizard")

    # Initialize plugin system with EventBus wiring
    from src.plugins.manager import PluginManager
    from src.plugins.api import PluginAPI

    app.state.plugin_manager = PluginManager()
    app.state.plugin_api = PluginAPI(event_bus=app.state.event_bus)

    logger.info("aikioku.started", version="0.1.0")

    import asyncio

    # Drain LLM work deferred while the daily budget was exhausted, once it resets.
    app.state._budget_drain_task = asyncio.create_task(_budget_drain_worker(app))

    from src.processing.consolidation_worker import _consolidation_worker

    app.state._consolidation_task = asyncio.create_task(_consolidation_worker(app))

    # Strong-reference set for fire-and-forget background tasks (Python 3.12+
    # holds only weak references to tasks, so ensure_future/create_task can be
    # silently GC'd). Populated by _spawn_background() in chat.py.
    app.state._bg_tasks: set[asyncio.Task[Any]] = set()  # type: ignore[misc]

    # Embedding fingerprint: adopt the on-disk collections as the baseline on
    # first run (no wipe), then auto-reembed in the background if the effective
    # embedding config has since changed or a prior run was interrupted.
    from src.knowledge.reembed import maybe_schedule_reembed
    from src.knowledge.reembed_fingerprint import effective_embedding_fingerprint

    if runtime_config.get_app_setting("embedding_active_fingerprint") is None:
        runtime_config.set_app_setting(
            "embedding_active_fingerprint", effective_embedding_fingerprint()
        )
        runtime_config.set_app_setting("embedding_notes_collection", notes_coll)
        runtime_config.set_app_setting("embedding_conversations_collection", conv_coll)
    app.state._reembed_task = None
    maybe_schedule_reembed(app)

    # Run the MCP streamable-HTTP session manager for the app's lifetime so the
    # mounted /mcp surface is live. Wraps the yield so it is torn down on shutdown.
    # The manager guards against being run twice per instance; production enters
    # lifespan once, but tests enter it repeatedly via TestClient. Resetting the
    # guard lets the singleton manager re-run — the mounted ASGI app references
    # this same instance, so it always picks up the freshly-created task group.
    mcp.session_manager._has_started = False
    async with mcp.session_manager.run():
        yield
    # Shutdown
    app.state._budget_drain_task.cancel()
    consolidation_task = getattr(app.state, "_consolidation_task", None)
    if consolidation_task is not None:
        consolidation_task.cancel()
    reembed_task = getattr(app.state, "_reembed_task", None)
    if reembed_task is not None:
        reembed_task.cancel()
    # Wait for fire-and-forget background tasks to finish (or 5s timeout)
    if hasattr(app.state, "_bg_tasks") and app.state._bg_tasks:
        await asyncio.wait_for(
            asyncio.gather(*app.state._bg_tasks, return_exceptions=True),
            timeout=5.0,
        )
    logger.info("aikioku.stopped")


app = FastAPI(
    title="Aikioku",
    version="0.1.0",
    description="AI-Augmented Personal Knowledge Management",
    lifespan=lifespan,
)

# MCP server: mount the streamable-HTTP app at /mcp. build_mcp closes over `app`
# so its tools can re-enter the REST API in-process. The session manager is run
# from the lifespan above (references the `mcp` global at startup).
mcp = build_mcp(app)
_mcp_app = mcp.streamable_http_app()


class _MCPClientDisconnectGuard:
    """Buffer the /mcp POST body so a benign client disconnect is handled here,
    not misclassified as a 500 by the MCP SDK.

    ``_handle_post_request`` reads the body off the live receive channel; when a
    client hangs up first — cancellation, idle/LB timeout, network blip, all
    normal — Starlette raises ``ClientDisconnect``, which the SDK logs as a full
    ERROR traceback and answers with a doomed HTTP 500 (upstream python-sdk
    #1648, unfixed as of mcp 1.28). A departed client is not a server error.

    Reading the whole body up front (before the SDK runs) means a mid-body
    disconnect is caught here and answered with silence — the client is gone —
    while a clean body is replayed to the SDK, whose ``body()`` then returns
    instantly and never disconnects. Only POST is intercepted: the GET/SSE stream
    needs its own live receive channel as a disconnect monitor. Buffering suits
    the JSON-RPC-sized payloads that are the only /mcp traffic.
    """

    __name__ = "mcp_client_disconnect_guard"  # SlowAPIMiddleware reads handler.__name__

    def __init__(self, asgi_app: ASGIApp) -> None:
        self._app = asgi_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self._app(scope, receive, send)
            return

        chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                logger.debug("mcp.client_disconnect", path=scope.get("path"))
                # The client is gone, but the BaseHTTPMiddleware layers in front
                # (auth/metrics/slowapi) raise "No response returned." unless one
                # is sent — so answer 499 (Client Closed Request). The bytes drop
                # at the dead socket; the point is a clean, log-free short-circuit.
                await send({"type": "http.response.start", "status": 499, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
            chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        body = b"".join(chunks)
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self._app(scope, replay, send)


_mcp_guarded = _MCPClientDisconnectGuard(_mcp_app)
app.mount("/mcp", _mcp_guarded)


class _MCPBarePath:
    """Serve the bare ``/mcp`` (no trailing slash) without a 307 redirect.

    The streamable app routes at ``/`` (``streamable_http_path="/"``) and is
    mounted at ``/mcp``, so a request to ``/mcp`` reaches the inner app as an
    empty path and Starlette's ``redirect_slashes`` answers 307 → ``/mcp/``.
    Many MCP clients are configured with the no-slash URL, so every call paid a
    redirect round-trip (and doubled the request log). This shim matches ``/mcp``
    exactly and hands the request to the SAME ASGI app with the path rewritten to
    the inner route, yielding a direct 200. ``/mcp/`` keeps working via the mount.

    Implemented as a class (not a function) so Starlette's ``Route`` treats the
    instance as a raw ASGI app rather than a request/response endpoint, passing
    every HTTP method straight through to the streamable app.
    """

    # SlowAPIMiddleware derives a route name via ``handler.__name__``; an ASGI
    # instance has none, so expose one to avoid an AttributeError on every call.
    __name__ = "mcp_bare_path"

    def __init__(self, asgi_app: ASGIApp) -> None:
        self._app = asgi_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Rewrite to the inner streamable route ("/" == streamable_http_path).
        scope = dict(scope, path="/", raw_path=b"/")
        await self._app(scope, receive, send)


# Insert before the Mount (appended above) so the exact-match shim wins over the
# mount's slash redirect for the bare path. Matches "/mcp" only — no sibling routes
# are shadowed. Auth still applies: enforce_auth runs in middleware, before routing.
app.router.routes.insert(0, Route("/mcp", _MCPBarePath(_mcp_guarded)))

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.middleware("http")(metrics_middleware)

# Central auth wall. When auth_required is on (set by the setup wizard), every
# request needs a valid bearer token except the exempt paths below — this gates
# all data routers without adding a dependency to each one. Registered BEFORE the
# CORS middleware so CORS (added last → outermost) still decorates the 401, which
# the browser fetch needs to read the response.
_AUTH_EXEMPT_PREFIXES = (
    "/api/auth",
    "/api/setup",
    "/api/websocket",
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
)


@app.middleware("http")
async def enforce_auth(request: Request, call_next: RequestResponseEndpoint) -> Response:
    path = request.url.path
    if request.method == "OPTIONS" or path == "/" or path.startswith(_AUTH_EXEMPT_PREFIXES):
        return await call_next(request)
    # The /mcp surface is gated by a personal access token (PAT), ALWAYS — it is
    # the externally-exposed machine-client entrypoint, independent of the
    # auth_required flag. This also protects the JSON-RPC initialize/tools-list
    # handshake, not just tool invocations.
    if path.startswith("/mcp"):
        from src import access_tokens

        header = request.headers.get("Authorization", "")
        token = header[7:] if header[:7].lower() == "bearer " else None
        if not token or access_tokens.verify_token(token) is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing access token"},
            )
        return await call_next(request)
    if runtime_config.auth_required():
        from src.auth import token_username

        header = request.headers.get("Authorization", "")
        token = header[7:] if header[:7].lower() == "bearer " else None
        if not token or token_username(token) is None:
            return JSONResponse(status_code=401, content={"detail": "Authentication required"})
    return await call_next(request)


_cors_origins = runtime_config.cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(httpx.HTTPError)
async def upstream_llm_exception_handler(request: Request, exc: httpx.HTTPError) -> JSONResponse:
    """Map upstream LLM/network failures to 503 (covers ConnectError/TimeoutException)."""
    logger.warning("Upstream language model error", exc_info=exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Upstream language model unavailable."},
    )


@app.exception_handler(EmbeddingUnavailableError)
async def embedding_unavailable_exception_handler(
    request: Request, exc: EmbeddingUnavailableError
) -> JSONResponse:
    """Map embedding-model failures to 503."""
    logger.warning("Embedding model unavailable", exc_info=exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Embedding model unavailable."},
    )


@app.exception_handler(LLMOutputParseError)
async def llm_output_parse_exception_handler(
    request: Request, exc: LLMOutputParseError
) -> JSONResponse:
    """Map unparseable LLM output to 502 (bad upstream response)."""
    logger.warning("Invalid language model output", exc_info=exc)
    return JSONResponse(
        status_code=502,
        content={"detail": "The language model returned invalid output."},
    )


@app.exception_handler(BudgetExceededError)
async def budget_exceeded_exception_handler(
    request: Request, exc: BudgetExceededError
) -> JSONResponse:
    """Map daily LLM budget exceeded to 429 (Too Many Requests)."""
    logger.warning("Daily LLM budget exceeded", exc_info=exc)
    # Flip the dashboard budget banner to "paused" — an interactive call (e.g.
    # chat) just hit the cap, so the state changed even if no ingestion did.
    from src.processing.budget_status import broadcast_budget_status

    await broadcast_budget_status(request.app)
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Daily LLM budget exceeded. Processing is paused until the "
            "budget resets at 00:00 UTC, or raise it in Settings."
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Include routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(notes_router)
app.include_router(search_router)
app.include_router(import_export_router)
app.include_router(review_router)
app.include_router(graph_router)
app.include_router(settings_router)
app.include_router(tokens_router)
app.include_router(setup_router)
app.include_router(entities_router)
app.include_router(stats_router)
app.include_router(retrieval_router)
app.include_router(serendipity_router)
app.include_router(summarization_router)
app.include_router(question_gen_router)
app.include_router(connections_router)
app.include_router(cognitive_state_router)
app.include_router(git_sync_router)
app.include_router(anomaly_router)
app.include_router(auto_tag_router)
app.include_router(plugin_router)
app.include_router(buddy_router)
app.include_router(websocket_router)
app.include_router(admin_router)
app.include_router(reembed_router)
app.include_router(schema_router)
app.include_router(budget_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics exposition (request counts/latency, embedding degraded gauge)."""
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Aikioku API", "docs": "/docs"}
