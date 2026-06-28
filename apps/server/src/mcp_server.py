"""MCP server exposing the Aikioku over streamable-HTTP.

Mounted into the main FastAPI app at ``/mcp`` (see ``main.py``). Tools are thin
wrappers that re-enter the SAME app over an in-process ``httpx`` ASGI transport,
reusing every existing REST endpoint (background tasks, fusion, extraction) with
no duplicated logic. ``app.state`` is shared because it is the same ``app`` object;
the ASGI transport sends only HTTP scope, never lifespan, so it never re-inits state.

Auth: the ``/mcp`` surface is gated by a personal access token (PAT) at the
``enforce_auth`` middleware boundary. Each tool ALSO reads the PAT from the request
headers (``ctx.request_context.request.headers``) to resolve its scope — read-only
tokens may not invoke write tools. Internal REST calls carry a short-lived JWT
minted for the token's owner.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import httpx
from fastapi import FastAPI
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from src import access_tokens
from src.auth import create_access_token

_Ctx = Context[ServerSession, Any]

_INTERNAL_BASE = "http://mcp.internal"

# Paths the generic escape-hatch tool must never reach (privilege escalation /
# secret exfiltration). Token + auth + setup management stay JWT-only.
_CALL_API_DENY = (
    "/api/auth",
    "/api/setup",
    "/api/settings/secrets",
    "/api/settings/tokens",
)


def _derive_title(text: str) -> str:
    """First non-empty line of a memory's text, trimmed for use as a note title."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return "Memory"


def build_mcp(app: FastAPI) -> FastMCP:
    """Construct the FastMCP server bound to the given FastAPI app."""
    mcp = FastMCP(
        "Aikioku",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
    )

    # --- auth + internal proxy helpers --------------------------------------

    def _token_from_ctx(ctx: _Ctx) -> str:
        try:
            request = ctx.request_context.request
            header = request.headers.get("authorization", "") if request else ""
        except (AttributeError, ValueError):
            header = ""
        if header[:7].lower() == "bearer ":
            return header[7:]
        return ""

    def _authorize(ctx: _Ctx, write: bool) -> access_tokens.AccessToken:
        record = access_tokens.verify_token(_token_from_ctx(ctx))
        if record is None:
            raise ValueError("Unauthorized: missing or invalid access token.")
        if write and record.scope != "full":
            raise ValueError(
                "This token is read-only; write operations require a full-scope token."
            )
        return record

    async def _call(
        record: access_tokens.AccessToken,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
    ) -> Any:
        """Proxy a call to the app's own REST API over an in-process ASGI transport."""
        jwt = create_access_token({"sub": record.username}, expires_delta=timedelta(minutes=5))
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url=_INTERNAL_BASE, timeout=120.0
        ) as client:
            resp = await client.request(
                method,
                path,
                params=clean_params or None,
                json=json,
                headers={"Authorization": f"Bearer {jwt}"},
            )
        if resp.status_code >= 400:
            raise ValueError(f"API {resp.status_code}: {resp.text[:1000]}")
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.text

    # --- read tools ---------------------------------------------------------

    @mcp.tool()
    async def search_notes(ctx: _Ctx, q: str, limit: int = 20) -> Any:
        """Full-text search across all notes. Returns matching notes."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", "/api/search/", params={"q": q, "limit": limit})

    @mcp.tool()
    async def hybrid_search(ctx: _Ctx, query: str, limit: int = 20) -> Any:
        """Hybrid semantic+keyword+graph search (RRF fusion). Best general recall."""
        rec = _authorize(ctx, write=False)
        return await _call(
            rec, "POST", "/api/retrieval/hybrid", json={"query": query, "limit": limit}
        )

    @mcp.tool()
    async def list_notes(
        ctx: _Ctx,
        tag: str | None = None,
        search: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Any:
        """List notes, optionally filtered by tag or a search string."""
        rec = _authorize(ctx, write=False)
        return await _call(
            rec,
            "GET",
            "/api/notes/",
            params={"tag": tag, "search": search, "skip": skip, "limit": limit},
        )

    @mcp.tool()
    async def get_note(ctx: _Ctx, note_id: str) -> Any:
        """Fetch a single note by its id."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", f"/api/notes/{note_id}")

    @mcp.tool()
    async def list_entities(
        ctx: _Ctx,
        type: str | None = None,
        search: str | None = None,
        limit: int = 50,
    ) -> Any:
        """List knowledge-graph entities, optionally filtered by type or name."""
        rec = _authorize(ctx, write=False)
        return await _call(
            rec,
            "GET",
            "/api/entities/",
            params={"type": type, "search": search, "limit": limit},
        )

    @mcp.tool()
    async def get_entity(ctx: _Ctx, entity_id: str) -> Any:
        """Fetch a single entity (with properties and source notes) by id."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", f"/api/entities/{entity_id}")

    @mcp.tool()
    async def get_entity_subgraph(ctx: _Ctx, entity_id: str, depth: int = 2) -> Any:
        """Get the BFS neighbourhood subgraph around an entity (depth 1-5)."""
        rec = _authorize(ctx, write=False)
        return await _call(
            rec, "GET", f"/api/entities/{entity_id}/subgraph", params={"depth": depth}
        )

    @mcp.tool()
    async def graph_paths(ctx: _Ctx, source: str, target: str, max_depth: int = 3) -> Any:
        """Find paths between two entities in the knowledge graph."""
        rec = _authorize(ctx, write=False)
        return await _call(
            rec,
            "GET",
            "/api/graph/paths",
            params={"source": source, "target": target, "max_depth": max_depth},
        )

    @mcp.tool()
    async def graph_stats(ctx: _Ctx) -> Any:
        """Entity/relation counts and entity types in the knowledge graph."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", "/api/graph/stats")

    @mcp.tool()
    async def list_memories(ctx: _Ctx, entity: str | None = None) -> Any:
        """List stored memories, optionally filtered by entity."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", "/api/memory/", params={"entity": entity})

    @mcp.tool()
    async def memory_stats(ctx: _Ctx) -> Any:
        """Memory statistics: total and tier (hot/warm/cold) counts."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", "/api/memory/stats")

    @mcp.tool()
    async def stats(ctx: _Ctx) -> Any:
        """System-wide counts: notes, entities, relations, memories, cards."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", "/api/stats/")

    @mcp.tool()
    async def discover_connections(ctx: _Ctx, entity_id: str, max_distance: int = 3) -> Any:
        """Discover indirect connections from an entity (graph + embedding similarity)."""
        rec = _authorize(ctx, write=False)
        return await _call(
            rec,
            "POST",
            "/api/connections/discover",
            params={"entity_id": entity_id, "max_distance": max_distance},
        )

    @mcp.tool()
    async def ask(ctx: _Ctx, query: str, mode: str = "simple") -> Any:
        """Ask the brain a question (RAG). mode: 'simple' or 'multi_hop'. Returns a grounded answer with citations."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "POST", "/api/chat/", json={"query": query, "mode": mode})

    @mcp.tool()
    async def list_due_cards(ctx: _Ctx, limit: int = 20) -> Any:
        """List flashcards due for review (next_review <= now). Returns up to `limit` cards."""
        rec = _authorize(ctx, write=False)
        result = await _call(rec, "GET", "/api/review/due")
        if isinstance(result, list) and len(result) > limit:
            return result[:limit]
        return result

    @mcp.tool()
    async def serendipity_walk(ctx: _Ctx, start_entity_id: str, steps: int = 5) -> Any:
        """Perform a random walk through the knowledge graph from a starting entity. Returns the path of entities visited."""
        rec = _authorize(ctx, write=False)
        return await _call(
            rec,
            "POST",
            "/api/serendipity/walk",
            params={"start_entity_id": start_entity_id, "steps": steps},
        )

    @mcp.tool()
    async def export_json(ctx: _Ctx) -> Any:
        """Export all system data (notes, entities, relations, memories, cards, settings) as JSON."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", "/api/export/json")

    @mcp.tool()
    async def get_note_summary(ctx: _Ctx, note_id: str) -> Any:
        """Get a multi-level progressive summary for a note (generated by the LLM)."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "POST", f"/api/notes/{note_id}/summarize")

    @mcp.tool()
    async def get_review_stats(ctx: _Ctx) -> Any:
        """Get spaced-repetition card collection statistics (total, due, new, learning, review, suspended)."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", "/api/review/stats")

    @mcp.tool()
    async def search_memories(ctx: _Ctx, q: str, limit: int = 20) -> Any:
        """Semantic search over stored memory triples (subject-predicate-object). Returns memories ranked by embedding similarity, each with a score."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", "/api/memory/search", params={"q": q, "limit": limit})

    @mcp.tool()
    async def get_memory(ctx: _Ctx, memory_id: str) -> Any:
        """Fetch a single memory triple by its id."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", f"/api/memory/{memory_id}")

    @mcp.tool()
    async def generate_questions(ctx: _Ctx, note_id: str, count: int = 5) -> Any:
        """Generate review questions from a note (count 1-20)."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", f"/api/notes/{note_id}/questions", params={"count": count})

    @mcp.tool()
    async def list_conversations(ctx: _Ctx, limit: int = 50, before: str | None = None) -> Any:
        """Load chat history, newest-first. Pass the oldest loaded message's `created` timestamp as `before` to page further back."""
        rec = _authorize(ctx, write=False)
        return await _call(
            rec, "GET", "/api/conversations/messages", params={"limit": limit, "before": before}
        )

    @mcp.tool()
    async def git_history(ctx: _Ctx, limit: int = 20) -> Any:
        """Get the git commit history of the notes vault."""
        rec = _authorize(ctx, write=False)
        return await _call(rec, "GET", "/api/sync/git/history", params={"limit": limit})

    # --- write tools (require full scope) -----------------------------------

    @mcp.tool()
    async def create_note(
        ctx: _Ctx,
        title: str,
        content: str = "",
        path: str | None = None,
        hidden: bool = True,
    ) -> Any:
        """Create a note. Triggers entity extraction + embedding. The note is hidden
        from the user-facing UI by default (source_type='hidden'); pass hidden=False
        to make it visible in the notes list. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        if not content.strip():
            raise ValueError("Note content must not be empty")
        body = {
            "title": title,
            "content": content,
            "path": path or f"{title}.md",
            "source_type": "hidden" if hidden else "note",
        }
        return await _call(rec, "POST", "/api/notes/", json=body)

    @mcp.tool()
    async def update_note(
        ctx: _Ctx,
        note_id: str,
        title: str | None = None,
        content: str | None = None,
        hidden: bool | None = None,
    ) -> Any:
        """Update a note's title and/or content. Pass hidden=True/False to change its
        UI visibility (omit to leave it unchanged). Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        body = {k: v for k, v in {"title": title, "content": content}.items() if v is not None}
        if hidden is not None:
            body["source_type"] = "hidden" if hidden else "note"
        return await _call(rec, "PUT", f"/api/notes/{note_id}", json=body)

    @mcp.tool()
    async def delete_note(ctx: _Ctx, note_id: str) -> Any:
        """Delete a note by id. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(rec, "DELETE", f"/api/notes/{note_id}")

    @mcp.tool()
    async def extract_memories(ctx: _Ctx, note_id: str) -> Any:
        """Extract memories from a note via the LLM. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(rec, "POST", "/api/memory/extract", json={"note_id": note_id})

    @mcp.tool()
    async def summarize_note(ctx: _Ctx, note_id: str) -> Any:
        """Generate a multi-level summary for a note. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(rec, "POST", f"/api/notes/{note_id}/summarize")

    @mcp.tool()
    async def generate_cards(ctx: _Ctx, note_id: str) -> Any:
        """Generate spaced-repetition flashcards from a note. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(rec, "POST", "/api/review/cards", json={"note_id": note_id})

    @mcp.tool()
    async def review_card(ctx: _Ctx, card_id: str, rating: int) -> Any:
        """Review a flashcard with a rating (1=again, 2=hard, 3=good, 4=easy). Returns the updated card. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(
            rec, "POST", f"/api/review/cards/{card_id}/review", json={"rating": rating}
        )

    @mcp.tool()
    async def import_markdown(
        ctx: _Ctx,
        content: str,
        title: str,
        path: str | None = None,
        hidden: bool = True,
    ) -> Any:
        """Import a markdown note directly (content + title). Triggers entity extraction
        + embedding. Hidden from the UI by default; pass hidden=False to make it visible.
        Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        body = {
            "title": title,
            "content": content,
            "path": path or f"{title}.md",
            "source_type": "hidden" if hidden else "note",
        }
        return await _call(rec, "POST", "/api/notes/", json=body)

    @mcp.tool()
    async def create_memory(ctx: _Ctx, text: str, source: str = "user") -> Any:
        """Store a memory. The text is saved as a hidden note that is fully
        processed (entity extraction, embeddings, knowledge graph), so it is
        retrievable in chat the same way notes are — but it never appears in the
        user's notes list. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        title = _derive_title(text)
        body = {
            "title": title,
            "content": text,
            "path": f"{title}.md",
            "source_type": "hidden",
            "frontmatter": {"memory_source": source},
        }
        return await _call(rec, "POST", "/api/notes/", json=body)

    @mcp.tool()
    async def update_memory(
        ctx: _Ctx,
        memory_id: str,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        confidence: float | None = None,
        tier: str | None = None,
    ) -> Any:
        """Update a memory's subject/predicate/object/confidence/tier. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        body = {
            k: v
            for k, v in {
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "confidence": confidence,
                "tier": tier,
            }.items()
            if v is not None
        }
        return await _call(rec, "PUT", f"/api/memory/{memory_id}", json=body)

    @mcp.tool()
    async def delete_memory(ctx: _Ctx, memory_id: str) -> Any:
        """Delete a memory triple by id. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(rec, "DELETE", f"/api/memory/{memory_id}")

    @mcp.tool()
    async def consolidate_memories(ctx: _Ctx) -> Any:
        """Run the memory consolidation pipeline (dedup, merge, tiering). Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(rec, "POST", "/api/memory/consolidate")

    @mcp.tool()
    async def auto_tag_note(ctx: _Ctx, note_id: str) -> Any:
        """Auto-generate tags for a note using the LLM. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(rec, "POST", f"/api/tags/auto/{note_id}")

    @mcp.tool()
    async def clear_conversations(ctx: _Ctx) -> Any:
        """Delete the entire chat history for the token's owner. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(rec, "DELETE", "/api/conversations/messages")

    @mcp.tool()
    async def git_commit(ctx: _Ctx, message: str) -> Any:
        """Stage all changes in the notes vault and commit them. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(rec, "POST", "/api/sync/git/commit", params={"message": message})

    @mcp.tool()
    async def scan_anomalies(ctx: _Ctx) -> Any:
        """Run all knowledge-base anomaly detection checks and return the results. Requires a full-scope token."""
        rec = _authorize(ctx, write=True)
        return await _call(rec, "POST", "/api/anomaly/scan")

    # --- generic escape hatch (full surface) --------------------------------

    @mcp.tool()
    async def call_api(
        ctx: _Ctx,
        method: str,
        path: str,
        query: dict[str, Any] | None = None,
        body: Any | None = None,
    ) -> Any:
        """Call any other Aikioku REST endpoint not covered by a dedicated tool.

        method: GET/POST/PUT/DELETE. path: must start with '/api/'. Non-GET methods
        require a full-scope token. Auth, setup, secret and token routes are blocked.
        """
        method = method.upper()
        write = method != "GET"
        rec = _authorize(ctx, write=write)
        if not path.startswith("/api/"):
            raise ValueError("path must start with '/api/'.")
        if any(path.startswith(p) for p in _CALL_API_DENY):
            raise ValueError(f"Access to '{path}' is not permitted via MCP.")
        return await _call(rec, method, path, params=query, json=body)

    return mcp
