
# MCP Server

Aikioku exposes a **Model Context Protocol (MCP)** server at `/mcp`, providing
41 tools (24 read, 16 write, and one generic `call_api`). Any MCP-compatible
client can connect and interact with your knowledge base as tools.

- **Endpoint:** `http://localhost:8869/mcp` (Docker default; `http://localhost:8000/mcp` when running the server outside Docker)
- **Transport:** Streamable-HTTP (FastMCP)
- **Auth:** Personal Access Token (PAT) in `Authorization: Bearer <token>` header

## Getting a Token

1. Open the Aikioku web UI → **Settings → API access**
2. Create a token with scope `read` or `full`
3. Copy the token immediately — it's shown only once (SHA-256 hashed in DB)

## Token Scopes

| Scope  | Capabilities |
|--------|-------------|
| `read` | All read tools (search, retrieval, graph, RAG, export) |
| `full` | All read + write tools (create/update/delete notes, generate cards, import) |

## Read Tools

### Search & Retrieval

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_notes` | Full-text search | `q`, `limit` (default 20) |
| `hybrid_search` | RRF fusion (dense+sparse+graph) | `query`, `limit` (default 20) |
| `list_notes` | List notes, filter by tag/search | `tag`, `search`, `skip`, `limit` |
| `get_note` | Fetch note by ID | `note_id` |
| `ask` | RAG question answering | `query`, `mode` ("simple" or "multi_hop") |

### Knowledge Graph

| Tool | Description | Parameters |
|------|-------------|------------|
| `list_entities` | List entities | `type`, `search`, `limit` |
| `get_entity` | Get entity with properties | `entity_id` |
| `get_entity_subgraph` | BFS neighbourhood | `entity_id`, `depth` (1–5) |
| `graph_paths` | Find paths between entities | `source`, `target`, `max_depth` |
| `graph_stats` | Entity/relation counts | — |
| `discover_connections` | Indirect connections | `entity_id`, `max_distance` |

### Memory

Memories are subject-predicate-object triples (e.g. `Alice — works_at — Acme`)
with a confidence, source, decay `vitality_score`, and a hot/warm/cold `tier`.
They are stored in SQLite and embedded into a dedicated `memories` vector
collection, so `search_memories` is semantic (embedding similarity), distinct
from the entity-exact `list_memories` filter.

Newly created/updated memories are embedded automatically. Memories that predate
this feature (or a model/dimension change) become searchable after a one-time
backfill: `POST /api/memory/backfill-embeddings` (poll
`GET /api/memory/backfill-embeddings/status`), reachable via the `call_api` tool.

| Tool | Description | Parameters |
|------|-------------|------------|
| `list_memories` | List memory triples, optionally filtered by entity (exact match) | `entity` (optional) |
| `search_memories` | Semantic search over memory triples; each hit carries a `score` | `q`, `limit` (default 20) |
| `get_memory` | Fetch a single memory triple by id | `memory_id` |
| `memory_stats` | Tier counts (hot/warm/cold) | — |

### Conversations & Augmentation

| Tool | Description | Parameters |
|------|-------------|------------|
| `list_conversations` | Load chat history, newest-first (paginated) | `limit` (default 50), `before` (cursor) |
| `generate_questions` | Generate review questions from a note | `note_id`, `count` (1–20, default 5) |
| `git_history` | Git commit history of the notes vault | `limit` (default 20) |

### Review & Serendipity

| Tool | Description | Parameters |
|------|-------------|------------|
| `list_due_cards` | Cards due for review | `limit` (default 20) |
| `get_review_stats` | Card collection stats | — |
| `serendipity_walk` | Random graph walk | `start_entity_id`, `steps` |

### Summarization & Export

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_note_summary` | Progressive summary | `note_id` |
| `stats` | System-wide counts | — |
| `export_json` | Export all data as JSON | — |

## Write Tools (require `full` scope)

| Tool | Description | Parameters |
|------|-------------|------------|
| `create_note` | Create note | `title`, `content`, `path` |
| `update_note` | Update note | `note_id`, `title`, `content` |
| `delete_note` | Delete note | `note_id` |
| `create_memory` | Create memory triples from free text (LLM-parsed, then embedded) | `text`, `source` (default `user`) |
| `update_memory` | Update a memory triple (re-embeds on triple change) | `memory_id`, `subject`, `predicate`, `object`, `confidence`, `tier` |
| `delete_memory` | Delete a memory triple (and its vector) | `memory_id` |
| `extract_memories` | Extract memories from a note | `note_id` |
| `consolidate_memories` | Run the consolidation pipeline (dedup / merge / re-tier) | — |
| `auto_tag_note` | Auto-generate tags for a note | `note_id` |
| `clear_conversations` | Delete the caller's entire chat history | — |
| `git_commit` | Stage all vault changes and commit | `message` |
| `scan_anomalies` | Run all knowledge-base anomaly checks | — |
| `summarize_note` | Generate summary | `note_id` |
| `generate_cards` | Generate flashcards | `note_id` |
| `review_card` | Review flashcard | `card_id`, `rating` (1–4) |
| `import_markdown` | Import markdown note | `content`, `title`, `path` |

## Escape Hatch

| Tool | Description | Parameters |
|------|-------------|------------|
| `call_api` | Call any REST endpoint | `method`, `path`, `query`, `body` |

**Security:** `call_api` blocks `/api/auth`, `/api/setup`, `/api/settings/secrets`,
and `/api/settings/tokens` to prevent privilege escalation.

## Architecture

Built with `FastMCP` (streamable-HTTP), mounted into the FastAPI app at `/mcp`.
Each tool is a thin wrapper that re-enters the same app over an in-process
`httpx` ASGI transport — no duplicated logic.

**Auth flow:**
1. Client sends `Authorization: Bearer <pat>`
2. Middleware validates PAT at `/mcp` boundary
3. Each tool reads PAT from request context, verifies scope
4. Internal REST calls carry short-lived JWT (5 min) minted for token owner

**Security:**
- PATs SHA-256 hashed — leaked DB cannot reveal usable tokens
- Write tools require `full` scope
- `call_api` blocks auth/setup/secret/token routes
- Internal JWTs short-lived (5 minutes)

## Client Configuration

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "aikioku": {
      "type": "streamableHttp",
      "url": "http://localhost:8869/mcp",
      "headers": {
        "Authorization": "Bearer sbk_your_token_here"
      }
    }
  }
}
```

### Cursor

`.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "aikioku": {
      "type": "streamableHttp",
      "url": "http://localhost:8869/mcp",
      "headers": {
        "Authorization": "Bearer sbk_your_token_here"
      }
    }
  }
}
```

### Hermes Agent

`~/.hermes/profiles/default/mcp.json`:

```json
{
  "mcpServers": {
    "aikioku": {
      "type": "streamableHttp",
      "url": "http://localhost:8869/mcp",
      "headers": {
        "Authorization": "Bearer sbk_your_token_here"
      }
    }
  }
}
```

### Claude Code

See [Claude Code setup guide](../clients/claude-code.md).

### Generic Client

Any MCP client supporting streamable-HTTP:

```
http://<host>:8869/mcp
Authorization: Bearer <pat>
```

### stdio-only Clients

Bridge with `mcp-remote`:

```json
{
  "mcpServers": {
    "aikioku": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:8869/mcp",
               "--header", "Authorization: Bearer sbk_your_token_here"]
    }
  }
}
```

## Remote Access

Compose binds to `127.0.0.1` only. For remote access, put a reverse proxy
(nginx, Caddy, Cloudflare Tunnel) in front of port 8869. The PAT gate on
`/mcp` makes this safe. Do not change the bind to `0.0.0.0` without
understanding the exposure.

## Rate Limiting

Tools re-enter the same REST app, so LLM-backed tools are bound by the daily LLM
budget (`llm_daily_budget_usd`, default `5.0`). When it is reached, interactive
tools return an error, while tools that ingest content (notes, memories) still
persist the content and queue its LLM processing until the budget resets at 00:00
UTC. The per-minute HTTP rate limits (import, chat) are keyed by client IP at the
REST boundary.
