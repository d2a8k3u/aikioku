# REST API

The backend exposes 95 endpoints across 29 routers. All application routes are
prefixed with `/api/` unless noted. Paths below are written without trailing
slashes; FastAPI accepts both forms.

Authentication is optional by default (`auth_required: false` in `config.py`).
Enable it in Settings for multi-user deployments. When enabled, obtain a JWT via
`/api/auth/login` and send it as `Authorization: Bearer <token>`.

## Auth

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/register` | Register a user, returns a JWT |
| POST | `/api/auth/login` | Log in (OAuth2 password form), returns a JWT |
| GET | `/api/auth/me` | Current authenticated user |

## Notes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/notes` | List notes (query: `tag`, `search`, `skip`, `limit`) |
| POST | `/api/notes` | Create note (`title`, `content`, `path`, `source_type`); triggers extraction + embedding |
| GET | `/api/notes/{note_id}` | Get note by id |
| PUT | `/api/notes/{note_id}` | Update note title/content |
| DELETE | `/api/notes/{note_id}` | Delete note |
| GET | `/api/notes/{note_id}/backlinks` | Notes linking to this note |
| GET | `/api/notes/{note_id}/related` | Related notes via hybrid similarity |
| GET | `/api/notes/{note_id}/history` | Git commit history for the note |
| GET | `/api/notes/{note_id}/diff` | Git diff for the note |
| POST | `/api/notes/{note_id}/summarize` | Multi-level progressive summary |
| GET | `/api/notes/{note_id}/questions` | Generate review questions from the note |

## Search & Retrieval

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/search` | Full-text search across notes |
| POST | `/api/retrieval/hybrid` | Hybrid search (dense + sparse + graph, RRF fusion) |

## Graph & Entities

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/graph/entities` | List entities (query: `type`, `search`, `limit`) |
| GET | `/api/graph/entities/{entity_id}` | Get entity |
| GET | `/api/graph/entities/{entity_id}/relations` | Relations for an entity |
| GET | `/api/graph/relations` | List relations |
| GET | `/api/graph/paths` | Find paths between two entities |
| GET | `/api/graph/stats` | Entity/relation counts and types |
| GET | `/api/entities` | List entities (filter by type/name) |
| GET | `/api/entities/{entity_id}` | Get entity with properties and source notes |
| GET | `/api/entities/{entity_id}/subgraph` | BFS neighbourhood subgraph |
| POST | `/api/connections/discover` | Discover indirect connections from an entity (query: `entity_id`, `max_distance`) |
| POST | `/api/serendipity/walk` | Random walk through the graph (body: `start_entity_id`, `steps`) |
| GET | `/api/serendipity/surprise` | Compute a surprise/novelty score (query: `entity_id`) |
| GET | `/api/schema/types` | List induced entity/relation types |
| POST | `/api/schema/induce` | Induce a graph schema from the corpus |

## Memory

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/memory` | List memory triples (query: `entity`) |
| POST | `/api/memory` | Create triples from free text (`text`, `source`) |
| GET | `/api/memory/{memory_id}` | Fetch a memory triple |
| PUT | `/api/memory/{memory_id}` | Update subject/predicate/object/confidence/tier |
| DELETE | `/api/memory/{memory_id}` | Delete a memory triple |
| GET | `/api/memory/search` | Semantic search over memory triples (query: `q`, `limit`) |
| GET | `/api/memory/stats` | Tier counts (hot/warm/cold) |
| POST | `/api/memory/extract` | Extract memories from a note (`note_id`) |
| POST | `/api/memory/consolidate` | Run the dedup/merge/tiering pipeline |
| POST | `/api/memory/backfill-embeddings` | Embed all stored memories (background) |
| GET | `/api/memory/backfill-embeddings/status` | Backfill progress |

## Chat

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Ask a RAG question (`simple` or `multi_hop`), grounded answer |
| GET | `/api/chat/stream` | Streaming chat over SSE (GET variant) |
| POST | `/api/chat/stream` | Streaming chat over SSE (POST variant) |
| POST | `/api/chat/stop` | Stop an in-progress generation |
| GET | `/api/conversations/messages` | Load chat history (paginated, newest-first) |
| DELETE | `/api/conversations/messages` | Clear the caller's chat history |

## Review (Spaced Repetition)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/review/due` | Cards due for review |
| POST | `/api/review/cards` | Generate flashcards from a note |
| POST | `/api/review/cards/{card_id}/review` | Submit a review rating (`rating`: 1–4) |
| GET | `/api/review/stats` | Spaced-repetition collection stats |

## Import / Export

Import endpoints are format-specific. Each is rate-limited to 20 requests/minute.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/import/markdown` | Import a markdown note |
| POST | `/api/import/obsidian` | Import an Obsidian vault |
| POST | `/api/import/bulk` | Bulk import notes |
| POST | `/api/import/pdf` | Import a PDF |
| POST | `/api/import/docx` | Import a DOCX |
| POST | `/api/import/audio` | Import / transcribe audio |
| POST | `/api/import/image` | Import / OCR an image |
| POST | `/api/import/web` | Import a web page |
| POST | `/api/import/email` | Import an email |
| POST | `/api/import/logseq` | Import a Logseq export (zip) |
| POST | `/api/import/notion` | Import a Notion export (zip) |
| POST | `/api/import/roam` | Import a Roam export (zip) |
| GET | `/api/export/note/{note_id}` | Export a single note as markdown |
| GET | `/api/export/all` | Export all notes (archive) |
| GET | `/api/export/json` | Export the entire system as JSON |
| GET | `/api/export/anki` | Export flashcards to Anki |
| GET | `/api/export/bibtex` | Export references as BibTeX |

## Settings & Tokens

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/settings` | Read app settings |
| PUT | `/api/settings` | Update settings (may rebuild runtime) |
| GET | `/api/settings/secrets` | List secret keys (values masked) |
| PUT | `/api/settings/secrets` | Set a secret value |
| DELETE | `/api/settings/secrets/{key}` | Delete a secret |
| GET | `/api/settings/models` | List available chat models |
| GET | `/api/settings/embedding-models` | List available embedding models |
| GET | `/api/settings/tokens` | List personal access tokens |
| POST | `/api/settings/tokens` | Create a PAT (`name`, `scope`) |
| DELETE | `/api/settings/tokens/{token_id}` | Revoke a PAT |

The daily LLM budget (`llm_daily_budget_usd`) is editable via `PUT /api/settings`
and applies live (no restart).

## Budget

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/budget/status` | Daily-budget state (`active` / `warning` / `paused`), today's spend, queue depth, and the next reset time |

## Setup

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/setup/status` | Whether the app is configured |
| POST | `/api/setup` | First-run setup (admin account, LLM config) |
| POST | `/api/setup/test` | Test an LLM/provider connection |

## Cognitive State

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/cognitive/state` | Read current cognitive state |
| POST | `/api/cognitive/state` | Record a cognitive signal (`type`, `value`) |

## Git Sync

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/sync/git/commit` | Stage all and commit the vault |
| GET | `/api/sync/git/history` | Vault commit history |
| GET | `/api/sync/git/diff/{note_id}` | Diff for a specific note |

## Tags & Anomaly

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/tags/auto/{note_id}` | Auto-generate tags for a note via LLM |
| POST | `/api/anomaly/scan` | Run all anomaly-detection checks |
| GET | `/api/anomaly/recent` | Recently detected anomalies |
| POST | `/api/anomaly/{anomaly_id}/resolve` | Mark an anomaly resolved |

## Plugins

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/plugins` | List registered plugins |
| POST | `/api/plugins` | Register a plugin (manifest) |
| POST | `/api/plugins/{name}/activate` | Activate a plugin |
| POST | `/api/plugins/{name}/deactivate` | Deactivate a plugin |
| DELETE | `/api/plugins/{name}` | Unregister a plugin |
| GET | `/api/plugins/{name}/hooks` | List a plugin's hook names |

## Buddy

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/buddy/profile` | Get the buddy profile |
| PUT | `/api/buddy/profile` | Update the buddy profile |
| GET | `/api/buddy/greeting` | Contextual greeting |
| GET | `/api/buddy/cards` | Buddy dashboard cards |

## Stats & Admin

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/stats` | System-wide counts (notes, entities, relations, memories, cards) |
| POST | `/api/admin/reextract` | Re-run entity extraction over all notes |
| GET | `/api/admin/reextract/status` | Re-extraction job status |
| GET | `/api/admin/reembed/status` | Re-embedding job status |

## Service routes

These are defined on the application, outside the `/api` routers.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness / version check |
| GET | `/metrics` | Prometheus metrics exposition |
| GET | `/` | API root banner |
| WS | `/ws/events` | Real-time event stream (reembed progress, chat streaming, extraction status, `budget.status`) |

The MCP server is mounted at `/mcp` — see [MCP Server](mcp.md).

## Rate limiting

- Import endpoints: 20 requests/minute per client.
- Chat endpoints: 10 requests/minute per client.
- LLM calls are bound by a daily budget (`llm_daily_budget_usd`, default `5.0`).
  When it is reached, **interactive** calls (chat, on-demand tools) return
  `429 Too Many Requests`, while **ingestion** (notes, memories) is still accepted
  and persisted — its LLM processing is queued and runs after the budget resets at
  00:00 UTC (or is raised in Settings). See [`GET /api/budget/status`](#budget).
