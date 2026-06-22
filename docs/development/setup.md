# Development Setup

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (optional, but the recommended way to run the full stack)

## Backend

```bash
cd apps/server
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Verify

```bash
pytest                          # unit tests (integration skipped by default)
ruff check src/ tests/          # lint
mypy src/                       # type check (strict)
```

`pytest` defaults to `-m 'not integration'`; run `pytest -m integration` to
include integration tests.

## Frontend

```bash
cd apps/dashboard
npm install
npm run dev                     # http://localhost:3000
```

### Verify

```bash
npm test                        # vitest
npx tsc --noEmit                # type check (there is no ESLint config)
npm run build                   # production build check
```

The dashboard has no ESLint setup, so `npm run lint` is unreliable — use
`npx tsc --noEmit` to verify types instead.

## Ports

Run outside Docker, the apps use their container ports: dashboard `3000`,
backend `8000`. The Docker stack remaps these to `3369` and `8869` on the host.

## Configuration

No `.env` file. Configuration is stored in an encrypted SQLite database:

- **Secrets:** API keys, tokens — AES-256-GCM encrypted
- **Runtime config:** LLM provider, embedding provider, budgets
- **Master key:** auto-generated next to the SQLite database (e.g. `/data/sqlite/secret.key`); never commit it

The first-run setup wizard (`/setup`) creates the admin account and configures
the LLM provider.

### LLM Providers

| Provider | Config Key | Notes |
|----------|-----------|-------|
| Ollama | `llm_provider: ollama` | Local or remote Ollama instance (default) |
| OpenRouter | `llm_provider: openrouter` | OpenAI-compatible chat proxy |
| Anthropic | `llm_provider: anthropic` | Direct Anthropic API |
| OpenAI | `llm_provider: openai` | Direct OpenAI API |

### Embedding Providers

| Provider | Config Key | Notes |
|----------|-----------|-------|
| Ollama | `embedding_provider: ollama` | Uses `ollama_embedding_model` (default) |
| HuggingFace | `embedding_provider: huggingface` | Requires `hf_api_key`, uses `hf_embedding_model` |
| OpenAI | `embedding_provider: openai` | Uses `openai_embedding_model` |

### Daily LLM budget

`llm_daily_budget_usd` (default `5.0`) caps daily LLM spend. Set it in the
first-run setup wizard and edit it any time under **Settings → LLM provider**;
changes apply live (no restart). `0` disables the cap.

When today's spend reaches the cap, the dashboard shows a paused banner and
LLM-backed processing **pauses project-wide**:

- New **notes and memories are still accepted and saved**, but their LLM
  processing (entity extraction, embedding, memory extraction) is **queued** and
  runs automatically once the budget resets.
- Interactive calls (chat, on-demand tools) return `429` with a clear message.
- A warning banner appears once spend crosses `llm_budget_warning_fraction`
  (default `0.9`) while still processing.

Spend is summed per UTC day, so the budget resets at **00:00 UTC**. Raising the
budget in Settings resumes the queued work immediately.

## Project Structure

```
aikioku/
├── README.md
├── docker-compose.yml
├── docker-compose.dev.yml
├── Dockerfile                  # multi-stage (server + dashboard targets)
├── Makefile
├── mkdocs.yml
├── docs/                       # GitHub Pages documentation
├── apps/
│   ├── server/
│   │   ├── pyproject.toml
│   │   ├── src/
│   │   │   ├── main.py         # FastAPI entry point
│   │   │   ├── config.py       # Pydantic settings
│   │   │   ├── mcp_server.py   # FastMCP server
│   │   │   ├── api/            # 29 REST routers
│   │   │   ├── knowledge/      # Graph, embeddings, entity resolution
│   │   │   ├── retrieval/      # Dense, sparse, graph, fusion
│   │   │   ├── memory/         # Extraction, consolidation, tiering
│   │   │   ├── reasoning/      # RAG, connections, multi-hop
│   │   │   ├── augmentation/   # Spaced repetition, summarization
│   │   │   ├── models/         # Note, Entity, Relation, Memory, Card
│   │   │   ├── llm/            # Provider abstraction
│   │   │   ├── storage/        # File import, note store, git sync
│   │   │   ├── ingestion/      # PDF, DOCX, HTML, image, audio parsers
│   │   │   └── plugins/        # Plugin manager
│   │   └── tests/              # pytest suite (see Testing)
│   └── dashboard/
│       ├── package.json
│       ├── src/
│       │   ├── app/            # App Router pages
│       │   ├── components/     # MarkdownEditor, HUD kit, ErrorBoundary
│       │   ├── hooks/          # WebSocket, data hooks
│       │   ├── lib/            # API client, WebSocket, chat events
│       │   ├── stores/         # Zustand stores
│       │   └── types/          # Shared TypeScript types
│       └── public/             # Static assets
```
