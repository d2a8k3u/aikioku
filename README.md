# AI-Augmented Personal Knowledge Management

A self-hosted personal knowledge management system that combines note-taking,
knowledge graphs, vector search, and AI reasoning into a single application.

**Stack:** FastAPI + Kuzu + ChromaDB + Tantivy + Next.js  
**Deployment:** Docker Compose (one command)  
**Access:** Web UI at `:3369`, MCP server at `:8869/mcp`

## Quick Start

```bash
git clone https://github.com/d2a8k3u/aikioku.git
cd aikioku
make up
open http://localhost:3369
```

No `.env` file needed — all secrets and runtime configuration live in an encrypted
database, set through the setup wizard on first launch.

## Architecture

```
aikioku/
├── apps/
│   ├── server/          # FastAPI backend
│   └── dashboard/       # Next.js frontend
├── docs/                # Documentation
├── docker/              # Container runtime scripts
├── Dockerfile           # Multi-stage (server + dashboard targets)
├── docker-compose.yml   # Full stack
├── Makefile             # up, down, dev-up, rebuild, logs, test
└── README.md
```

Seven integrated layers:

1. **Ingestion** — notes, web pages, PDFs, DOCX, images, audio, email
2. **Knowledge Representation** — property graph (Kuzu) + dense vectors (ChromaDB) + sparse index (Tantivy)
3. **Retrieval** — hybrid RRF fusion (dense + sparse + graph), multi-hop reasoning
4. **Memory** — episodic extraction, consolidation, hot/warm/cold tiering
5. **Reasoning** — grounded RAG with citations, connection discovery, question generation
6. **Cognitive Augmentation** — spaced repetition (Anki-compatible), progressive summarization, serendipity walks
7. **Interface** — markdown editor, conversational RAG chat, spaced-repetition review

Full documentation: **[docs/index.md](docs/index.md)** — architecture, API, MCP clients, development guides

## MCP Server

The backend exposes a full Model Context Protocol server. Any MCP client
(Claude Desktop, Cursor, Hermes Agent, Claude Code) can query and edit your
knowledge base as tools.

- **Endpoint:** `http://localhost:8869/mcp`
- **Auth:** Personal Access Token (PAT) generated in Settings → API access
- **Tools:** 41 typed tools — search, graph exploration, note + memory CRUD, semantic memory search, RAG Q&A, flashcards, import/export

Client setup guides: [Claude Code](docs/clients/claude-code.md) · [Claude Desktop](docs/clients/claude-desktop.md) · [Cursor](docs/clients/cursor.md) · [Hermes Agent](docs/clients/hermes-agent.md)

## Development

```bash
# Backend
cd apps/server
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # unit tests (integration skipped by default)

# Frontend
cd apps/dashboard
npm install
npm run dev                     # http://localhost:3000
npm test                        # vitest

# Or use Makefile
make dev-server                 # backend with hot reload
make dev-dashboard              # frontend with hot reload
make test                       # all tests
make lint-server                # ruff
make typecheck-server           # mypy
```

## Ports

| Service   | Container | Host (mapped) |
|-----------|-----------|---------------|
| Server    | 8000      | 8869          |
| Dashboard | 3000      | 3369          |

## Contributing

Contributions are welcome! Open an issue to discuss a change, or submit a pull request.

## License

MIT
