
# Aikioku

AI-Augmented Personal Knowledge Management — a self-hosted system that combines
note-taking, knowledge graphs, vector search, and AI reasoning.

**Stack:** FastAPI + Kuzu + ChromaDB + Tantivy + Next.js  
**Deployment:** Docker Compose (`server` + `dashboard`)  
**Access:** Web UI at `:3369`, MCP server at `:8869/mcp`

## Quick Start

```bash
docker compose up -d
open http://localhost:3369
```

Complete the first-run setup wizard — no `.env` file needed. All secrets and
configuration live in an encrypted database.

## What It Does

- **Capture** — notes, web pages, PDFs, DOCX, images, audio, email
- **Organize** — automatic knowledge graph, entity extraction, semantic indexing
- **Retrieve** — hybrid search (dense + sparse + graph), multi-hop reasoning
- **Remember** — episodic memory extraction, consolidation, spaced repetition
- **Reason** — grounded RAG with citations, connection discovery
- **Connect** — MCP server for Claude Desktop, Cursor, Claude Code, Hermes Agent

## Documentation

| Section | Description |
|---------|-------------|
| [Architecture](architecture/index.md) | 7-layer system design |
| [API](api/index.md) | REST API + MCP server |
| [Clients](clients/index.md) | Connect Claude, Cursor, Hermes |
| [Development](development/index.md) | Setup, testing, Docker |
| [Reference](reference/index.md) | Stack, project structure |

## License

MIT
