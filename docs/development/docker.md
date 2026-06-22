# Docker Deployment

The stack is Docker-first. Prefer the Makefile targets (`make up`, `make down`,
`make dev-up`) over hand-rolled `docker compose` invocations.

## Quick Start

```bash
make up        # production stack (docker-compose.yml)
# or
make dev-up    # hot-reload dev stack (docker-compose.dev.yml)
```

Two services: `server` (FastAPI) and `dashboard` (Next.js).

## Ports

Both are bound to `127.0.0.1` only.

| Service   | Container | Host (mapped) |
|-----------|-----------|---------------|
| server    | 8000      | 8869          |
| dashboard | 3000      | 3369          |

## Volumes

| Mount | Purpose |
|-------|---------|
| `./apps/server/src:/app/src` | Backend source |
| `./apps/server/tests:/app/tests` | Backend tests |
| `./apps/dashboard/src:/app/src` | Frontend source |
| `server-data:/data` | Persistent data (notes, DB, indices) |

## Data Directory

```
/data/
├── notes/          # Markdown note files
├── kuzu/           # Kuzu graph database
├── chroma/         # ChromaDB vector store
└── sqlite/         # SQLite databases
    ├── aikioku.db
    └── secret.key  # Master encryption key
```

## Commands

```bash
# Logs
docker compose logs -f server
docker compose logs -f dashboard

# Run tests in the server container
docker compose exec server pytest
docker compose exec server pytest -m "not slow"

# Rebuild after dependency changes
make rebuild

# Stop
make down
```

## Dockerfile

A single multi-stage `Dockerfile` builds both services via named targets:

- **`server`** — builder on `python:3.11`, runtime on `python:3.11-slim`; `EXPOSE 8000`; `CMD uvicorn src.main:app --host 0.0.0.0 --port 8000` (single worker — Kuzu is single-writer).
- **`dashboard`** — `dashboard-deps` → `dashboard-builder` (`npm run build`) → `dashboard` runtime on `node:20-alpine`; `EXPOSE 3000`; `CMD npm start`.
- **`server-dev` / `dashboard-dev`** — used by `docker-compose.dev.yml` for hot reload (`uvicorn --reload`, `next dev`).

## Health Checks

The `server` image has a built-in health check:

```bash
curl http://localhost:8869/health
```

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
```

## Security Notes

- Both services bind to `127.0.0.1` only — not exposed to the network.
- No `.env` file — secrets live in the encrypted SQLite database.
- Master key at `/data/sqlite/secret.key` — auto-generated, never committed.
- PAT authentication on `/mcp` for external access.
