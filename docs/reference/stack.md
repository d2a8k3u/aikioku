# Technology Stack

## Backend (Python 3.11+)

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | FastAPI | ≥0.110 |
| Server | Uvicorn | ≥0.29 |
| Validation | Pydantic | ≥2.6 |
| Settings | pydantic-settings | ≥2.2 |
| ORM | SQLAlchemy | ≥2.0 |
| Async SQLite | aiosqlite | ≥0.20 |
| HTTP client | httpx | ≥0.27 |
| LLM — Anthropic | anthropic | ≥0.25 |
| LLM — OpenAI | openai | ≥1.16 |
| Async files | aiofiles | ≥23.0 |
| Multipart | python-multipart | ≥0.0.9 |
| File watching | watchfiles | ≥0.21 |
| Logging | structlog | ≥24.1 |
| Metrics | prometheus-client | ≥0.20 |
| Math | numpy | ≥1.24, <2.0 |
| Graph DB | kuzu | ≥0.4.2 |
| Vector DB | chromadb | ≥0.4.24 |
| Full-text search | tantivy | ≥0.11.0 |
| PDF parser | PyMuPDF | ≥1.24.0 |
| DOCX parser | python-docx | ≥1.1.0 |
| HTML parser | readability-lxml | ≥0.8.1 |
| JWT | python-jose[cryptography] | ≥3.3.0 |
| Encryption | cryptography | ≥42 |
| Password hashing | passlib[bcrypt] | ≥1.7.4 |
| Rate limiting | slowapi | ≥0.1.9 |
| MCP server | mcp | ≥1.27, <2 |
| Flashcards | genanki | ≥0.13.0 |

### Dev Dependencies

| Tool | Version |
|------|---------|
| pytest | ≥8.1 |
| pytest-asyncio | ≥0.23 |
| pytest-cov | ≥5.0 |
| ruff | ≥0.3 |
| mypy | ≥1.9 |

## Frontend (Node.js 20+)

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Next.js | ^16.2.9 |
| UI library | React | 18.3.1 |
| Styling | Tailwind CSS | ^4 |
| Data fetching | TanStack React Query | ^5 |
| State | Zustand | ^5 |
| Markdown rendering | react-markdown + remark-gfm | ^10 / ^4 |
| Rich text | TipTap (@tiptap/react) + tiptap-markdown | ^2 / 0.8.10 |
| Animation | framer-motion | ^11 |
| Class utilities | clsx + tailwind-merge | ^2 / ^2 |
| Canvas visuals | Three.js | ^0.170.0 |

`react-force-graph-3d` and `three-spritetext` are present as dependencies but are
not currently wired into the UI.

### Dev Dependencies

| Tool | Version |
|------|---------|
| TypeScript | ^5 |
| Vitest | ^4.1.9 |
| @vitejs/plugin-react | ^6.0.2 |
| @testing-library/react | ^16.3.2 |
| @testing-library/jest-dom | ^6.9.1 |
| jsdom | ^29.1.1 |
| PostCSS | ^8 |

## Infrastructure

| Component | Technology |
|-----------|-----------|
| Containerization | Docker + Docker Compose |
| server image | python:3.11-slim (multi-stage) |
| dashboard image | node:20-alpine (multi-stage) |
| Data persistence | Docker named volume (`server-data`) |
| Health checks | Docker HEALTHCHECK + `/health` endpoint |
