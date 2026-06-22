# Project Structure

```
aikioku/
├── README.md
├── docker-compose.yml              # production stack
├── docker-compose.dev.yml          # hot-reload dev stack
├── Dockerfile                      # multi-stage (server + dashboard targets)
├── Makefile                        # up, down, dev-up, test, lint, ...
├── mkdocs.yml                      # MkDocs (Material) docs config
│
├── docs/                           # documentation site (this content)
│   ├── index.md
│   ├── architecture/               # 7-layer system design
│   ├── api/                        # rest.md (29 routers), mcp.md (41 tools)
│   ├── clients/                    # MCP client setup guides
│   ├── development/                # setup, testing, docker
│   └── reference/                  # stack, project structure
│
├── apps/
│   ├── server/                     # FastAPI backend
│   │   ├── pyproject.toml
│   │   ├── tests/                  # pytest suite (89 files)
│   │   └── src/
│   │       ├── main.py             # FastAPI app entry point (+ /health, /metrics, /)
│   │       ├── config.py           # Pydantic settings
│   │       ├── runtime_config.py   # DB-backed runtime config
│   │       ├── auth.py             # JWT auth
│   │       ├── access_tokens.py    # Personal Access Tokens
│   │       ├── secrets_store.py    # AES-256-GCM encrypted secrets
│   │       ├── events.py           # Event bus
│   │       ├── limiter.py          # Rate limiter
│   │       ├── observability.py    # Logging + Prometheus metrics
│   │       ├── mcp_server.py       # FastMCP server (41 tools)
│   │       │
│   │       ├── api/                # 29 REST routers
│   │       │   ├── notes.py            # Note CRUD, backlinks, related, history
│   │       │   ├── search.py           # Full-text search
│   │       │   ├── retrieval.py        # Hybrid (RRF) retrieval
│   │       │   ├── graph.py            # Graph queries
│   │       │   ├── entities.py         # Entity queries + subgraph
│   │       │   ├── connections.py      # Indirect entity connections
│   │       │   ├── serendipity.py      # Random walks, surprise
│   │       │   ├── schema.py           # Graph schema induction
│   │       │   ├── memory.py           # Memory CRUD, search, consolidation
│   │       │   ├── chat.py             # Streaming RAG chat
│   │       │   ├── conversations.py    # Chat history
│   │       │   ├── review.py           # Spaced repetition
│   │       │   ├── summarization.py    # Progressive summaries
│   │       │   ├── question_gen.py     # Question generation
│   │       │   ├── auto_tag.py         # LLM tag suggestions
│   │       │   ├── anomaly.py          # Anomaly detection
│   │       │   ├── cognitive_state.py  # Cognitive signals
│   │       │   ├── buddy.py            # Buddy profile + dashboard cards
│   │       │   ├── import_export.py    # Format-specific import/export
│   │       │   ├── git_sync.py         # Vault git commit/history/diff
│   │       │   ├── plugins.py          # Plugin manager
│   │       │   ├── stats.py            # System counts
│   │       │   ├── admin.py            # Re-extraction jobs
│   │       │   ├── reembed.py          # Re-embedding status
│   │       │   ├── settings.py         # Settings + secrets
│   │       │   ├── tokens.py           # PAT management
│   │       │   ├── auth.py             # Register, login, me
│   │       │   ├── setup.py            # First-run wizard
│   │       │   └── websocket.py        # /ws/events real-time stream
│   │       │
│   │       ├── knowledge/          # embeddings, entity_resolution, graph,
│   │       │                       #   pipeline, schema, schema_induction, reembed
│   │       ├── retrieval/          # dense, sparse, graph_retrieval, fusion,
│   │       │                       #   conversation_retrieval, search_result
│   │       ├── memory/             # extraction, consolidation, tiers
│   │       ├── reasoning/          # rag, multi_hop, connections, anomaly, question_gen
│   │       ├── augmentation/       # spaced_repetition (SM-2), summarization,
│   │       │                       #   card_auto, serendipity, auto_tag, cognitive_state
│   │       ├── models/             # note, entity, relation, memory, card, conversation, user
│   │       ├── llm/                # base, factory, router (+ budget), ollama,
│   │       │                       #   ollama_remote, openrouter, openai_embeddings, json_parse
│   │       ├── storage/            # file_import, note_store, git_sync
│   │       ├── ingestion/          # pdf, docx, html/web, image, audio, email parsers
│   │       └── plugins/            # api, manager
│   │
│   └── dashboard/                  # Next.js frontend
│       ├── package.json
│       ├── next.config.js
│       ├── tsconfig.json
│       ├── tailwind.config.ts
│       ├── postcss.config.js
│       ├── vitest.config.ts
│       └── src/
│           ├── app/
│           │   ├── layout.tsx          # Root layout
│           │   ├── globals.css
│           │   ├── login/page.tsx
│           │   ├── setup/              # First-run wizard
│           │   └── (app)/              # Authenticated routes
│           │       ├── layout.tsx
│           │       ├── page.tsx        # Dashboard home
│           │       ├── chat/page.tsx
│           │       ├── notes/page.tsx
│           │       ├── review/page.tsx
│           │       └── settings/       # page.tsx, TokensSection.tsx
│           │
│           ├── components/
│           │   ├── ErrorBoundary.tsx
│           │   ├── ReembedProvider.tsx     # WebSocket re-embed progress
│           │   ├── ReembedBanner.tsx
│           │   ├── SetupGate.tsx
│           │   ├── auth/                    # AuthGuard, LockScreen
│           │   ├── hud/                     # HUD UI kit (Hud*.tsx)
│           │   ├── layout/                  # RailLayout, MobileNav, NavItem, navConfig
│           │   └── markdown/                # MarkdownEditor, MarkdownPreview, MarkdownToolbar
│           │
│           ├── hooks/                  # useAuth, useHealth, useMediaQuery, useWebSocket
│           ├── lib/                    # api, chat-events, reembed-ws, cn, constants
│           ├── stores/                 # Zustand: auth, connection, neural, system
│           ├── types/                  # Shared TypeScript types
│           └── __tests__/             # setup.ts, smoke.test.tsx
```
