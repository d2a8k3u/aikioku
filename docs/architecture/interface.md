# Layer 7: Interface

Web UI and API surface for interacting with Aikioku.

## Web UI (Next.js)

### Pages

| Route | Description |
|-------|-------------|
| `/` | Dashboard home |
| `/chat` | Conversational RAG interface with streaming |
| `/notes` | Note list with search, filtering, and markdown editing |
| `/review` | Spaced-repetition card review |
| `/settings` | LLM (incl. daily budget), embedding, tokens, and secrets management |
| `/login` | Authentication |
| `/setup` | First-run setup wizard |

### Components

- **MarkdownEditor:** textarea-based markdown editor with a formatting toolbar (`MarkdownToolbar`) and live preview (`MarkdownPreview`)
- **RailLayout / MobileNav / NavItem:** application navigation
- **HUD kit:** reusable UI primitives (`HudPanel`, `HudButton`, `HudModal`, `HudToast`, …)
- **ErrorBoundary:** graceful error handling
- **ReembedProvider / ReembedBanner:** WebSocket-driven re-embedding progress
- **BudgetProvider / BudgetBanner:** WebSocket-driven daily-budget state — warning near the limit, and a paused banner (with queued-item count and reset time) when the cap is reached
- **AuthGuard / LockScreen / SetupGate:** authentication and first-run gating

### Tech Stack

- Next.js 16 (App Router)
- React 18
- Tailwind CSS v4
- Zustand (state)
- TanStack React Query (data fetching)
- Three.js (ambient canvas visuals on the home screen)

## API Surface

### REST API
95 endpoints across 29 routers cover all functionality. See [REST API](../api/rest.md).

### MCP Server
41 typed tools for external AI clients. See [MCP Server](../api/mcp.md).

### WebSocket
Real-time events at `/ws/events`:

- Re-embedding progress
- Chat streaming
- Entity extraction status
