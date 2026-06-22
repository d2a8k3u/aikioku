
# Cursor Setup

Connect Cursor to your Aikioku.

## Configuration

### Project-level

Create `.cursor/mcp.json` in your project root:

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

### Global

Cursor Settings → MCP → Add Server, or edit the global MCP config.

## Verification

Open Cursor's AI chat and ask:

> search my notes for anything about TypeScript

Cursor will invoke `search_notes` and ground its answer in your knowledge base.

## Common Workflows

### Code + Knowledge Integration

```
> what do I know about PostgreSQL indexing strategies?
> create a note documenting our database schema decisions
```

### Graph-Aware Coding

```
> what entities are connected to "API" in my knowledge graph?
> find a path between "authentication" and "JWT"
```

## Token Scopes

| Scope  | Use Case |
|--------|----------|
| `read` | Code research, Q&A, graph exploration |
| `full` | Note creation from coding sessions |

## Troubleshooting

**Tools not appearing** — Verify `.cursor/mcp.json` is in the correct directory
and JSON is valid. Restart Cursor.

**"Unauthorized"** — Regenerate PAT in Settings → API access.
