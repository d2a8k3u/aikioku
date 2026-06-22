
# Claude Code Setup

Connect Claude Code to your Aikioku so it can search notes, explore the
knowledge graph, and manage your PKM directly from the terminal.

## Prerequisites

- Aikioku running: backend accessible at `http://localhost:8869`
- A Personal Access Token (PAT) from **Settings → API access**
- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)

## Configuration

### Project-level (recommended)

Create `.mcp.json` in your project root:

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

### Global (all projects)

`~/.claude/.mcp.json`:

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

### Per-session

```bash
claude --mcp-config '{"aikioku":{"type":"streamableHttp","url":"http://localhost:8869/mcp","headers":{"Authorization":"Bearer sbk_your_token_here"}}}'
```

## Verification

Start Claude Code and check tools:

```
> claude
> /mcp
```

You should see `aikioku` listed with all 41 tools.

## Common Workflows

### Research

```
> what do I know about distributed systems? search my brain
```

Uses `ask` (RAG) or `hybrid_search`.

### Graph Exploration

```
> show me everything connected to "Python" in my knowledge graph
> find a path between "Django" and "PostgreSQL"
```

Uses `get_entity_subgraph` and `graph_paths`.

### Note Creation (requires `full` scope)

```
> create a note summarizing what we just discussed about the API refactor
```

Uses `create_note`.

### Spaced Repetition

```
> what flashcards are due for review?
> review card xyz789 — rating 3 (good)
```

Uses `list_due_cards` and `review_card`.

## Troubleshooting

**"No MCP servers found"** — `.mcp.json` must be in the directory where you
launched `claude`, or use `~/.claude/.mcp.json`.

**"Unauthorized"** — Verify PAT is correct. Generate a new one if needed.

**"Connection refused"** — Ensure Aikioku is running:
```bash
docker compose ps
curl http://localhost:8869/health
```

**"Write tool rejected"** — Token has `read` scope. Generate a `full` scope token.

## Remote Aikioku

Replace `localhost:8869` with your remote host behind a reverse proxy:

```json
{
  "mcpServers": {
    "aikioku": {
      "type": "streamableHttp",
      "url": "https://brain.example.com/mcp",
      "headers": {
        "Authorization": "Bearer sbk_your_token_here"
      }
    }
  }
}
```

## Security

- Never commit `.mcp.json` with a real token
- Use environment variables or a secrets manager for the token value
- PATs are SHA-256 hashed — a leaked DB cannot reveal usable tokens
