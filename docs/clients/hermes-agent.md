
# Hermes Agent Setup

Connect Hermes Agent (Nous Research) to your Aikioku.

## Configuration

Edit `~/.hermes/profiles/default/mcp.json` (or your active profile):

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

Restart Hermes or run `hermes mcp reload` to pick up the new config.

## Verification

In a Hermes session:

```
> search my notes for anything about knowledge graphs
```

Hermes will invoke `search_notes` or `hybrid_search` on your Aikioku.

## Common Workflows

### Research & Recall

```
> what do I know about vector databases?
> find connections between "ChromaDB" and "Kuzu" in my knowledge graph
```

### Note Management

```
> create a note titled "Hermes Session 2026-06-17" with our discussion
> summarize note abc123
```

### Spaced Repetition

```
> what cards are due for review?
> review card xyz789 — rating 4 (easy)
```

## Token Scopes

| Scope  | Use Case |
|--------|----------|
| `read` | Research, Q&A, graph exploration |
| `full` | Note creation, memory extraction, card generation |

## Troubleshooting

**Tools not appearing** — Run `hermes mcp list` to verify the server is registered.
Check that the profile path is correct.

**"Unauthorized"** — Regenerate PAT in Settings → API access.

**"Connection refused"** — Verify Aikioku is running:
```bash
curl http://localhost:8869/health
```
