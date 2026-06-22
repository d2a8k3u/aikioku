
# Claude Desktop Setup

Connect Claude Desktop to your Aikioku.

## Configuration

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

Restart Claude Desktop after saving.

## Verification

Open Claude Desktop and ask:

> search my notes for anything about machine learning

Claude will invoke `search_notes` on your Aikioku and return results.

## Token Scopes

| Scope  | Use Case |
|--------|----------|
| `read` | Research, Q&A, graph exploration |
| `full` | Note creation, flashcard generation, import |

Use `read` for most sessions. Use `full` when you want Claude to actively
write into your brain.

## Troubleshooting

**Tools not appearing** — Check that the config path is correct and JSON is
valid. Restart Claude Desktop completely.

**"Unauthorized"** — Regenerate your PAT in Settings → API access.

**"Connection refused"** — Verify Aikioku is running on port 8869.
