
# Layer 1: Ingestion

Accepts and normalizes content from multiple sources into the Aikioku.

## Supported Formats

| Format | Parser | Technology |
|--------|--------|------------|
| Markdown | Native | — |
| PDF | `pdf_parser.py` | PyMuPDF |
| DOCX | `docx_parser.py` | python-docx |
| HTML | `web_parser.py` | readability-lxml |
| Images | `image_parser.py` | OCR (LLM vision) |
| Audio | `audio_parser.py` | Transcription (LLM) |
| Email | `email_parser.py` | Raw text extraction |

## Entry Points

- **REST API:** `POST /api/notes` with `content` and `source_type`
- **MCP Tools:** `create_note`, `import_markdown`
- **Format imports:** `POST /api/import/{pdf,docx,web,image,audio,email,obsidian,logseq,notion,roam,bulk}`

## Pipeline

1. Receive raw content + source type
2. Route to appropriate parser
3. Extract text, metadata, structure
4. Normalize to `Note` model
5. Trigger downstream: entity extraction, embedding, sparse indexing

## Budget-gated deferral

The downstream LLM steps (entity extraction, embedding, memory extraction) pass
through a budget gate (`src/processing/budget_gate.py`). When the daily LLM
budget is exhausted, the note/memory is still **persisted** (the markdown file is
the source of truth), but its LLM processing is enqueued in a durable
`pending_llm_work` table instead of running. A background drain worker re-runs the
queue once the budget resets (00:00 UTC) or is raised in Settings — so ingestion
never drops content, it only defers the expensive part. Sparse (BM25) indexing
is not LLM-backed and always runs immediately.

## Note Model

```python
class Note:
    id: str
    title: str
    content: str          # Markdown body
    source_type: str      # "markdown", "pdf", "web", etc.
    path: str | None      # Filesystem path
    tags: list[str]
    created_at: datetime
    updated_at: datetime
```
