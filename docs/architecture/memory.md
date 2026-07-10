# Memory

Agent memories are stored as **hidden notes** — notes with `source_type="hidden"`
that are fully indexed (entity extraction, embeddings, BM25) but never shown in
the UI's notes list. They are retrievable in chat and search exactly like regular
notes, but remain invisible to the user.

## Hidden Notes

- **source_type:** `"hidden"` — hidden from the notes UI, but fully indexed
- **Created via MCP:** the `create_memory` tool accepts free text and creates a
  hidden note with `source_type="hidden"`, frontmatter `memory_source`, and an
  auto-derived title
- **Full pipeline:** entity extraction, embeddings, knowledge-graph linking —
  everything a regular note gets
- **Retrieval:** hidden notes participate in hybrid search, RAG, and graph queries
  identically to visible notes

## Consolidation Worker

A nightly background worker that merges similar hidden notes to prevent
unbounded growth of agent-generated content.

- **Schedule:** nightly (runs once per 24h)
- **Opt-in:** controlled by the `auto_note_consolidation` config flag (off by
  default)
- **Budget-gated:** skips if the daily LLM budget is exhausted
- **Similarity detection:** finds pairs of hidden notes with cosine embedding
  similarity ≥ 0.85 **and** at least one shared knowledge-graph entity
- **Merge:** LLM merges the two notes into a single note, preserving all
  information from both originals
- **Cleanup:** deletes the originals after a successful merge
- **Re-indexing:** re-runs entity extraction + embedding on the merged note

Files: `src/processing/consolidation.py` (core logic),
`src/processing/consolidation_worker.py` (scheduling loop).

## Legacy Memory Migration

Notes with `source_type="memory"` (left over from the old SPO triple system)
are automatically migrated to `source_type="hidden"` on startup via
`src/scripts/migrate_memory_to_hidden.py`. No manual intervention needed.