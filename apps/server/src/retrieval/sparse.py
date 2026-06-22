"""SparseRetriever: BM25 keyword-based retrieval over .md files using Tantivy."""

from __future__ import annotations

import threading
from pathlib import Path

import tantivy

from src.retrieval.search_result import SearchResult


def _build_index(notes_dir: Path) -> tantivy.Index:
    """Build an in-memory Tantivy index from all .md files in notes_dir."""
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("note_id", stored=True, tokenizer_name="raw")
    schema_builder.add_text_field("body", stored=True, tokenizer_name="default")
    schema = schema_builder.build()
    index = tantivy.Index(schema)
    writer = index.writer()

    for md_file in sorted(notes_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        doc = tantivy.Document(note_id=str(md_file.stem), body=text)
        writer.add_document(doc)

    writer.commit()
    index.reload()
    return index


class SparseRetriever:
    """Retrieves notes by keyword matching with BM25 scoring via Tantivy."""

    def __init__(self, notes_dir: str) -> None:
        """Create a SparseRetriever.

        Args:
            notes_dir: Path to the directory containing .md note files.
        """
        self._notes_dir = Path(notes_dir)
        self._index: tantivy.Index | None = None
        self._dirty = True
        # Guards index (re)builds so concurrent searches (run via a
        # ThreadPoolExecutor in HybridFusion) never double-build or observe a
        # half-built index.
        self._lock = threading.Lock()

    def mark_dirty(self) -> None:
        """Flag the cached index for a rebuild on the next search.

        Call this when a note is added, changed, or removed so the next
        ``search`` reflects the new corpus. Cheap and thread-safe.
        """
        with self._lock:
            self._dirty = True

    # Alias: the API wiring and callers may use either spelling.
    invalidate = mark_dirty

    def _get_index(self) -> tantivy.Index:
        """Return the cached Tantivy index, rebuilding only if dirty.

        Double-checked under the lock so a single rebuild happens even when
        several searches race in different threads.
        """
        if self._index is not None and not self._dirty:
            return self._index
        with self._lock:
            if self._index is None or self._dirty:
                self._index = _build_index(self._notes_dir)
                self._dirty = False
            return self._index

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Search for notes matching the query.

        Reuses a cached Tantivy index across calls (built lazily on first use,
        rebuilt only after ``mark_dirty``) and returns the top results scored by
        Tantivy's BM25 implementation. Repeated searches are O(1) amortized.

        Args:
            query: The search query text.
            limit: Maximum number of results to return.

        Returns:
            A list of SearchResult objects ordered by descending score.
        """
        if not self._notes_dir.exists():
            return []

        index = self._get_index()
        searcher = index.searcher()

        try:
            parsed_query = index.parse_query(query, ["body", "note_id"])
        except Exception:
            # Tantivy throws on empty/invalid queries
            return []

        snippet_gen = tantivy.SnippetGenerator.create(searcher, parsed_query, index.schema, "body")

        results: list[SearchResult] = []
        search_result = searcher.search(parsed_query, limit)
        for score, doc_address in search_result.hits:
            doc = searcher.doc(doc_address)
            note_id = doc.get_first("note_id") or ""
            body = doc.get_first("body") or ""
            snippet_obj = snippet_gen.snippet_from_doc(doc)
            snippet = snippet_obj.to_html() if snippet_obj.fragment() else (body[:200] + "..." if len(body) > 200 else body)
            results.append(
                SearchResult(
                    note_id=str(note_id),
                    score=float(score),
                    source="sparse",
                    snippet=str(snippet),
                )
            )

        return results
