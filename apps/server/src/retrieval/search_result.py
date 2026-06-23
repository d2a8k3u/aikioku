"""SearchResult dataclass for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single retrieval result from any retriever.

    Attributes:
        note_id: Identifier of the matching note (or ``entity:<id>`` for synthetic
            entity-backed results that have no source note).
        score: Relevance score (higher is better).
        source: Which retriever produced this result ("dense", "sparse", "graph").
        snippet: A short text snippet for display.
        metadata: Additional metadata about the result.
        source_type: Provenance of the result — ``"note"`` for note-backed results,
            ``"entity"`` for synthetic results derived from an orphaned knowledge-graph
            entity with no source notes.
    """

    note_id: str
    score: float
    source: str  # "dense" | "sparse" | "graph"
    snippet: str = ""
    metadata: dict = field(default_factory=dict)
    source_type: str = "note"
