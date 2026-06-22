"""EmbeddingStore: ChromaDB-backed vector storage for note embeddings with semantic chunking."""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

# Approximate characters per token (English ~4). 512 tokens ~ 2000 chars.
_CHUNK_CHAR_TARGET = 2000


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex on punctuation."""
    # Keep the delimiter attached to each sentence for reconstruction
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _semantic_chunks(text: str, target_chars: int = _CHUNK_CHAR_TARGET) -> list[str]:
    """Split text into chunks at sentence boundaries, targeting ~target_chars per chunk."""
    sentences = _split_sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sent_len = len(sentence)
        if current_len + sent_len > target_chars and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sent_len
        else:
            current.append(sentence)
            current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks if chunks else [text]


class EmbeddingStore:
    """Manages note embeddings in ChromaDB with semantic text chunking."""

    def __init__(
        self,
        db_path: str,
        collection_name: str = "notes",
        dimension: int = 768,
    ) -> None:
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._client = chromadb.PersistentClient(path=db_path)
        self._dimension = dimension
        self._collection_name = collection_name
        self._coll = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        """Return the collection, creating it if absent.

        NEVER wipes on a dimension mismatch: an existing collection is adopted
        as-is (its stored dimension wins). A dimension/model change is handled by
        reembedding into a NEW fingerprint-named collection and atomically
        swapping (see ``src.knowledge.reembed``), so a fixed name never needs a
        destructive recreate.
        """
        try:
            coll = self._client.get_collection(name=self._collection_name)
        except Exception:
            logger.info(
                "Creating new collection '%s' with dimension %s.",
                self._collection_name,
                self._dimension,
            )
            return self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine", "embedding_dimension": self._dimension},
            )
        existing_dim = coll.metadata.get("embedding_dimension") if coll.metadata else None
        if existing_dim is not None and int(existing_dim) != self._dimension:
            logger.warning(
                "EmbeddingStore '%s' adopted existing dimension %s (requested %s); "
                "a reembed will rebuild it under a new collection.",
                self._collection_name,
                existing_dim,
                self._dimension,
            )
            self._dimension = int(existing_dim)
        return coll

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def delete_self(self) -> None:
        """Delete this store's underlying collection (post-swap cleanup by reembed)."""
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            logger.warning(
                "delete_collection failed for '%s'", self._collection_name, exc_info=True
            )

    def add(self, note_id: str, text: str, embedding: list[float]) -> None:
        """Store a note's text (chunked) and its embedding in ChromaDB.

        Each semantic chunk is stored as a separate document sharing the same
        embedding.  On search we deduplicate by note_id and return the top
        chunk per note as the snippet.

        .. deprecated::
            Prefer :meth:`add_document` for sub-window embeddings.  This
            method remains for backward compatibility with callers that
            embed the full note text as a single vector.
        """
        chunks = _semantic_chunks(text)
        if not chunks:
            chunks = [text]
        ids = [f"{note_id}#{i}" for i in range(len(chunks))]
        embeddings = [embedding] * len(chunks)
        metadatas: list[dict[str, Any]] = [
            {"note_id": note_id, "chunk_index": i, "text": chunk}
            for i, chunk in enumerate(chunks)
        ]
        self._coll.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def add_document(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a single document (sub-window) with its own embedding.

        Unlike :meth:`add`, this does NOT apply semantic chunking — the
        caller is responsible for splitting the text into sub-windows.
        The ``metadata`` dict must include ``note_id`` so the document
        can be grouped and deleted by note.

        Args:
            doc_id: Unique identifier for this document (e.g. ``note-1#c0#sw3``).
            text: The sub-window text.
            embedding: The embedding vector for this sub-window.
            metadata: Dict with at least ``note_id``; may also include
                       ``chunk_index``, ``sub_window_index``, ``window_start``,
                       ``window_end``.
        """
        meta = dict(metadata or {})
        meta["text"] = text
        self._coll.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def search(self, query_embedding: list[float], limit: int = 20) -> list[dict]:
        """Search the store by embedding similarity.

        Groups results by ``note_id`` and takes the **maximum** similarity
        across all sub-windows belonging to the same note.  This way a note
        matches if *any* of its sub-windows is relevant to the query.

        Returns a list of dicts (one per note) ordered by descending score,
        each with keys: ``note_id``, ``text`` (best sub-window), ``score``.
        """
        # Over-fetch heavily because each note may have many sub-windows.
        n_results = max(limit * 8, 40)
        raw = self._coll.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        ids_list: list[str] = raw["ids"][0]
        dists_list: list[float] = raw["distances"][0]
        metas_list: list[dict[str, Any]] = raw["metadatas"][0]

        # Group by note_id, keeping max score and best text.
        best: dict[str, tuple[float, str]] = {}  # note_id -> (max_score, best_text)
        for doc_id, distance, meta in zip(ids_list, dists_list, metas_list):
            note_id = meta.get("note_id", "")
            if not note_id:
                continue
            score = 1.0 - float(distance)
            if score <= 0:
                continue
            if note_id not in best or score > best[note_id][0]:
                best[note_id] = (score, meta.get("text", ""))

        # Sort by max score descending, then take top `limit`.
        sorted_notes = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
        results: list[dict] = []
        for note_id, (score, text) in sorted_notes[:limit]:
            results.append({
                "note_id": note_id,
                "text": text,
                "score": max(score, 0.0),
            })
        return results

    def delete(self, note_id: str) -> None:
        """Delete all chunks belonging to a note."""
        self._coll.delete(where={"note_id": note_id})

    def count(self) -> int:
        return self._coll.count()

    def get_embeddings(self, note_ids: list[str]) -> dict[str, list[float]]:
        """Return embeddings for specific note IDs.

        Args:
            note_ids: List of note IDs to fetch.

        Returns:
            Dict mapping note_id to embedding vector.
        """
        result: dict[str, list[float]] = {}
        for note_id in note_ids:
            # Retrieve one chunk per note to get its embedding
            raw = self._coll.get(
                where={"note_id": note_id},
                limit=1,
                include=["embeddings"],
            )
            if raw["ids"]:
                result[note_id] = raw["embeddings"][0]
        return result
