"""EmbeddingStore: numpy-backed vector storage for note embeddings with semantic chunking."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Approximate characters per token (English ~4). 512 tokens ~ 2000 chars.
_CHUNK_CHAR_TARGET = 2000
_GROWTH_ROWS = 64
_F32 = np.dtype("float32")


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


class _CollectionState:
    """Mutable backing state for one collection.

    All public methods of EmbeddingStore route through here so that a
    delete_self() on one instance is immediately visible to every other
    instance that shares the same backing collection.
    """

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.lock = threading.Lock()
        self.deleted = False
        self.dimension = 0
        self.matrix = np.empty((0, 0), dtype=_F32)
        self.norms = np.empty(0, dtype=np.float64)
        self.doc_ids: list[str] = []
        self.note_ids: list[str] = []
        self.texts: list[str] = []
        self.metas: list[dict[str, Any]] = []
        self._allocated_rows = 0
        self._capacity = 0

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        vectors_path = os.path.join(self.directory, "vectors.npz")
        docs_path = os.path.join(self.directory, "docs.json")

        # Crash recovery: finals win; a lone .tmp (crash between write and
        # replace) is promoted, a .tmp beside a valid final is stale.
        for final_name, tmp_name in (("vectors.npz", "vectors.tmp"), ("docs.json", "docs.tmp")):
            final_path = os.path.join(self.directory, final_name)
            tmp_path = os.path.join(self.directory, tmp_name)
            if os.path.isfile(tmp_path):
                if os.path.isfile(final_path):
                    logger.warning("Discarding stale %s for collection dir %s", tmp_name, self.directory)
                    os.remove(tmp_path)
                else:
                    os.replace(tmp_path, final_path)

        if not os.path.isfile(vectors_path):
            return

        loaded = np.load(vectors_path)
        matrix = np.asarray(loaded["vectors"], dtype=_F32)
        with open(docs_path, encoding="utf-8") as f:
            docs = json.load(f)
        self._set_contents(
            matrix,
            [d["doc_id"] for d in docs],
            [d["note_id"] for d in docs],
            [d["text"] for d in docs],
            [d["metadata"] for d in docs],
        )

    def save(self) -> None:
        if self.deleted:
            return
        os.makedirs(self.directory, exist_ok=True)

        matrix = self.matrix
        if matrix.shape[0] > self._allocated_rows:
            matrix = matrix[: self._allocated_rows]

        vectors_tmp = os.path.join(self.directory, "vectors.tmp")
        docs_tmp = os.path.join(self.directory, "docs.tmp")
        try:
            with open(vectors_tmp, "wb") as f:
                np.savez(f, vectors=matrix)
            docs = [
                {
                    "doc_id": self.doc_ids[i],
                    "note_id": self.note_ids[i],
                    "text": self.texts[i],
                    "metadata": self.metas[i],
                }
                for i in range(self._allocated_rows)
            ]
            with open(docs_tmp, "w", encoding="utf-8") as f:
                json.dump(docs, f, ensure_ascii=False)
            os.replace(vectors_tmp, os.path.join(self.directory, "vectors.npz"))
            os.replace(docs_tmp, os.path.join(self.directory, "docs.json"))
        finally:
            for tmp in (vectors_tmp, docs_tmp):
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    # -- mutation helpers (caller holds lock) ------------------------------

    def _set_contents(
        self,
        matrix: npt.NDArray[np.float32],
        doc_ids: list[str],
        note_ids: list[str],
        texts: list[str],
        metas: list[dict[str, Any]],
    ) -> None:
        self.matrix = matrix
        self._allocated_rows = matrix.shape[0]
        self._capacity = matrix.shape[0]
        self._refresh_norms()
        self.doc_ids = list(doc_ids)
        self.note_ids = list(note_ids)
        self.texts = list(texts)
        self.metas = [dict(m) for m in metas]

    def append_rows(self, vectors: list[list[float]], rows: list[dict[str, Any]]) -> None:
        n_new = len(rows)
        if not n_new:
            return
        dim = len(vectors[0])
        total = self._allocated_rows + n_new
        if total > self._capacity:
            self._capacity = max(total, self._capacity + _GROWTH_ROWS)
            new_matrix = np.empty((self._capacity, dim), dtype=_F32)
            if self._allocated_rows:
                new_matrix[: self._allocated_rows] = self.matrix[: self._allocated_rows]
            self.matrix = new_matrix
        elif self._allocated_rows == 0:
            self.matrix = np.empty((self._capacity, dim), dtype=_F32)

        start = self._allocated_rows
        for offset, vector in enumerate(vectors):
            self.matrix[start + offset] = vector
        for row in rows:
            self.doc_ids.append(row["doc_id"])
            self.note_ids.append(row["note_id"])
            self.texts.append(row["text"])
            self.metas.append(dict(row["metadata"]))
        self._allocated_rows = total
        self._refresh_norms()

    def remove_note(self, note_id: str) -> int:
        keep = [i for i, nid in enumerate(self.note_ids) if nid != note_id]
        if len(keep) == self._allocated_rows:
            return 0
        removed = self._allocated_rows - len(keep)
        if keep:
            new_matrix = np.empty((len(keep), self.matrix.shape[1]), dtype=_F32)
            for dst, src in enumerate(keep):
                new_matrix[dst] = self.matrix[src]
        else:
            new_matrix = np.empty((0, self.matrix.shape[1]), dtype=_F32)
        self._set_contents(
            new_matrix,
            [self.doc_ids[i] for i in keep],
            [self.note_ids[i] for i in keep],
            [self.texts[i] for i in keep],
            [self.metas[i] for i in keep],
        )
        return removed

    def reset(self) -> None:
        self._set_contents(
            np.empty((0, self.dimension), dtype=_F32), [], [], [], []
        )

    # -- helpers -----------------------------------------------------------

    def _refresh_norms(self) -> None:
        used = self.matrix[: self._allocated_rows] if self._allocated_rows else self.matrix
        norms = np.linalg.norm(used, axis=1)
        self.norms = np.maximum(norms, np.finfo(np.float32).eps)


class EmbeddingStore:
    """Manages note embeddings with semantic text chunking.

    Backed by a numpy brute-force store: vectors in a float32 matrix on disk
    (``vectors.npz``) plus document metadata (``docs.json``), loaded eagerly
    into memory. Search is exact cosine similarity — no approximation.
    """

    _collections: ClassVar[dict[tuple[str, str], _CollectionState]] = {}

    def __init__(
        self,
        db_path: str,
        collection_name: str = "notes",
        dimension: int = 768,
    ) -> None:
        self._dimension = dimension
        self._collection_name = collection_name
        self._dir = os.path.join(db_path, collection_name)
        os.makedirs(self._dir, exist_ok=True)

        self._state = self._get_state()
        with self._state.lock:
            self._state.load()
            if self._state._allocated_rows and self._state.matrix.shape[1] != self._dimension:
                logger.warning(
                    "EmbeddingStore '%s' adopted existing dimension %s (requested %s); "
                    "a reembed will rebuild it under a new collection.",
                    self._collection_name,
                    self._state.matrix.shape[1],
                    self._dimension,
                )
                self._dimension = self._state.matrix.shape[1]
            self._state.dimension = self._dimension

    def _get_state(self) -> _CollectionState:
        key = (os.path.abspath(self._dir), self._collection_name)
        state = self._collections.get(key)
        if state is None:
            state = _CollectionState(self._dir)
            self._collections[key] = state
        return state

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def _matrix(self) -> npt.NDArray[np.float32]:
        """Current backing matrix (design §4.3). Writers replace, never mutate."""
        return self._state.matrix

    def delete_self(self) -> None:
        """Delete this store's underlying collection (post-swap cleanup by reembed)."""
        with self._state.lock:
            self._state.deleted = True
            self._state.reset()
            if os.path.isdir(self._dir):
                shutil.rmtree(self._dir, ignore_errors=True)
        logger.info("Deleted embedding collection '%s' at %s", self._collection_name, self._dir)

    def add(self, note_id: str, text: str, embedding: list[float]) -> None:
        """Store a note's text (chunked) and its embedding.

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
            {"note_id": note_id, "chunk_index": i, "text": chunk} for i, chunk in enumerate(chunks)
        ]
        rows = [
            {
                "doc_id": doc_id,
                "note_id": note_id,
                "text": chunk,
                "metadata": metadata,
            }
            for doc_id, chunk, metadata in zip(ids, chunks, metadatas)
        ]
        self._append(embeddings, rows)

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
        self._append(
            [embedding],
            [{"doc_id": doc_id, "note_id": str(meta.get("note_id", "")), "text": text, "metadata": meta}],
        )

    def _append(self, embeddings: list[list[float]], rows: list[dict[str, Any]]) -> None:
        with self._state.lock:
            if self._state.deleted:
                logger.warning("Store '%s' was deleted; dropping add", self._collection_name)
                return
            self._state.append_rows(embeddings, rows)
            self._state.save()

    def search(self, query_embedding: list[float], limit: int = 20) -> list[dict[str, Any]]:
        """Search the store by embedding similarity.

        Groups results by ``note_id`` and takes the **maximum** similarity
        across all sub-windows belonging to the same note.  This way a note
        matches if *any* of its sub-windows is relevant to the query.

        Returns a list of dicts (one per note) ordered by descending score,
        each with keys: ``note_id``, ``text`` (best sub-window), ``score``.
        """
        with self._state.lock:
            matrix = self._state.matrix[: self._state._allocated_rows]
            norms = self._state.norms
            note_ids = list(self._state.note_ids)
            texts = list(self._state.texts)

        if matrix.shape[0] == 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float64)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return []
        q = q / q_norm

        sims = matrix @ q / norms

        best: dict[str, tuple[float, str]] = {}
        for note_id, text, sim in zip(note_ids, texts, sims):
            score = float(sim)
            if score <= 0:
                continue
            if note_id not in best or score > best[note_id][0]:
                best[note_id] = (score, text)

        sorted_notes = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
        results: list[dict[str, Any]] = []
        for note_id, (score, text) in sorted_notes[:limit]:
            results.append(
                {
                    "note_id": note_id,
                    "text": text,
                    "score": max(score, 0.0),
                }
            )
        return results

    def delete(self, note_id: str) -> None:
        """Delete all chunks belonging to a note."""
        with self._state.lock:
            self._state.remove_note(note_id)
            self._state.save()

    def count(self) -> int:
        with self._state.lock:
            return self._state._allocated_rows

    def get_embeddings(self, note_ids: list[str]) -> dict[str, list[float]]:
        """Return embeddings for specific note IDs.

        Args:
            note_ids: List of note IDs to fetch.

        Returns:
            Dict mapping note_id to embedding vector.
        """
        with self._state.lock:
            matrix = self._state.matrix[: self._state._allocated_rows]
            note_list = self._state.note_ids

        wanted = set(note_ids)
        result: dict[str, list[float]] = {}
        if matrix.shape[0] == 0:
            return result
        seen: set[str] = set()
        for note_id, vector in zip(note_list, matrix):
            if note_id in wanted and note_id not in seen:
                result[note_id] = [float(v) for v in vector]
                seen.add(note_id)
        return result
