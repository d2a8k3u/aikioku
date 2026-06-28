"""NoteStore: file-based storage for Notes using Markdown with YAML frontmatter.

The markdown files on disk are the single source of truth (Git-backed). To keep
list/count/tag/path operations from scanning and YAML-parsing the entire corpus
on every request, NoteStore maintains a persistent SQLite index of note
*metadata* (id, path, title, tags, timestamps, file mtime) alongside the files.

The index is a cache: if it ever disagrees with disk, disk wins and ``reindex``
rebuilds it. It is kept in sync incrementally by create/update/delete, and is
lazily rebuilt on first use when missing or out of sync.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, cast

import yaml

from src.models.note import Note

# A note_index row: (id, path, title, tags_json, source_type, created, modified, mtime).
_IndexRow = tuple[str, str, str, str, str, str, str, float]

# source_type values whose notes are fully processed + retrievable (graph,
# embeddings, BM25) but hidden from the user-facing list/search/count — e.g.
# agent-written memories created over MCP.
HIDDEN_SOURCE_TYPES: frozenset[str] = frozenset({"hidden"})


def _hidden_exclusion() -> tuple[str, tuple[str, ...]]:
    """SQL clause + params that exclude hidden source_types from a note_index query."""
    placeholders = ",".join("?" for _ in HIDDEN_SOURCE_TYPES)
    return f"COALESCE(source_type, 'note') NOT IN ({placeholders})", tuple(HIDDEN_SOURCE_TYPES)


def _note_to_markdown(note: Note) -> str:
    """Serialize a Note to a Markdown string with YAML frontmatter."""
    frontmatter = {
        "id": note.id,
        "title": note.title,
        "tags": note.frontmatter.get("tags", []),
        "aliases": note.frontmatter.get("aliases", []),
        "links": note.links,
        "path": note.path,
        "source_type": note.source_type,
        "created": note.created.isoformat(),
        "modified": note.modified.isoformat(),
    }
    yaml_block = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()
    parts = ["---", yaml_block, "---", "", note.content]
    return "\n".join(parts)


def _markdown_to_note(content: str, note_path: str) -> Note:
    """Parse a Markdown file with YAML frontmatter into a Note."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        raise ValueError(f"Invalid note file format: missing YAML frontmatter in {note_path}")

    yaml_text = match.group(1)
    body = match.group(2)
    metadata = yaml.safe_load(yaml_text)

    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid frontmatter in {note_path}")

    frontmatter = {}
    if "tags" in metadata:
        frontmatter["tags"] = metadata["tags"] or []
    if "aliases" in metadata:
        frontmatter["aliases"] = metadata["aliases"] or []

    created = metadata.get("created")
    if isinstance(created, str):
        created = datetime.fromisoformat(created)
    elif not isinstance(created, datetime):
        created = datetime.utcnow()

    modified = metadata.get("modified")
    if isinstance(modified, str):
        modified = datetime.fromisoformat(modified)
    elif not isinstance(modified, datetime):
        modified = datetime.utcnow()

    return Note(
        id=metadata.get("id", str(uuid.uuid4())),
        title=metadata.get("title", "Untitled"),
        content=body,
        frontmatter=frontmatter,
        links=metadata.get("links", []),
        path=metadata.get("path", note_path),
        source_type=metadata.get("source_type", "note"),
        created=created,
        modified=modified,
    )


def _atomic_write_text(file_path: Path, content: str) -> None:
    """Write text so the destination is never left partial or 0-byte.

    Writes to a temp file in the same directory, fsyncs it, then atomically
    renames over the target. An interrupted write leaves the old file intact
    (or, for a new note, nothing) — never a truncated one, which would break a
    full-corpus parser such as the reembed scan.
    """
    tmp_path = file_path.parent / f".{file_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class NoteStore:
    """File-based storage for Notes using Markdown with YAML frontmatter.

    Each note is stored as a single .md file named by its UUID. A SQLite
    metadata index alongside the files accelerates list/count/tag/path
    operations so they do not scan the whole corpus.
    """

    def __init__(self, notes_dir: str, index_db_path: Optional[str] = None) -> None:
        """Initialize NoteStore.

        Args:
            notes_dir: Directory holding the markdown notes (source of truth).
            index_db_path: Optional path for the SQLite metadata index. Defaults
                to ``<notes_dir>/.note_index.db`` so it is isolated per notes_dir
                (and per test, which sets a tmp notes_dir).
        """
        self.notes_dir = notes_dir
        os.makedirs(notes_dir, exist_ok=True)
        self.index_db_path = index_db_path or os.path.join(notes_dir, ".note_index.db")
        self._index_synced = False
        self._ensure_table()

    # ------------------------------------------------------------------ index

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.index_db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS note_index (
                    id TEXT PRIMARY KEY,
                    path TEXT,
                    title TEXT,
                    tags TEXT,
                    source_type TEXT DEFAULT 'note',
                    created TEXT,
                    modified TEXT,
                    mtime REAL
                )
                """
            )
            # Migration: add source_type column if table existed before this field
            try:
                conn.execute("ALTER TABLE note_index ADD COLUMN source_type TEXT DEFAULT 'note'")
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_note_index_modified ON note_index (modified DESC)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_note_index_path ON note_index (path)")
            conn.commit()
        finally:
            conn.close()

    def _file_path(self, note_id: str) -> Path:
        """Return the file path for a given note ID."""
        return Path(self.notes_dir) / f"{note_id}.md"

    def _md_files(self) -> list[Path]:
        return sorted(Path(self.notes_dir).glob("*.md"))

    def _row_from_note(self, note: Note) -> _IndexRow:
        try:
            mtime = self._file_path(note.id).stat().st_mtime
        except OSError:
            mtime = 0.0
        tags = note.frontmatter.get("tags", []) or []
        return (
            note.id,
            note.path,
            note.title,
            json.dumps([str(t) for t in tags]),
            note.source_type,
            note.created.isoformat(),
            note.modified.isoformat(),
            mtime,
        )

    def _upsert_index(self, note: Note) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO note_index (id, path, title, tags, source_type, created, modified, mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    path = excluded.path,
                    title = excluded.title,
                    tags = excluded.tags,
                    source_type = excluded.source_type,
                    created = excluded.created,
                    modified = excluded.modified,
                    mtime = excluded.mtime
                """,
                self._row_from_note(note),
            )
            conn.commit()
        finally:
            conn.close()

    def _delete_index(self, note_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM note_index WHERE id = ?", (note_id,))
            conn.commit()
        finally:
            conn.close()

    def _index_count(self) -> int:
        conn = self._connect()
        try:
            return cast(int, conn.execute("SELECT COUNT(*) FROM note_index").fetchone()[0])
        finally:
            conn.close()

    def _ensure_synced(self) -> None:
        """Lazily rebuild the index if it is missing or out of sync with disk.

        Cheap check on the hot path: compare the index row count to the number
        of markdown files. If they differ, the index is stale (fresh deploy,
        externally edited files, manual git pull) and we rebuild from disk.
        """
        if self._index_synced:
            return
        disk_count = len(self._md_files())
        if self._index_count() != disk_count:
            self.reindex()
        self._index_synced = True

    def reindex(self) -> int:
        """Rebuild the metadata index from the markdown files on disk.

        Idempotent and authoritative: drops stale rows, (re)inserts a row for
        every .md file. Returns the number of notes indexed. Files that fail to
        parse are skipped (they are not valid notes) rather than aborting.
        """
        self._ensure_table()
        disk_ids: set[str] = set()
        rows: list[_IndexRow] = []
        for md_file in self._md_files():
            try:
                content = md_file.read_text(encoding="utf-8")
                note = _markdown_to_note(content, str(md_file))
            except (ValueError, OSError):
                continue
            disk_ids.add(note.id)
            rows.append(self._row_from_note(note))

        conn = self._connect()
        try:
            existing = {r[0] for r in conn.execute("SELECT id FROM note_index").fetchall()}
            stale = existing - disk_ids
            if stale:
                conn.executemany("DELETE FROM note_index WHERE id = ?", [(i,) for i in stale])
            conn.executemany(
                """
                INSERT INTO note_index (id, path, title, tags, source_type, created, modified, mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    path = excluded.path,
                    title = excluded.title,
                    tags = excluded.tags,
                    source_type = excluded.source_type,
                    created = excluded.created,
                    modified = excluded.modified,
                    mtime = excluded.mtime
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        self._index_synced = True
        return len(rows)

    # ------------------------------------------------------------------ CRUD

    def create(self, note: Note) -> Note:
        """Save a note as a Markdown file with YAML frontmatter. Returns the saved Note."""
        file_path = self._file_path(note.id)
        content = _note_to_markdown(note)
        _atomic_write_text(file_path, content)
        self._upsert_index(note)
        return note

    def get(self, note_id: str) -> Optional[Note]:
        """Retrieve a note by its ID. Returns None if not found."""
        file_path = self._file_path(note_id)
        if not file_path.exists():
            return None
        content = file_path.read_text(encoding="utf-8")
        return _markdown_to_note(content, str(file_path))

    def update(self, note: Note) -> Note:
        """Update an existing note file. Updates the modified timestamp. Returns the updated Note."""
        if note.modified.tzinfo is not None:
            note.modified = datetime.now(tz=note.modified.tzinfo)
        else:
            note.modified = datetime.utcnow()
        file_path = self._file_path(note.id)
        content = _note_to_markdown(note)
        _atomic_write_text(file_path, content)
        self._upsert_index(note)
        return note

    def delete(self, note_id: str) -> bool:
        """Delete a note file by ID. Returns True if deleted, False if not found."""
        file_path = self._file_path(note_id)
        if not file_path.exists():
            self._delete_index(note_id)
            return False
        file_path.unlink()
        self._delete_index(note_id)
        return True

    # ------------------------------------------------------------------ reads

    def count(self) -> int:
        """Return the number of user-facing notes (excludes hidden source_types)."""
        self._ensure_synced()
        clause, hidden_params = _hidden_exclusion()
        conn = self._connect()
        try:
            return cast(
                int,
                conn.execute(
                    f"SELECT COUNT(*) FROM note_index WHERE {clause}", hidden_params
                ).fetchone()[0],
            )
        finally:
            conn.close()

    def _read_note_file(self, note_id: str, fallback_path: str) -> Optional[Note]:
        """Read and parse a single note file by id, returning None if missing/invalid."""
        file_path = self._file_path(note_id)
        if not file_path.exists():
            return None
        try:
            content = file_path.read_text(encoding="utf-8")
            return _markdown_to_note(content, str(file_path))
        except (ValueError, OSError):
            return None

    def list(self, skip: int = 0, limit: int = 50, source_type: str | None = None) -> List[Note]:
        """Return a page of notes ordered by ``modified`` DESC.

        Reads ONLY the page's files (at most ``limit``), using the index to pick
        which ids to read. This is the path the notes API list endpoint uses, so
        a list request never scans the whole corpus.

        When ``source_type`` is provided, only notes with that source_type are
        returned (so callers can fetch hidden notes explicitly via
        ``source_type="hidden"``). With no ``source_type``, hidden notes are
        excluded from the user-facing listing.
        """
        self._ensure_synced()
        conn = self._connect()
        try:
            if source_type:
                rows = conn.execute(
                    "SELECT id, path FROM note_index "
                    "WHERE source_type = ? "
                    "ORDER BY modified DESC, id ASC LIMIT ? OFFSET ?",
                    (source_type, limit, skip),
                ).fetchall()
            else:
                clause, hidden_params = _hidden_exclusion()
                rows = conn.execute(
                    "SELECT id, path FROM note_index "
                    f"WHERE {clause} "
                    "ORDER BY modified DESC, id ASC LIMIT ? OFFSET ?",
                    (*hidden_params, limit, skip),
                ).fetchall()
        finally:
            conn.close()

        notes: list[Note] = []
        for row in rows:
            note = self._read_note_file(row["id"], row["path"])
            if note is not None:
                notes.append(note)
        return notes

    def list_all(self) -> List[Note]:
        """List all notes in the storage directory (full scan, for back-fill/compat).

        Skips files that fail to parse — e.g. an empty/truncated .md left by an
        interrupted write — so one corrupt file never aborts a full-corpus
        consumer such as the background reembed. Mirrors ``_read_note_file``.
        """
        notes = []
        for md_file in self._md_files():
            try:
                content = md_file.read_text(encoding="utf-8")
                note = _markdown_to_note(content, str(md_file))
            except (ValueError, OSError):
                continue
            notes.append(note)
        return notes

    def search(self, query: str) -> List[Note]:
        """Full-text search across note titles and content (case-insensitive).

        Hidden notes (e.g. agent-written memories) are excluded — they are
        retrievable in chat but not surfaced in the user-facing note search.
        """
        notes = [n for n in self.list_all() if n.source_type not in HIDDEN_SOURCE_TYPES]
        if not query:
            return notes

        query_lower = query.lower()
        return [
            note
            for note in notes
            if query_lower in note.title.lower() or query_lower in note.content.lower()
        ]

    def get_by_tag(self, tag: str) -> List[Note]:
        """Return all notes that have the given tag (case-insensitive).

        Uses the index to find matching ids, then reads ONLY those files.
        """
        self._ensure_synced()
        tag_lower = tag.lower()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, path, tags, source_type FROM note_index ORDER BY modified DESC, id ASC"
            ).fetchall()
        finally:
            conn.close()

        results: list[Note] = []
        for row in rows:
            if (row["source_type"] or "note") in HIDDEN_SOURCE_TYPES:
                continue
            try:
                tags = json.loads(row["tags"] or "[]")
            except (ValueError, TypeError):
                tags = []
            if any(str(t).lower() == tag_lower for t in tags):
                note = self._read_note_file(row["id"], row["path"])
                if note is not None:
                    results.append(note)
        return results

    def get_by_path(self, path: str) -> Optional[Note]:
        """Retrieve a note by its file path. Returns None if not found.

        Index lookup by path -> read that one file.
        """
        self._ensure_synced()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, path FROM note_index WHERE path = ? LIMIT 1", (path,)
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        return self._read_note_file(row["id"], row["path"])
