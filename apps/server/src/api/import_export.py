"""Import/Export API endpoints for Aikioku."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.auth import require_auth
from src.config import settings
from src.limiter import limiter
from src.models.note import Note
from src.storage.file_import import parse_markdown_file
from src.storage.note_store import NoteStore, _note_to_markdown

router = APIRouter(tags=["import-export"])

# Module-level store (same pattern as notes.py / search.py)
_note_store: NoteStore | None = None


def get_note_store() -> NoteStore:
    global _note_store
    if _note_store is None:
        from src.config import settings

        _note_store = NoteStore(settings.notes_dir)
    return _note_store


def _extract_and_store_entities(request: Request, note: Note) -> None:
    """Schedule budget-gated processing (entity extraction + embedding) for an
    imported note. Fire-and-forget; deferred to a queue when the daily LLM
    budget is exhausted and drained after it resets or is raised."""
    import asyncio
    import logging

    logger = logging.getLogger(__name__)

    from src.processing.budget_gate import WORK_NOTE_PROCESSING, gated

    async def _do_processing():
        try:
            await gated(request.app, WORK_NOTE_PROCESSING, note.id, {"note_id": note.id})
        except Exception as exc:
            logger.error(
                "Note processing failed for imported note %s: %s", note.id, exc, exc_info=True
            )

    asyncio.ensure_future(_do_processing())


def _store_note_embeddings(request: Request, note: Note) -> None:
    """No-op: an imported note's embeddings are produced inside the budget-gated
    processing task scheduled by :func:`_extract_and_store_entities`."""
    return None


def _process_imported_note(request: Request, note: Note) -> Note:
    """Save a note and trigger extraction + embedding."""
    import asyncio

    from src.cache.semantic_cache import cache_invalidate

    store = get_note_store()
    created = store.create(note)
    _extract_and_store_entities(request, created)
    _store_note_embeddings(request, created)
    # Invalidate semantic cache — new notes may change correct answers
    asyncio.ensure_future(cache_invalidate())
    return created


# ---------------------------------------------------------------------------
# Import endpoints
# ---------------------------------------------------------------------------


@router.post("/api/import/markdown")
@limiter.limit("20/minute")
async def import_markdown(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(require_auth),
) -> Note:
    """Accept a single markdown file and create a note."""
    content = await file.read()
    text = content.decode("utf-8")
    filename = file.filename or "imported.md"
    note = parse_markdown_file(text, filename)
    return _process_imported_note(request, note)


@router.post("/api/import/obsidian")
@limiter.limit("20/minute")
async def import_obsidian(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(require_auth),
) -> dict:
    """Accept an Obsidian vault zip, extract all .md files, and create notes."""
    content = await file.read()
    buf = io.BytesIO(content)
    if not zipfile.is_zipfile(buf):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip archive")

    buf.seek(0)
    store = get_note_store()
    imported_count = 0

    with zipfile.ZipFile(buf, "r") as zf:
        md_names = [n for n in zf.namelist() if n.endswith(".md") and ".obsidian" not in n]
        for name in md_names:
            try:
                text = zf.read(name).decode("utf-8")
                note = parse_markdown_file(text, name)
                created = store.create(note)
                _extract_and_store_entities(request, created)
                _store_note_embeddings(request, created)
                imported_count += 1
            except Exception:
                continue

    # Invalidate semantic cache after bulk import
    import asyncio as _asyncio
    from src.cache.semantic_cache import cache_invalidate as _cache_invalidate
    _asyncio.ensure_future(_cache_invalidate())
    return {"imported_count": imported_count}


@router.post("/api/import/bulk")
@limiter.limit("20/minute")
async def import_bulk(
    request: Request,
    files: list[UploadFile] = File(...),
    user=Depends(require_auth),
) -> dict:
    """Accept multiple markdown files and create notes for each."""
    store = get_note_store()
    imported_count = 0

    for upload_file in files:
        try:
            content = await upload_file.read()
            text = content.decode("utf-8")
            filename = upload_file.filename or f"imported-{imported_count}.md"
            note = parse_markdown_file(text, filename)
            created = store.create(note)
            _extract_and_store_entities(request, created)
            _store_note_embeddings(request, created)
            imported_count += 1
        except Exception:
            continue

    # Invalidate semantic cache after bulk import
    import asyncio as _asyncio
    from src.cache.semantic_cache import cache_invalidate as _cache_invalidate
    _asyncio.ensure_future(_cache_invalidate())
    return {"imported_count": imported_count}


@router.post("/api/import/pdf")
@limiter.limit("20/minute")
async def import_pdf(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(require_auth),
) -> Note:
    """Accept a PDF file and create a note."""
    from src.ingestion.pdf_parser import parse_pdf

    content = await file.read()
    filename = file.filename or "imported.pdf"
    try:
        note = parse_pdf(content, filename)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _process_imported_note(request, note)


@router.post("/api/import/docx")
@limiter.limit("20/minute")
async def import_docx(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(require_auth),
) -> Note:
    """Accept a DOCX file and create a note."""
    from src.ingestion.docx_parser import parse_docx

    content = await file.read()
    filename = file.filename or "imported.docx"
    try:
        note = parse_docx(content, filename)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _process_imported_note(request, note)


@router.post("/api/import/audio")
@limiter.limit("20/minute")
async def import_audio(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(require_auth),
) -> Note:
    """Accept an audio file and create a note (requires Whisper)."""
    from src.ingestion.audio_parser import parse_audio

    content = await file.read()
    filename = file.filename or "imported.mp3"
    try:
        note = parse_audio(content, filename)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _process_imported_note(request, note)


@router.post("/api/import/image")
@limiter.limit("20/minute")
async def import_image(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(require_auth),
) -> Note:
    """Accept an image file and create a note via OCR (requires Tesseract)."""
    from src.ingestion.image_parser import parse_image

    content = await file.read()
    filename = file.filename or "imported.png"
    try:
        note = parse_image(content, filename)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _process_imported_note(request, note)


@router.post("/api/import/web")
@limiter.limit("20/minute")
async def import_web(
    request: Request,
    url: str,
    user=Depends(require_auth),
) -> Note:
    """Fetch a web page and create a note from the article text."""
    from src.ingestion.web_parser import parse_web_clip

    try:
        note = parse_web_clip(url)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _process_imported_note(request, note)


@router.post("/api/import/email")
@limiter.limit("20/minute")
async def import_email(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(require_auth),
) -> Note:
    """Accept a raw email (.eml) file and create a note."""
    from src.ingestion.email_parser import parse_email

    content = await file.read()
    filename = file.filename or "imported.eml"
    try:
        note = parse_email(content, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _process_imported_note(request, note)


# ---------------------------------------------------------------------------
# Logseq / Notion / Roam import
# ---------------------------------------------------------------------------


def _import_logseq_zip(request: Request, content: bytes) -> int:
    """Import Logseq graph zip: extract pages/ *.md and assets."""
    buf = io.BytesIO(content)
    if not zipfile.is_zipfile(buf):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip archive")
    buf.seek(0)
    store = get_note_store()
    imported_count = 0
    with zipfile.ZipFile(buf, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".md"):
                continue
            try:
                text = zf.read(name).decode("utf-8")
                note = parse_markdown_file(text, name)
                created = store.create(note)
                _extract_and_store_entities(request, created)
                _store_note_embeddings(request, created)
                imported_count += 1
            except Exception:
                continue
    return imported_count


def _import_notion_zip(request: Request, content: bytes) -> int:
    """Import Notion export zip: extract CSV + Markdown pages."""
    import csv

    buf = io.BytesIO(content)
    if not zipfile.is_zipfile(buf):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip archive")
    buf.seek(0)
    store = get_note_store()
    imported_count = 0
    with zipfile.ZipFile(buf, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".md"):
                try:
                    text = zf.read(name).decode("utf-8")
                    note = parse_markdown_file(text, name)
                    created = store.create(note)
                    _extract_and_store_entities(request, created)
                    _store_note_embeddings(request, created)
                    imported_count += 1
                except Exception:
                    continue
            elif name.endswith(".csv"):
                try:
                    text = zf.read(name).decode("utf-8")
                    reader = csv.DictReader(io.StringIO(text))
                    for row in reader:
                        title = row.get("Name", row.get("Title", "Untitled"))
                        body = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
                        note = Note(title=title, content=body, path=name)
                        created = store.create(note)
                        _extract_and_store_entities(request, created)
                        _store_note_embeddings(request, created)
                        imported_count += 1
                except Exception:
                    continue
    return imported_count


def _import_roam_zip(request: Request, content: bytes) -> int:
    """Import Roam Research JSON export: extract pages as notes."""
    import json as _json

    store = get_note_store()
    imported_count = 0
    try:
        data = _json.loads(content.decode("utf-8"))
    except (_json.JSONDecodeError, UnicodeDecodeError):
        # Maybe it's a zip with a JSON file inside
        buf = io.BytesIO(content)
        if zipfile.is_zipfile(buf):
            buf.seek(0)
            with zipfile.ZipFile(buf, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".json"):
                        try:
                            data = _json.loads(zf.read(name).decode("utf-8"))
                            break
                        except Exception:
                            continue
                else:
                    raise HTTPException(status_code=400, detail="No JSON file found in Roam zip")
        else:
            raise HTTPException(status_code=400, detail="Roam export is not valid JSON or zip")

    for page in data:
        if not isinstance(page, dict):
            continue
        title = page.get("title", "Untitled")
        children = page.get("children", [])
        lines: list[str] = []

        def _walk(nodes: list[dict], depth: int = 0):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                text = node.get("string", "")
                if text:
                    lines.append("  " * depth + text)
                kids = node.get("children", [])
                if kids:
                    _walk(kids, depth + 1)

        _walk(children)
        body = "\n".join(lines)
        note = Note(title=title, content=body, path=f"roam/{title}.md")
        created = store.create(note)
        _extract_and_store_entities(request, created)
        _store_note_embeddings(request, created)
        imported_count += 1
    return imported_count


@router.post("/api/import/logseq")
@limiter.limit("20/minute")
async def import_logseq(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(require_auth),
) -> dict:
    """Accept a Logseq graph export zip and import markdown pages."""
    content = await file.read()
    imported_count = _import_logseq_zip(request, content)
    # Invalidate semantic cache after bulk import
    import asyncio as _asyncio
    from src.cache.semantic_cache import cache_invalidate as _cache_invalidate
    _asyncio.ensure_future(_cache_invalidate())
    return {"imported_count": imported_count}


@router.post("/api/import/notion")
@limiter.limit("20/minute")
async def import_notion(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(require_auth),
) -> dict:
    """Accept a Notion export zip and import pages and CSV databases."""
    content = await file.read()
    imported_count = _import_notion_zip(request, content)
    # Invalidate semantic cache after bulk import
    import asyncio as _asyncio
    from src.cache.semantic_cache import cache_invalidate as _cache_invalidate
    _asyncio.ensure_future(_cache_invalidate())
    return {"imported_count": imported_count}


@router.post("/api/import/roam")
@limiter.limit("20/minute")
async def import_roam(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(require_auth),
) -> dict:
    """Accept a Roam Research JSON export and import pages as notes."""
    content = await file.read()
    imported_count = _import_roam_zip(request, content)
    # Invalidate semantic cache after bulk import
    import asyncio as _asyncio
    from src.cache.semantic_cache import cache_invalidate as _cache_invalidate
    _asyncio.ensure_future(_cache_invalidate())
    return {"imported_count": imported_count}


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------


@router.get("/api/export/note/{note_id}")
async def export_note(note_id: UUID) -> Response:
    """Export a single note as a markdown file."""
    store = get_note_store()
    note = store.get(str(note_id))
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")

    markdown_text = _note_to_markdown(note)
    filename = f"{note.title.replace(' ', '_')}.md"

    return Response(
        content=markdown_text,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/api/export/all")
async def export_all() -> Response:
    """Export all notes as a zip archive."""
    store = get_note_store()
    notes = store.list_all()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for note in notes:
            markdown_text = _note_to_markdown(note)
            safe_name = note.title.replace(" ", "_").replace("/", "_")
            arcname = f"{safe_name}_{note.id[:8]}.md"
            zf.writestr(arcname, markdown_text)

    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="aikioku_export.zip"'},
    )


@router.get("/api/export/json")
async def export_json(request: Request) -> JSONResponse:
    """Export all system data as JSON, including cognitive signals, settings, and plugin state."""
    store = get_note_store()
    notes = store.list_all()

    graph = getattr(request.app.state, "knowledge_graph", None)
    entities: list = []
    relations: list = []
    if graph is not None:
        entities = graph.find_entities(limit=10000)
        seen_rel_ids: set[str] = set()
        for entity in entities:
            for rel in graph.get_relations(entity.id):
                if rel.id not in seen_rel_ids:
                    seen_rel_ids.add(rel.id)
                    relations.append(rel)

    def _serialize_note(note):
        return {
            "id": note.id,
            "title": note.title,
            "content": note.content,
            "frontmatter": note.frontmatter,
            "links": note.links,
            "path": note.path,
            "created": note.created.isoformat()
            if isinstance(note.created, datetime)
            else str(note.created),
            "modified": note.modified.isoformat()
            if isinstance(note.modified, datetime)
            else str(note.modified),
        }

    def _serialize_entity(entity):
        return {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type.value if hasattr(entity.type, "value") else entity.type,
            "aliases": entity.aliases,
            "properties": entity.properties,
            "confidence": entity.confidence,
            "source_note_ids": entity.source_note_ids,
        }

    def _serialize_relation(relation):
        return {
            "id": relation.id,
            "source_entity_id": relation.source_entity_id,
            "target_entity_id": relation.target_entity_id,
            "type": relation.type.value if hasattr(relation.type, "value") else relation.type,
            "confidence": relation.confidence,
            "properties": relation.properties,
        }

    # Memories from SQLite
    memories = []
    try:
        import sqlite3

        conn = sqlite3.connect(settings.sqlite_db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM memories").fetchall()
        conn.close()
        for row in rows:
            row_dict = dict(row)
            memories.append(
                {
                    "id": row_dict.get("id"),
                    "subject": row_dict.get("subject"),
                    "predicate": row_dict.get("predicate"),
                    "object": row_dict.get("object"),
                    "confidence": row_dict.get("confidence"),
                    "source": row_dict.get("source"),
                    "created": row_dict.get("created"),
                    "modified": row_dict.get("modified"),
                    "vitality_score": row_dict.get("vitality_score"),
                    "tier": row_dict.get("tier"),
                }
            )
    except Exception:
        pass

    # Cards from SQLite
    cards = []
    try:
        import sqlite3

        conn = sqlite3.connect(settings.sqlite_db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM cards").fetchall()
        conn.close()
        for row in rows:
            row_dict = dict(row)
            cards.append(
                {
                    "id": row_dict.get("id"),
                    "note_id": row_dict.get("note_id"),
                    "type": row_dict.get("type"),
                    "front": row_dict.get("front"),
                    "back": row_dict.get("back"),
                    "ease_factor": row_dict.get("ease_factor"),
                    "interval": row_dict.get("interval"),
                    "repetitions": row_dict.get("repetitions"),
                    "next_review": row_dict.get("next_review"),
                    "status": row_dict.get("status"),
                }
            )
    except Exception:
        pass

    # Cognitive signals from SQLite
    cognitive_signals = []
    try:
        import sqlite3

        conn = sqlite3.connect(settings.sqlite_db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM cognitive_signals").fetchall()
        conn.close()
        for row in rows:
            row_dict = dict(row)
            cognitive_signals.append(
                {
                    "id": row_dict.get("id"),
                    "signal_type": row_dict.get("signal_type"),
                    "value": row_dict.get("value"),
                    "timestamp": row_dict.get("timestamp"),
                    "metadata": row_dict.get("metadata"),
                }
            )
    except Exception:
        pass

    # Settings from SQLite
    app_settings = {}
    try:
        import sqlite3

        conn = sqlite3.connect(settings.sqlite_db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        conn.close()
        app_settings = {row["key"]: row["value"] for row in rows}
    except Exception:
        pass

    # Plugin state from SQLite
    plugin_state = []
    try:
        import sqlite3

        conn = sqlite3.connect(settings.sqlite_db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM plugin_state").fetchall()
        conn.close()
        for row in rows:
            row_dict = dict(row)
            plugin_state.append(
                {
                    "id": row_dict.get("id"),
                    "plugin_id": row_dict.get("plugin_id"),
                    "key": row_dict.get("key"),
                    "value": row_dict.get("value"),
                }
            )
    except Exception:
        pass

    payload = {
        "version": "0.1.0",
        "notes": [_serialize_note(n) for n in notes],
        "entities": [_serialize_entity(e) for e in entities],
        "relations": [_serialize_relation(r) for r in relations],
        "memories": memories,
        "cards": cards,
        "cognitive_signals": cognitive_signals,
        "settings": app_settings,
        "plugin_state": plugin_state,
    }

    return JSONResponse(content=json.loads(json.dumps(payload, default=str)))


@router.get("/api/export/anki")
async def export_anki(
    format: str = Query("json", description="Export format: 'json' or 'apkg'"),
) -> Response:
    """Export all notes as Anki cards.

    - format=json (default): JSON representation of cards compatible with genanki.
    - format=apkg: Binary .apkg file ready for import into Anki.
    """
    store = get_note_store()
    notes = store.list_all()

    deck_name = "Aikioku"
    deck_id = 1234567890

    if format == "apkg":
        import genanki

        # Basic front/back model
        model = genanki.Model(
            1607392319,
            "Aikioku Basic",
            fields=[
                {"name": "Front"},
                {"name": "Back"},
            ],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": "{{Front}}",
                    "afmt": "{{FrontSide}}<hr id=answer>{{Back}}",
                },
            ],
        )

        deck = genanki.Deck(deck_id, deck_name)
        for note in notes:
            front = f"<b>{note.title}</b><br><br>{note.content[:500]}"
            back = note.content
            tags = note.frontmatter.get("tags", [])
            genanki_note = genanki.Note(
                model=model,
                fields=[front, back],
                tags=tags,
                guid=str(note.id),
            )
            deck.add_note(genanki_note)

        package = genanki.Package(deck)
        apkg_bytes = io.BytesIO()
        package.write_to_file(apkg_bytes)
        apkg_bytes.seek(0)

        return StreamingResponse(
            apkg_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="aikioku.apkg"',
            },
        )

    # Default: JSON format
    anki_notes = []
    for note in notes:
        front = f"<b>{note.title}</b><br><br>{note.content[:500]}"
        back = note.content
        anki_notes.append(
            {
                "guid": note.id,
                "modelName": "Basic",
                "fields": {
                    "Front": front,
                    "Back": back,
                },
                "tags": note.frontmatter.get("tags", []),
            }
        )

    payload = {
        "version": "0.1.0",
        "decks": [
            {
                "deck_id": deck_id,
                "name": deck_name,
                "description": "Exported from Aikioku",
            }
        ],
        "notes": anki_notes,
    }
    return JSONResponse(content=payload)


@router.get("/api/export/bibtex")
async def export_bibtex() -> Response:
    """Export notes as a BibTeX file.

    Each note becomes a @misc entry with title, note content as abstract,
    and tags as keywords.
    """
    store = get_note_store()
    notes = store.list_all()

    lines: list[str] = []
    for note in notes:
        key = f"note_{note.id[:8]}"
        title_escaped = note.title.replace("{", "}").replace("}", "}")
        abstract_escaped = note.content.replace("{", "}").replace("}", "}")[:2000]
        tags = ", ".join(note.frontmatter.get("tags", []))
        lines.append(f"@misc{{{key},")
        lines.append(f"  title = {{{title_escaped}}},")
        lines.append(f"  abstract = {{{abstract_escaped}}},")
        lines.append(f"  keywords = {{{tags}}},")
        lines.append(f"  url = {{{note.path}}},")
        lines.append(
            f"  date = {{{note.created.isoformat() if isinstance(note.created, datetime) else str(note.created)}}},"
        )
        lines.append("}")
        lines.append("")

    bibtex_text = "\n".join(lines)
    return Response(
        content=bibtex_text,
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="aikioku.bib"'},
    )
