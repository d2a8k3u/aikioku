"""DOCX parser using python-docx."""

from __future__ import annotations

import io

from src.models.note import Note
from src.storage.file_import import _title_from_filename


def parse_docx(content: bytes, filename: str = "imported.docx") -> Note:
    """Extract text from a DOCX file and return a Note.

    Args:
        content: Raw DOCX bytes.
        filename: Original filename for title fallback.

    Returns:
        A Note with the extracted text as content.

    Raises:
        ImportError: If python-docx is not installed.
        ValueError: If the DOCX cannot be parsed.
    """
    try:
        import docx
    except ImportError as exc:
        raise ImportError(
            "python-docx is not installed. Install it with: pip install python-docx"
        ) from exc

    try:
        document = docx.Document(io.BytesIO(content))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
    except Exception as exc:
        raise ValueError(f"Failed to parse DOCX: {exc}") from exc

    title = _title_from_filename(filename)
    if paragraphs:
        first = paragraphs[0].strip()
        if len(first) < 120:
            title = first

    return Note(
        title=title,
        content=full_text,
        path=filename,
    )
