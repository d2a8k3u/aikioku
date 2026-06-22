"""PDF parser using PyMuPDF (fitz)."""

from __future__ import annotations

from src.models.note import Note
from src.storage.file_import import _title_from_filename


def parse_pdf(content: bytes, filename: str = "imported.pdf") -> Note:
    """Extract text from a PDF and return a Note.

    Args:
        content: Raw PDF bytes.
        filename: Original filename for title fallback.

    Returns:
        A Note with the extracted text as content.

    Raises:
        ImportError: If PyMuPDF is not installed.
        ValueError: If the PDF cannot be parsed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is not installed. Install it with: pip install PyMuPDF"
        ) from exc

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        parts: list[str] = []
        for page in doc:
            text = page.get_text()
            if text:
                parts.append(text)
        full_text = "\n\n".join(parts)
        doc.close()
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF: {exc}") from exc

    title = _title_from_filename(filename)
    # Try to extract title from first line if it looks like a heading
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    if lines and len(lines[0]) < 120:
        title = lines[0]

    return Note(
        title=title,
        content=full_text,
        path=filename,
    )
