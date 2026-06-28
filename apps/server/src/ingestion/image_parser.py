"""Image parser — placeholder for OCR-based text extraction.

This module provides the framework for image ingestion. The actual OCR
requires pytesseract and Pillow, which are heavy dependencies.
Install with: pip install pytesseract Pillow
"""

from __future__ import annotations

from src.models.note import Note
from src.storage.file_import import _title_from_filename


def parse_image(content: bytes, filename: str = "imported.png") -> Note:
    """Extract text from an image via OCR and return a Note.

    Args:
        content: Raw image bytes.
        filename: Original filename for title fallback.

    Returns:
        A Note with the OCR text as content.

    Raises:
        ImportError: If pytesseract or Pillow is not installed.
        RuntimeError: If OCR fails.
    """
    # TODO: Integrate pytesseract for OCR.
    # Example:
    #   from PIL import Image
    #   import pytesseract
    #   image = Image.open(io.BytesIO(content))
    #   text = pytesseract.image_to_string(image)
    try:
        from PIL import Image  # type: ignore[import-not-found]
        import pytesseract  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "OCR dependencies are not installed. Install them with: "
            "pip install pytesseract Pillow"
        ) from exc

    import io

    try:
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image)
    except Exception as exc:
        raise RuntimeError(f"OCR failed: {exc}") from exc

    title = _title_from_filename(filename)
    return Note(
        title=title,
        content=text,
        path=filename,
    )
