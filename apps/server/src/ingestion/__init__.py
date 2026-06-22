"""Ingestion package for importing various file formats into Aikioku notes.

Each parser is importable individually so heavy dependencies are only loaded
when needed.
"""

__all__ = [
    "parse_pdf",
    "parse_docx",
    "parse_audio",
    "parse_image",
    "parse_web_clip",
    "parse_email",
]
