"""Ingestion package for importing various file formats into Aikioku notes.

Each parser is importable individually so heavy dependencies are only loaded
when needed.
"""

__all__ = [
    "parse_audio",
    "parse_docx",
    "parse_email",
    "parse_image",
    "parse_pdf",
    "parse_web_clip",
]
