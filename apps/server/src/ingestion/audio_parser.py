"""Audio parser.

Audio transcription is not currently supported (the OpenAI Whisper API is no
longer a configured provider), so this returns a graceful fallback Note rather
than failing the import.
"""

from __future__ import annotations

import logging

from src.models.note import Note
from src.storage.file_import import _title_from_filename

logger = logging.getLogger(__name__)


def parse_audio(content: bytes, filename: str = "imported.mp3") -> Note:
    """Return a fallback Note — audio transcription is unavailable.

    Args:
        content: Raw audio bytes (unused; kept for the parser interface).
        filename: Original filename, used for the note title.

    Returns:
        A Note explaining that transcription is unavailable.
    """
    logger.warning(
        "Audio transcription unavailable — OpenAI provider removed. Returning empty note for %s",
        filename,
    )
    return Note(
        title=_title_from_filename(filename),
        content="[Transcription unavailable: OpenAI provider not supported]",
        path=filename,
    )
