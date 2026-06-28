"""Email parser using imaplib and email.parser."""

from __future__ import annotations

import email
import email.utils
from datetime import datetime

from src.models.note import Note


def parse_email(raw_bytes: bytes, filename: str = "imported.eml") -> Note:
    """Parse a raw email message (RFC 822) and return a Note.

    Args:
        raw_bytes: Raw email bytes.
        filename: Original filename for fallback path.

    Returns:
        A Note with the email subject and body text.

    Raises:
        ValueError: If the email cannot be parsed.
    """
    try:
        msg = email.message_from_bytes(raw_bytes)
    except Exception as exc:
        raise ValueError(f"Failed to parse email: {exc}") from exc

    subject = msg.get("Subject", "Imported Email")
    date_str = msg.get("Date")
    created: datetime | None = None
    if date_str:
        try:
            created = email.utils.parsedate_to_datetime(date_str)
        except Exception:
            pass

    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            if isinstance(payload, bytes):
                text = payload.decode("utf-8", errors="replace")
            else:
                text = str(payload)
            if ctype == "text/plain":
                body_parts.append(text)
            elif ctype == "text/html":
                # TODO: Use html2text or BeautifulSoup for better HTML->text
                body_parts.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if payload is not None:
            if isinstance(payload, bytes):
                body_parts.append(payload.decode("utf-8", errors="replace"))
            else:
                body_parts.append(str(payload))

    full_text = "\n\n".join(body_parts)
    note = Note(
        title=subject,
        content=full_text,
        path=filename,
    )
    if created:
        note.created = created
        note.modified = created
    return note
