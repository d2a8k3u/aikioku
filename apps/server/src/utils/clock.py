from __future__ import annotations

from datetime import UTC, datetime


def utc_now_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
