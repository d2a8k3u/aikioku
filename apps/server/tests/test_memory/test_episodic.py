"""Tests for the deterministic episodic-memory builder."""

from __future__ import annotations

from datetime import datetime, timezone


def test_build_episodic_memory_records_the_question():
    from src.memory.extraction import build_episodic_memory

    created = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
    mem = build_episodic_memory("When is the deadline?", created=created)

    assert mem.subject == "user"
    assert mem.predicate == "asked_about"
    assert mem.object == "When is the deadline?"
    assert mem.source == "conversation"
    assert mem.created == created
    assert mem.modified == created
    # High confidence so it tiers as a live memory, not a faded one.
    assert mem.confidence >= 0.8
    assert mem.vitality_score >= 0.8


def test_build_episodic_memory_preserves_full_question():
    from src.memory.extraction import build_episodic_memory

    long_q = "x" * 1000
    mem = build_episodic_memory(long_q)
    # Text is never truncated — the full question is recorded.
    assert mem.object == long_q


def test_build_episodic_memory_handles_empty_question():
    from src.memory.extraction import build_episodic_memory

    mem = build_episodic_memory("")
    # object has min_length=1 on the model; the builder must supply a placeholder.
    assert mem.object
