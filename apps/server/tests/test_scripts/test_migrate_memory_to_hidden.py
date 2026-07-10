"""Tests for the memory-to-hidden migration script."""

from __future__ import annotations

from pathlib import Path

from src.scripts.migrate_memory_to_hidden import migrate_file, main, run_migration


def _write_note(path: Path, source_type: str, title: str = "Test") -> None:
    content = f"""---
id: test-{source_type}
title: "{title}"
source_type: {source_type}
---

This is the note body.
"""
    path.write_text(content, encoding="utf-8")


def test_migrate_file_memory_to_hidden(tmp_path: Path) -> None:
    md = tmp_path / "note-001.md"
    _write_note(md, "memory")
    assert migrate_file(md) is True
    text = md.read_text(encoding="utf-8")
    assert "source_type: hidden" in text
    assert "source_type: memory" not in text


def test_migrate_file_not_memory_skipped(tmp_path: Path) -> None:
    md = tmp_path / "note-002.md"
    _write_note(md, "note")
    assert migrate_file(md) is False
    text = md.read_text(encoding="utf-8")
    assert "source_type: note" in text


def test_migrate_file_already_hidden_skipped(tmp_path: Path) -> None:
    md = tmp_path / "note-003.md"
    _write_note(md, "hidden")
    assert migrate_file(md) is False


def test_migrate_file_no_frontmatter_skipped(tmp_path: Path) -> None:
    md = tmp_path / "note-004.md"
    md.write_text("No frontmatter here.", encoding="utf-8")
    assert migrate_file(md) is False


def test_main_migrates_directory(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_note(notes_dir / "a.md", "memory", "Note A")
    _write_note(notes_dir / "b.md", "note", "Note B")
    _write_note(notes_dir / "c.md", "memory", "Note C")

    exit_code = main(["--notes-dir", str(notes_dir)])

    assert exit_code == 0
    a_text = (notes_dir / "a.md").read_text(encoding="utf-8")
    c_text = (notes_dir / "c.md").read_text(encoding="utf-8")
    b_text = (notes_dir / "b.md").read_text(encoding="utf-8")
    assert "source_type: hidden" in a_text
    assert "source_type: hidden" in c_text
    assert "source_type: note" in b_text


def test_main_no_memory_notes(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_note(notes_dir / "a.md", "note", "Note A")

    exit_code = main(["--notes-dir", str(notes_dir)])

    assert exit_code == 0


def test_main_empty_directory(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    exit_code = main(["--notes-dir", str(notes_dir)])

    assert exit_code == 0


def test_main_nonexistent_directory(tmp_path: Path) -> None:
    exit_code = main(["--notes-dir", str(tmp_path / "nonexistent")])
    assert exit_code == 1


def test_run_migration_migrates_memory_notes(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_note(notes_dir / "a.md", "memory", "Note A")
    _write_note(notes_dir / "b.md", "note", "Note B")

    result = run_migration(str(notes_dir))

    assert result["migrated"] == 1
    assert result["skipped"] == 1
    assert result["failed"] == 0
    assert "source_type: hidden" in (notes_dir / "a.md").read_text(encoding="utf-8")


def test_run_migration_idempotent(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_note(notes_dir / "a.md", "memory", "Note A")

    first = run_migration(str(notes_dir))
    assert first["migrated"] == 1

    second = run_migration(str(notes_dir))
    assert second["migrated"] == 0
    assert second["skipped"] == 1


def test_run_migration_nonexistent_dir(tmp_path: Path) -> None:
    result = run_migration(str(tmp_path / "nonexistent"))
    assert result == {"migrated": 0, "skipped": 0, "failed": 0}


def test_run_migration_empty_dir(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    result = run_migration(str(notes_dir))
    assert result == {"migrated": 0, "skipped": 0, "failed": 0}