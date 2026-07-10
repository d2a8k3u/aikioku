#!/usr/bin/env python3
"""Migrate legacy source_type="memory" notes to source_type="hidden".

After the SPO memory system removal, old notes with source_type="memory"
become visible in the UI (because HIDDEN_SOURCE_TYPES only contains "hidden").
Run this script once after deploy to convert all legacy memory notes to hidden.

Usage:
    cd apps/server && python3 -m src.scripts.migrate_memory_to_hidden
    # or: python3 src/scripts/migrate_memory_to_hidden.py [--notes-dir /data/notes]

Prints a summary: how many files were found, how many were migrated, how many
failed. Exits 0 on success, 1 if any file failed to migrate.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def migrate_file(md_path: Path) -> bool:
    content = md_path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        return False
    yaml_block = match.group(1)
    body = match.group(2)
    if "source_type: memory" not in yaml_block:
        return False
    new_yaml = yaml_block.replace("source_type: memory", "source_type: hidden")
    new_content = f"---\n{new_yaml}\n---\n{body}"
    tmp_path = md_path.parent / f".{md_path.name}.migrate.tmp"
    try:
        tmp_path.write_text(new_content, encoding="utf-8")
        os.replace(tmp_path, md_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return True


def run_migration(notes_dir: str) -> dict[str, int]:
    """Migrate all source_type='memory' notes to source_type='hidden' on disk.

    Returns a summary dict: {"migrated": N, "skipped": M, "failed": F}.
    Idempotent: a second run finds no memory notes and returns all skipped.
    """
    notes_path = Path(notes_dir)
    if not notes_path.is_dir():
        return {"migrated": 0, "skipped": 0, "failed": 0}

    md_files = sorted(notes_path.glob("*.md"))
    migrated = 0
    skipped = 0
    failed = 0

    for md_file in md_files:
        try:
            if migrate_file(md_file):
                migrated += 1
            else:
                skipped += 1
        except Exception:
            failed += 1

    return {"migrated": migrated, "skipped": skipped, "failed": failed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy source_type='memory' notes to source_type='hidden'."
    )
    parser.add_argument(
        "--notes-dir",
        default=None,
        help="Path to the notes directory. Defaults to the configured notes_dir.",
    )
    args = parser.parse_args(argv)

    if args.notes_dir:
        notes_dir = args.notes_dir
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from src.config import settings

        notes_dir = settings.notes_dir

    notes_path = Path(notes_dir)
    if not notes_path.is_dir():
        print(f"Error: notes directory not found: {notes_dir}", file=sys.stderr)
        return 1

    result = run_migration(notes_dir)
    print(
        f"\nMigration complete: {result['migrated']} migrated, "
        f"{result['skipped']} skipped, {result['failed']} failed."
    )

    if result["migrated"] > 0 and not args.notes_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from src.storage.note_store import NoteStore

        store = NoteStore(notes_dir)
        reindexed = store.reindex()
        print(f"Note index rebuilt: {reindexed} notes reindexed.")

    if result["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())