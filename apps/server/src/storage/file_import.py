"""File import helpers for parsing markdown, extracting wikilinks and tags."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

import yaml

from src.models.note import Note


def parse_markdown_file(content: str, filename: str) -> Note:
    """Parse a markdown string with optional YAML frontmatter into a Note.

    Args:
        content: Raw markdown text, optionally with YAML frontmatter delimited by ---.
        filename: Original filename, used as fallback title and stored in path.

    Returns:
        A fully populated Note instance.
    """
    title = _title_from_filename(filename)
    frontmatter: dict = {}
    body = content
    created = datetime.utcnow()
    modified = datetime.utcnow()

    # Try to extract YAML frontmatter
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if fm_match:
        yaml_text = fm_match.group(1)
        body = fm_match.group(2)
        try:
            metadata = yaml.safe_load(yaml_text)
            if isinstance(metadata, dict):
                if "title" in metadata:
                    title = metadata["title"]
                if "tags" in metadata:
                    frontmatter["tags"] = metadata["tags"] or []
                if "aliases" in metadata:
                    frontmatter["aliases"] = metadata["aliases"] or []
                if "links" in metadata:
                    frontmatter["links"] = metadata["links"] or []

                created_raw = metadata.get("created")
                if isinstance(created_raw, str):
                    created = datetime.fromisoformat(created_raw)
                elif isinstance(created_raw, datetime):
                    created = created_raw

                modified_raw = metadata.get("modified")
                if isinstance(modified_raw, str):
                    modified = datetime.fromisoformat(modified_raw)
                elif isinstance(modified_raw, datetime):
                    modified = modified_raw
        except yaml.YAMLError:
            pass  # Keep defaults if YAML is malformed
    else:
        # No frontmatter: try to extract title from first H1 heading
        h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()

    # Extract wikilinks and tags from the body
    wikilinks = extract_wikilinks(body)
    tags_from_body = extract_tags(body)

    # Merge body tags with frontmatter tags
    fm_tags = frontmatter.get("tags", [])
    merged_tags = list(dict.fromkeys(fm_tags + tags_from_body))  # preserve order, deduplicate
    if merged_tags:
        frontmatter["tags"] = merged_tags

    return Note(
        id=str(uuid.uuid4()),
        title=title,
        content=body,
        frontmatter=frontmatter,
        links=wikilinks,
        path=filename,
        created=created,
        modified=modified,
    )


def extract_wikilinks(content: str) -> list[str]:
    """Extract [[wikilinks]] from markdown content.

    Returns a deduplicated list of wikilink targets (including any alias text).
    """
    pattern = r"\[\[([^\]]+)\]\]"
    matches = re.findall(pattern, content)
    # Deduplicate while preserving order
    seen: dict[str, None] = {}
    for m in matches:
        seen.setdefault(m)
    return list(seen.keys())


def extract_tags(content: str) -> list[str]:
    """Extract #tags from markdown content, excluding tags inside code blocks.

    Returns a deduplicated list of tag names (without the # prefix).
    """
    # Remove code blocks (fenced with ``` or indented)
    # Handle fenced code blocks first
    cleaned = re.sub(r"```[\s\S]*?```", "", content)
    # Handle inline code
    cleaned = re.sub(r"`[^`]+`", "", cleaned)

    # Find #tags: must start at word boundary or after whitespace
    pattern = r"(?:^|\s)#([a-zA-Z][a-zA-Z0-9_-]*)"
    matches = re.findall(pattern, cleaned)

    # Deduplicate while preserving order
    seen: dict[str, None] = {}
    for m in matches:
        seen.setdefault(m)
    return list(seen.keys())


def _title_from_filename(filename: str) -> str:
    """Derive a human-readable title from a filename.

    Strips extension, replaces hyphens/underscores with spaces, title-cases.
    """
    base = filename.rsplit(".", 1)[0]  # Remove extension
    base = base.replace("-", " ").replace("_", " ")
    return base.title()
