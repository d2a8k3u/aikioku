"""Web clip parser using requests and readability-lxml (with fallback)."""

from __future__ import annotations

from src.models.note import Note


def parse_web_clip(url: str) -> Note:
    """Fetch a web page and extract readable article text.

    Args:
        url: The URL to fetch.

    Returns:
        A Note with the article title and content.

    Raises:
        ImportError: If requests is not installed.
        RuntimeError: If the page cannot be fetched or parsed.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "Web clip dependencies are not installed. Install them with: pip install requests"
        ) from exc

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Aikioku/0.1.0"})
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch URL: {exc}") from exc

    html_text = resp.text

    # Try readability-lxml first, then fall back to lxml.html / html.parser
    try:
        from readability import Document  # type: ignore[import-untyped]

        doc = Document(html_text)
        title = doc.short_title() or url
        summary = doc.summary()
    except ImportError:
        try:
            from lxml import html as lh

            tree = lh.fromstring(html_text.encode("utf-8"))
            title_elem = tree.find(".//title")
            title = (title_elem.text if title_elem is not None else "") or url
            # Remove script/style tags
            for bad in tree.iter("script", "style", "nav", "footer", "header", "aside"):
                parent = bad.getparent()
                if parent is not None:
                    parent.remove(bad)
            body_elem = tree.find(".//body")
            if body_elem is not None:
                paragraphs = body_elem.iter(
                    "p", "h1", "h2", "h3", "h4", "h5", "h6", "article", "section"
                )
                parts: list[str] = []
                for el in paragraphs:
                    text = " ".join(str(t) for t in el.itertext()).strip()
                    if text:
                        parts.append(text)
                summary = "\n\n".join(parts)
            else:
                summary = html_text
        except Exception:
            # Last resort: strip tags with a regex
            import re

            summary = re.sub(r"<[^>]+>", "", html_text)
            title = url

    return Note(
        title=title,
        content=summary,
        path=url,
    )
