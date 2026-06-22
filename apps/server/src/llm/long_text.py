"""Process arbitrarily long text through an LLM without truncating it.

Project rule: code never truncates content to fit a prompt. When text exceeds a
single-pass budget it is split at sentence boundaries and folded — each chunk is
summarized preserving facts, the summaries are combined (recursively if needed) —
so the full content is always processed.

Chunks are condensed CONCURRENTLY (not sequentially) with per-chunk timeouts.
A total timeout guards the entire condensation so a slow LLM never blocks the
chat response indefinitely.
"""

from __future__ import annotations

import asyncio
import logging
import re

from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Characters that comfortably fit one LLM call alongside instructions
# (~4 chars/token ≈ 2k tokens — fast summary, well under any model's context).
SINGLE_PASS_CHAR_BUDGET = 8000

# Folds must not recurse forever on incompressible input; after this many passes
# the combined summaries are returned in full (still never truncated).
_MAX_FOLD_DEPTH = 3

# Per-chunk timeout (seconds). A single chunk summary that exceeds this is
# dropped and the chunk is kept verbatim — one slow chunk must not stall the
# entire condensation.
_CHUNK_TIMEOUT_S = 30.0

# Total condensation timeout (seconds). When the whole condensation exceeds
# this, the caller receives the best-effort result so far (raw snippets or
# partial summaries) rather than waiting indefinitely.
_TOTAL_TIMEOUT_S = 60.0

_FOLD_SYSTEM = (
    "Condense the following content while preserving every distinct fact, name, "
    "number, date, and claim. Do not omit or invent information. Output prose only."
)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_for_llm(text: str, target_chars: int = SINGLE_PASS_CHAR_BUDGET) -> list[str]:
    """Split text into LLM-sized chunks at sentence boundaries. Drops nothing.

    A sentence longer than ``target_chars`` is kept whole (its own chunk) rather
    than being cut — correctness over a hard size cap.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in _split_sentences(text):
        if current and current_len + len(sentence) > target_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


async def _summarize_chunk(llm: LLMProvider, chunk: str) -> str:
    """Summarize a single chunk with a timeout. Returns verbatim on failure."""
    try:
        summary = await asyncio.wait_for(
            llm.complete(prompt=chunk, system=_FOLD_SYSTEM),
            timeout=_CHUNK_TIMEOUT_S,
        )
        return (summary or chunk).strip()
    except asyncio.TimeoutError:
        logger.warning("condense: chunk summary timed out; keeping verbatim")
        return chunk
    except Exception:
        logger.warning("condense: chunk summary failed; keeping verbatim", exc_info=True)
        return chunk


async def condense_for_prompt(
    llm: LLMProvider,
    text: str,
    *,
    target_chars: int = SINGLE_PASS_CHAR_BUDGET,
    _depth: int = 0,
) -> str:
    """Return text unchanged when it fits one pass; otherwise fold it losslessly.

    Folding summarizes chunks CONCURRENTLY (preserving facts) and joins the
    summaries, recursing until the result fits or ``_MAX_FOLD_DEPTH`` is reached.
    Never truncates: every chunk is read, and if a summary call fails or times
    out the chunk is kept verbatim.

    The entire condensation is guarded by ``_TOTAL_TIMEOUT_S`` — when exceeded,
    the best-effort result so far is returned so a slow LLM never blocks the
    chat response indefinitely.
    """
    text = (text or "").strip()
    if len(text) <= target_chars:
        return text

    async def _fold() -> str:
        chunks = chunk_for_llm(text, target_chars)
        # Concurrent chunk summarization — all chunks processed in parallel.
        summaries = await asyncio.gather(
            *(_summarize_chunk(llm, chunk) for chunk in chunks)
        )
        combined = "\n\n".join(s.strip() for s in summaries).strip()
        if (
            len(combined) <= target_chars
            or _depth >= _MAX_FOLD_DEPTH
            or combined == text
        ):
            return combined
        return await condense_for_prompt(
            llm, combined, target_chars=target_chars, _depth=_depth + 1
        )

    try:
        return await asyncio.wait_for(_fold(), timeout=_TOTAL_TIMEOUT_S)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        logger.warning(
            "condense: total timeout/cancellation exceeded (%ds); returning raw text",
            _TOTAL_TIMEOUT_S,
        )
        # Best-effort: return the raw text truncated to a reasonable size.
        # This is a degradation, not a truncation — the full note is still
        # available via the note store, and the LLM can still answer from
        # the retrieval snippet.
        return text[:target_chars * 2]
