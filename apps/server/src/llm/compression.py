"""Token-level perplexity compression via LLMLingua.

Provides ``compress_for_prompt`` — an async, drop-in alternative to
``condense_for_prompt`` that uses LLMLingua-2 token classification instead of
LLM-based summarisation.  When compression is disabled or unavailable the
function falls back to returning the raw text unchanged (never truncated).

Environment
-----------
``COMPRESSION_ENABLED`` (default ``"true"``) — set to ``"false"`` to skip
compression entirely and fall back to ``condense_for_prompt`` in the caller.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Default target token count for compressed output (~500 words / ~8000 chars).
_DEFAULT_TARGET_TOKENS = 2000

# Singleton compressor instance (loaded lazily, once per process).
_compressor: Any | None = None
_compressor_load_attempted: bool = False


def _get_compressor() -> Any | None:
    """Return the singleton PromptCompressor, loading it on first call.

    Returns ``None`` when ``llmlingua`` is not installed or the model cannot
    be loaded — callers must handle the ``None`` case gracefully.
    """
    global _compressor, _compressor_load_attempted

    if _compressor_load_attempted:
        return _compressor

    _compressor_load_attempted = True
    try:
        from llmlingua import PromptCompressor

        _compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
        logger.info("LLMLingua-2 PromptCompressor loaded (bert-base-multilingual, cpu)")
    except ImportError:
        logger.warning(
            "llmlingua is not installed; compression will fall back to raw text. "
            "Install with: pip install llmlingua"
        )
        _compressor = None
    except Exception:
        logger.warning(
            "Failed to load LLMLingua PromptCompressor; falling back to raw text.", exc_info=True
        )
        _compressor = None

    return _compressor


def _is_compression_enabled() -> bool:
    """Check the ``COMPRESSION_ENABLED`` environment variable.

    Returns ``True`` unless the variable is explicitly set to a falsy value
    (``"false"``, ``"0"``, ``"no"``, ``"off"``, or empty).
    """
    raw = os.environ.get("COMPRESSION_ENABLED", "true").strip().lower()
    return raw not in ("false", "0", "no", "off", "")


async def compress_for_prompt(
    text: str,
    *,
    query: str = "",
    target_tokens: int = _DEFAULT_TARGET_TOKENS,
) -> str:
    """Compress *text* using token-level perplexity pruning.

    This is an async, drop-in replacement for ``condense_for_prompt`` with the
    same interface shape: it accepts text and returns (possibly compressed)
    text.  Unlike ``condense_for_prompt`` it does **not** require an
    ``LLMProvider`` — compression runs locally on CPU via LLMLingua-2.

    Parameters
    ----------
    text:
        The text to compress.  Empty or very short text is returned unchanged.
    query:
        Accepted for interface parity with ``condense_for_prompt``.
        LLMLingua-2 classifies token importance query-agnostically, so this
        is currently unused.  Optional.
    target_tokens:
        Approximate token budget for the compressed output (default 2000).

    Returns
    -------
    str
        The compressed text, or the original *text* when compression is
        disabled, unavailable, or fails.
    """
    text = (text or "").strip()
    if not text:
        return text

    # Short-circuit: text already fits the target budget.
    # Rough heuristic: ~4 chars/token for English.
    if len(text) <= target_tokens * 4:
        return text

    if not _is_compression_enabled():
        logger.debug("COMPRESSION_ENABLED is false; returning raw text")
        return text

    compressor = _get_compressor()
    if compressor is None:
        return text

    # LLMLingua is synchronous — run it in a thread to keep the async event
    # loop free.
    try:
        result: dict[str, Any] = await asyncio.to_thread(
            compressor.compress_prompt,
            text,
            target_token=target_tokens,
            force_tokens=["\n", ".", "!", "?", ","],
        )
        compressed: str = result.get("compressed_prompt", "")
        if compressed and compressed.strip():
            logger.debug(
                "Compressed text from %d → %d chars (target %d tokens)",
                len(text),
                len(compressed),
                target_tokens,
            )
            return compressed.strip()
        logger.warning("LLMLingua returned empty compressed text; falling back to raw text")
        return text
    except Exception:
        logger.warning("LLMLingua compression failed; falling back to raw text", exc_info=True)
        return text
