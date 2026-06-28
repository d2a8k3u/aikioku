"""ConversationRetriever: recall past chat turns by embedding similarity.

Searches a dedicated ``"conversations"`` embedding collection (separate from the
note collection, so the note fusion path and ``NoteStore.get`` are never
touched). Each indexed turn embeds and stores the text
``"[<date>] user: <q>\nassistant: <a>"`` — the date travels inside the text, so
retrieved turns carry their timestamp and the model can answer questions about
what or when the user previously asked.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.knowledge.embeddings import EmbeddingStore
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class ConversationRetriever:
    """Embeds a query and searches the conversation-turn vector store."""

    def __init__(self, embedding_store: EmbeddingStore, llm_provider: LLMProvider) -> None:
        """Create a ConversationRetriever.

        Args:
            embedding_store: The ``"conversations"`` vector store to search.
            llm_provider: The LLM provider used to embed the query.
        """
        self._store = embedding_store
        self._llm = llm_provider

    async def search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Return up to ``limit`` past turns most similar to the query.

        Each result is a dict with keys ``text`` (the dated turn text), ``score``
        and ``note_id`` (the turn id). Never raises — recall is best-effort and
        must never fail the chat.
        """
        try:
            embedding = await asyncio.wait_for(self._llm.embed(query), timeout=3.0)
            return await asyncio.to_thread(self._store.search, embedding, limit=limit)
        except asyncio.TimeoutError:
            logger.warning("Conversation recall timed out; continuing without it.")
            return []
        except Exception:
            logger.warning("Conversation recall failed; continuing without it.", exc_info=True)
            return []
