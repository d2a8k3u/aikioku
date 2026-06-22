"""RAG Generator: retrieval-augmented generation pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.llm.base import LLMProvider
from src.llm.compression import compress_for_prompt
from src.memory.extraction import MemoryExtractor
from src.models.memory import Memory

if TYPE_CHECKING:
    from src.retrieval.conversation_retrieval import ConversationRetriever
    from src.retrieval.fusion import HybridFusion
    from src.storage.note_store import NoteStore

logger = logging.getLogger(__name__)

# Number of top fused results to ground the answer on.
_CONTEXT_TOP_K = 5


class RAGGenerator:
    """Retrieval-Augmented Generation pipeline.

    Combines hybrid retrieval with LLM generation and memory extraction
    to produce cited, context-aware responses.
    """

    def __init__(
        self,
        fusion: HybridFusion,
        llm_provider: LLMProvider,
        memory_extractor: MemoryExtractor,
        conversation_retriever: "ConversationRetriever | None" = None,
    ) -> None:
        """Create a RAGGenerator.

        Args:
            fusion: The hybrid fusion retriever for searching notes.
            llm_provider: The LLM provider for generating responses.
            memory_extractor: The memory extractor for extracting
                structured memories from the generated response.
            conversation_retriever: Optional retriever over past chat turns.
                When supplied, similar past turns (with their dates) are added to
                the grounding context so the assistant can answer questions about
                what or when the user previously asked. Kept SEPARATE from the
                note fusion path — conversation ids are not note ids.
        """
        self._fusion = fusion
        self._llm = llm_provider
        self._memory_extractor = memory_extractor
        self._conversation_retriever = conversation_retriever

    async def build_context(
        self,
        query: str,
        note_store: NoteStore,
        history: list[dict] | None = None,
    ) -> tuple[str, list[dict]]:
        """Retrieve and build the grounding context for a query.

        Performs retrieval, full-note-content context assembly, system-prompt
        construction, and citation extraction. Does NOT call the LLM, so it can
        be reused by both ``generate`` and true single-pass streaming.

        Args:
            query: The user's question.
            note_store: The note store (used for context lookup).
            history: Optional recent conversation turns (chronological) injected
                into the system prompt for short-term continuity.

        Returns:
            A tuple of (system_prompt, citations). Citations cover only the note
            context; recalled conversation turns are injected as context text.
        """
        # Step 1: Retrieve relevant chunks
        results = await self._fusion.search(query)

        # Step 2: Build context from the top-K results, grounding on the full
        # note content (title + body) rather than the retrieval snippet alone.
        context = list(
            await asyncio.gather(
                *(self._build_context_entry(r, note_store, query=query) for r in results[:_CONTEXT_TOP_K])
            )
        )

        # Step 2b: Recall similar past conversation turns (best-effort, separate
        # from the note fusion path).
        conversation_context = await self._recall_conversations(query)

        # Step 3: Build the system prompt and citations.
        system_prompt = self._build_system_prompt(
            context, conversation_context, history
        )
        citations = self._extract_citations("", context)
        return system_prompt, citations

    async def _recall_conversations(self, query: str) -> list[dict]:
        """Recall past chat turns similar to the query (best-effort, may be empty)."""
        if self._conversation_retriever is None:
            return []
        try:
            return await asyncio.wait_for(
                self._conversation_retriever.search(query), timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("Conversation recall timed out; continuing without it.")
            return []
        except Exception:
            logger.warning(
                "Conversation recall failed; continuing without it.", exc_info=True
            )
            return []

    async def generate(
        self,
        query: str,
        note_store: NoteStore,
        *,
        extract_memories: bool = True,
        history: list[dict] | None = None,
    ) -> dict:
        """Generate a RAG response for the given query.

        Steps:
            1. Build grounding context (retrieve + system prompt + citations).
            2. Call llm_provider.complete() to generate response.
            3. Optionally extract memories from the response.
            4. Return response, citations, and memories.

        Args:
            query: The user's question.
            note_store: The note store (used for context lookup).
            extract_memories: When True (default), run a second LLM pass to
                extract memories from the answer. Callers that fan many
                ``generate`` calls out concurrently under a tight timeout
                (e.g. multi-hop sub-questions) pass False: the extra LLM call
                doubles backend load and, when wrapped in a per-call timeout,
                can discard an already-generated answer because its trailing
                extraction call is still queued. Such callers extract memories
                once from the final answer instead.

        Returns:
            A dict with keys: response (str), citations (list[dict]),
            memories (list[Memory]). ``memories`` is empty when
            ``extract_memories`` is False.
        """
        # Step 1: Build grounding context (retrieval + system prompt + citations).
        system_prompt, citations = await self.build_context(
            query, note_store, history=history
        )
        user_prompt = self._build_user_prompt(query)

        # Step 2: Generate response via LLM
        response = await self._llm.complete(
            prompt=user_prompt,
            system=system_prompt,
        )

        # Step 3: Extract memories from the conversation (best-effort, opt-out).
        # A failure here (e.g. a ReadTimeout against the remote LLM) must never
        # discard the already-generated answer.
        #
        # Only the conversational exchange (user question + assistant answer) is
        # passed to extraction. The system prompt embeds the FULL retrieved note
        # content (Phase 1 grounding); including it would re-derive every fact
        # from every retrieved note, polluting captured memory with dozens of
        # note facts instead of what was actually asked and answered.
        memories: list[Memory] = []
        if extract_memories:
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ]
            try:
                memories = await self._memory_extractor.extract_from_conversation(messages)
            except Exception:
                logger.warning("Memory extraction failed; returning answer without memories.", exc_info=True)
                memories = []

        return {
            "response": response,
            "citations": citations,
            "memories": memories,
        }

    async def _build_context_entry(
        self, result, note_store: NoteStore, *, query: str = ""
    ) -> dict:
        """Build a single context entry for a fused result.

        Fetches the real note via ``note_store.get`` and grounds on its full
        title + body. The content is never truncated: notes that fit a single
        pass are used verbatim, oversized notes are compressed via token-level
        perplexity pruning (LLMLingua) so the whole note is still represented.
        Falls back to the retrieval snippet when the note cannot be resolved.

        Args:
            result: A fused SearchResult.
            note_store: The note store used to resolve full content.
            query: The user's question, passed to the compressor so it can
                condition which tokens to keep.

        Returns:
            A context dict with note_id, snippet, content, score, and title.
        """
        note = note_store.get(result.note_id)
        if note is not None:
            body = (note.content or "").strip()
            full = f"{note.title}\n{body}".strip()
            content = await compress_for_prompt(full, query=query)
            title = note.title
        else:
            content = result.snippet
            title = None
        return {
            "note_id": result.note_id,
            "snippet": result.snippet,
            "content": content,
            "score": result.score,
            "title": title,
        }

    def _build_system_prompt(
        self,
        context: list[dict],
        conversation_context: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> str:
        """Build the system prompt with retrieved context and instructions.

        Args:
            context: A list of dicts with note_id, snippet, and score.
            conversation_context: Recalled past chat turns (each with dated
                ``text``), surfaced so the model can answer "when did I ask X".
            history: Recent conversation turns (chronological) for short-term
                continuity ("what was my previous question").

        Returns:
            The formatted system prompt string.
        """
        lines = [
            "You are a helpful assistant with access to the user's knowledge base.",
            "Answer the user's question directly using the context below. The answer "
            "may require SYNTHESIZING across multiple notes — combine related "
            "information even when no single note explicitly defines the term or "
            "states the answer verbatim.",
            "Only reply that information is missing when the context is clearly "
            "irrelevant to the question. When relevant material is present, give a "
            "best-effort answer grounded in it and note any uncertainty briefly — do "
            "not refuse just because no note is an exact match.",
            "Always respond in the same language as the user's most recent message.",
            "",
            "Context:",
        ]

        if context:
            for i, chunk in enumerate(context, 1):
                text = chunk.get("content") or chunk.get("snippet", "")
                lines.append(
                    f"[{i}] (note: {chunk['note_id']}, score: {chunk['score']:.4f}) "
                    f"{text}"
                )
        else:
            lines.append("No relevant context found.")

        if conversation_context:
            lines.append("")
            lines.append("Relevant past conversations (each prefixed with its date):")
            for turn in conversation_context:
                turn_text = turn.get("text", "").strip()
                if turn_text:
                    lines.append(f"- {turn_text}")

        if history:
            lines.append("")
            lines.append("Recent conversation (oldest first, most recent last):")
            for msg in history:
                lines.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")

        if conversation_context or history:
            lines.append("")
            lines.append(
                "The blocks above may include the user's own earlier messages and "
                "past conversations with dates. Use them to answer questions about "
                "what the user previously asked or when they asked it."
            )

        lines.append("")
        lines.append(
            "Now answer the user's question directly and concisely, grounded in the "
            "context above. Lead with the answer (synthesized across notes if needed), "
            "then support it and reference the context. Do not hedge or list 'closest "
            "matches' when the context already supports an answer."
        )

        return "\n".join(lines)

    def _build_user_prompt(self, query: str) -> str:
        """Build the user prompt from the query.

        Args:
            query: The user's question.

        Returns:
            The formatted user prompt string.
        """
        return f"Question: {query}"

    def _extract_citations(
        self, response: str, context: list[dict]
    ) -> list[dict]:
        """Extract citation information from the response and context.

        Returns a list of citation dicts for each context chunk,
        including note_id, score, and a relevance indicator.

        Args:
            response: The generated LLM response text.
            context: The list of context dicts used for generation.

        Returns:
            A list of citation dicts with note_id, score, and snippet
            (and title when the note was resolved), deduplicated by note_id.
        """
        citations = []
        seen: set[str] = set()
        for chunk in context:
            note_id = chunk["note_id"]
            if note_id in seen:
                continue
            seen.add(note_id)
            citation = {
                "note_id": note_id,
                "score": chunk["score"],
                "snippet": chunk["snippet"],
                "title": chunk.get("title") or "",
            }
            citations.append(citation)
        return citations
