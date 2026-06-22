"""MultiHopReasoner: decompose complex queries and combine RAG answers."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.llm.base import LLMProvider
from src.llm.json_parse import LLMOutputParseError, parse_llm_json

if TYPE_CHECKING:
    from src.knowledge.graph import KnowledgeGraph
    from src.memory.extraction import MemoryExtractor
    from src.reasoning.rag import RAGGenerator
    from src.storage.note_store import NoteStore

logger = logging.getLogger(__name__)

# Maximum number of sub-questions answered per query. Bounds remote-LLM fan-out.
_MAX_SUB_QUESTIONS = 3
# Per sub-question timeout (seconds). A slow/hung sub-question is dropped, not fatal.
_SUB_Q_TIMEOUT_S = 45.0
# Maximum number of graph entities used to ground the synthesis.
_MAX_GRAPH_ENTITIES = 3
# Maximum relations rendered per entity in the graph-context block.
_MAX_GRAPH_RELATIONS_PER_ENTITY = 5

# Prompt template for query decomposition
_DECOMPOSE_SYSTEM_PROMPT = """You are a query decomposition assistant.
Break the user's complex question into 2-4 simpler sub-questions that can be answered independently.
Return ONLY a JSON array of strings, e.g. ["sub-question 1", "sub-question 2", "sub-question 3"]."""

_DECOMPOSE_USER_PROMPT_TEMPLATE = """Decompose this question into sub-questions: {query}"""

# Prompt template for the final synthesis pass.
_SYNTHESIS_SYSTEM_PROMPT = """You are a reasoning assistant. You are given a user's
question, several sub-questions with their researched answers, and (optionally)
known facts from a knowledge graph. Synthesize ONE coherent, accurate answer to
the original question by COMBINING the material across sub-answers and facts — the
answer often requires connecting information that no single source states verbatim.
Lead with a direct answer, then support it. Ground your answer in the provided
material; do not invent facts. Only say the information is missing when the material
is genuinely irrelevant — do NOT refuse or fall back to listing "closest matches"
when the material already supports an answer. Be concise. Always respond in the same language as the user's original question."""


class MultiHopReasoner:
    """Multi-hop reasoning via query decomposition and RAG answer synthesis.

    Decomposes a complex query into sub-questions using an LLM, answers each
    via RAG concurrently, grounds the result with knowledge-graph facts, then
    synthesizes the sub-answers into a single coherent response.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        llm_provider: LLMProvider,
        rag: RAGGenerator,
        memory_extractor: MemoryExtractor | None = None,
    ) -> None:
        """Create a MultiHopReasoner.

        Args:
            graph: The knowledge graph for entity/relation lookups.
            llm_provider: The LLM provider for decomposition and synthesis.
            rag: The RAG generator for answering sub-questions.
            memory_extractor: Optional extractor used to capture memories once
                from the final synthesized answer. When None, the success path
                returns no memories. Sub-questions never extract memories
                themselves (see ``_answer_sub_questions``), so this is the only
                memory source for a successful multi-hop answer.
        """
        self._graph = graph
        self._llm = llm_provider
        self._rag = rag
        self._memory_extractor = memory_extractor

    async def reason(
        self,
        query: str,
        note_store: NoteStore,
        history: list[dict] | None = None,
    ) -> dict:
        """Perform multi-hop reasoning on the query.

        Steps:
            1. Decompose query into sub-questions via LLM (capped).
            2. Answer each sub-question via RAG concurrently, dropping failures.
            3. Ground with knowledge-graph facts and synthesize a final answer.

        Args:
            query: The user's complex question.
            note_store: The note store for context lookup.
            history: Optional recent conversation turns (chronological) used to
                ground the synthesis (and the single-pass fallback) so the model
                can answer questions about the conversation itself.

        Returns:
            A dict with keys: response (str), sub_questions (list[str]),
            citations (list[dict]).
        """
        # Step 1: Decompose
        sub_questions = await self._decompose_query(query)
        logger.debug("Decomposed query into %d sub-questions", len(sub_questions))

        # Step 2: Answer each sub-question via RAG, concurrently. A sub-question
        # that raises or times out is skipped; reasoning continues with the rest.
        answers = await self._answer_sub_questions(sub_questions, note_store)

        # If every sub-question failed, fall back to a single grounded pass.
        if not answers:
            logger.warning(
                "All sub-questions failed; falling back to single-pass generation."
            )
            fallback = await self._rag.generate(query, note_store, history=history)
            return {
                "response": fallback.get("response", ""),
                "citations": fallback.get("citations", []),
                "sub_questions": sub_questions,
                "memories": fallback.get("memories", []),
            }

        # Step 3: Ground with graph facts and synthesize.
        graph_context = self._graph_context(query)
        combined = await self._synthesize(
            query, sub_questions, answers, graph_context, history
        )
        combined["sub_questions"] = sub_questions
        # Capture memories once, from the final synthesized answer. Sub-questions
        # deliberately skip extraction (it doubled the concurrent LLM load and,
        # under the per-sub-question timeout, discarded answers whose trailing
        # extraction call was still queued). The synthesized answer is also a
        # cleaner extraction source than the fragmented intermediate sub-answers.
        combined["memories"] = await self._extract_memories(
            query, combined.get("response", "")
        )

        return combined

    async def _answer_sub_questions(
        self, sub_questions: list[str], note_store: NoteStore
    ) -> list[dict]:
        """Answer sub-questions concurrently, dropping any that fail or time out.

        Args:
            sub_questions: The sub-questions to answer.
            note_store: The note store for context lookup.

        Returns:
            The list of successful answer dicts, preserving sub-question order.
        """

        async def _answer(sub_q: str) -> dict | None:
            try:
                # extract_memories=False: the 45s budget must cover only
                # retrieval + the answer generate. Memory extraction is a second
                # LLM call; fanning it out per sub-question doubled the load on a
                # single local backend and let a still-queued extraction discard
                # an already-produced answer. Memories are taken once from the
                # final synthesized answer instead (see reason()).
                return await asyncio.wait_for(
                    self._rag.generate(sub_q, note_store, extract_memories=False),
                    timeout=_SUB_Q_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning("Sub-question timed out, dropping: %s", sub_q)
                return None
            except Exception:
                logger.warning(
                    "Sub-question failed, dropping: %s", sub_q, exc_info=True
                )
                return None

        results = await asyncio.gather(*[_answer(sq) for sq in sub_questions])
        return [r for r in results if r is not None]

    async def _extract_memories(self, query: str, answer: str) -> list:
        """Extract memories from the final answer (best-effort, single LLM call).

        Returns ``[]`` when no extractor is configured, the answer is empty, or
        extraction fails — capturing memories must never fail the chat response.

        Args:
            query: The original user query.
            answer: The synthesized final answer.

        Returns:
            A list of extracted Memory objects (possibly empty).
        """
        if self._memory_extractor is None or not answer:
            return []
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        try:
            return await self._memory_extractor.extract_from_conversation(messages)
        except Exception:
            logger.warning(
                "Final memory extraction failed; continuing without memories.",
                exc_info=True,
            )
            return []

    async def _decompose_query(self, query: str) -> list[str]:
        """Decompose a complex query into sub-questions using the LLM.

        Args:
            query: The original complex query.

        Returns:
            A list of sub-question strings, capped to _MAX_SUB_QUESTIONS. Falls
            back to ``[query]`` when the LLM output cannot be parsed.
        """
        user_prompt = _DECOMPOSE_USER_PROMPT_TEMPLATE.format(query=query)
        raw = await self._llm.complete(
            prompt=user_prompt,
            system=_DECOMPOSE_SYSTEM_PROMPT,
        )

        try:
            parsed = parse_llm_json(raw, expect="list")
        except LLMOutputParseError:
            logger.warning("Failed to parse decomposition, using original query")
            return [query]

        sub_questions = [item for item in parsed if isinstance(item, str) and item.strip()]
        if not sub_questions:
            logger.warning("Decomposition yielded no usable sub-questions; using query")
            return [query]

        return sub_questions[:_MAX_SUB_QUESTIONS]

    async def _synthesize(
        self,
        query: str,
        sub_questions: list[str],
        answers: list[dict],
        graph_context: str,
        history: list[dict] | None = None,
    ) -> dict:
        """Synthesize sub-answers into one coherent response via the LLM.

        On synthesis failure, falls back to concatenating the sub-answers.
        Citations are merged and deduplicated across all sub-answers.

        Args:
            query: The original user query.
            sub_questions: The sub-questions that were answered.
            answers: The successful sub-answer dicts.
            graph_context: A formatted block of knowledge-graph facts (may be "").
            history: Optional recent conversation turns for continuity.

        Returns:
            A dict with 'response' (synthesized or concatenated) and 'citations'.
        """
        citations = self._merge_citations(answers)

        prompt = self._build_synthesis_prompt(
            query, sub_questions, answers, graph_context, history
        )
        try:
            response = await self._llm.complete(
                prompt=prompt,
                system=_SYNTHESIS_SYSTEM_PROMPT,
            )
            if not response or not response.strip():
                raise ValueError("empty synthesis response")
        except Exception:
            logger.warning(
                "Synthesis failed; falling back to concatenation.", exc_info=True
            )
            concatenated = self._combine_answers(answers)
            return {
                "response": concatenated["response"],
                "citations": citations,
            }

        return {"response": response, "citations": citations}

    @staticmethod
    def _build_synthesis_prompt(
        query: str,
        sub_questions: list[str],
        answers: list[dict],
        graph_context: str,
        history: list[dict] | None = None,
    ) -> str:
        """Build the user prompt for the synthesis pass."""
        lines = [f"Original question: {query}", ""]
        if history:
            lines.append("Recent conversation (oldest first, most recent last):")
            for msg in history:
                lines.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
            lines.append(
                "(You may use the conversation above to answer questions about "
                "what or when the user previously asked.)"
            )
            lines.append("")
        if graph_context:
            lines.append(graph_context)
            lines.append("")
        lines.append("Sub-questions and their answers:")
        for i, answer in enumerate(answers):
            sub_q = sub_questions[i] if i < len(sub_questions) else f"Sub-question {i + 1}"
            resp = answer.get("response", "")
            lines.append(f"- Q: {sub_q}")
            lines.append(f"  A: {resp}")
        lines.append("")
        lines.append(
            "Synthesize a single coherent answer to the original question."
        )
        return "\n".join(lines)

    def _graph_context(self, query: str) -> str:
        """Build a short graph-grounding block for the query.

        Looks up salient entities by name and gathers their relations,
        formatting a bounded "Known facts" block. Never raises: on any error
        or empty graph, returns "".

        Args:
            query: The original user query.

        Returns:
            A formatted facts block, or "" when no facts are available.
        """
        try:
            entities = self._collect_entities(query)
            if not entities:
                return ""

            fact_lines: list[str] = []
            for entity in entities[:_MAX_GRAPH_ENTITIES]:
                relations = self._graph.get_relations(entity.id)
                for relation in relations[:_MAX_GRAPH_RELATIONS_PER_ENTITY]:
                    other_id = (
                        relation.target_entity_id
                        if relation.source_entity_id == entity.id
                        else relation.source_entity_id
                    )
                    other = self._graph.get_entity(other_id)
                    other_name = other.name if other is not None else other_id
                    rel_type = getattr(relation.type, "value", str(relation.type))
                    fact_lines.append(
                        f"- {entity.name} {rel_type} {other_name}"
                    )

            if not fact_lines:
                return ""

            return "Known facts from the knowledge graph:\n" + "\n".join(fact_lines)
        except Exception:
            logger.warning("Graph context lookup failed; continuing without it.", exc_info=True)
            return ""

    def _collect_entities(self, query: str) -> list:
        """Collect salient entities for the query from the graph (bounded)."""
        seen_ids: set[str] = set()
        entities: list = []

        # Probe with capitalized tokens (likely proper nouns / concepts), then
        # fall back to the whole query.
        tokens = [t.strip(".,!?;:\"'") for t in query.split()]
        candidates = [t for t in tokens if t and t[0].isupper()]
        candidates.append(query)

        for candidate in candidates:
            if len(entities) >= _MAX_GRAPH_ENTITIES:
                break
            found = self._graph.find_entities(name=candidate, limit=_MAX_GRAPH_ENTITIES)
            for entity in found:
                if entity.id not in seen_ids:
                    seen_ids.add(entity.id)
                    entities.append(entity)
                if len(entities) >= _MAX_GRAPH_ENTITIES:
                    break

        return entities

    @staticmethod
    def _merge_citations(answers: list[dict]) -> list[dict]:
        """Merge and deduplicate citations across sub-answers (by note_id)."""
        seen_note_ids: set[str] = set()
        merged: list[dict] = []
        for answer in answers:
            for citation in answer.get("citations", []):
                note_id = citation.get("note_id", "")
                if note_id and note_id not in seen_note_ids:
                    seen_note_ids.add(note_id)
                    merged.append(citation)
                elif not note_id:
                    merged.append(citation)
        return merged

    def _combine_answers(self, answers: list[dict]) -> dict:
        """Concatenate sub-answers into a final response (synthesis fallback).

        Args:
            answers: List of answer dicts from RAG, each with 'response'
                     and optionally 'citations'.

        Returns:
            A dict with 'response' (combined text) and 'citations' (merged list).
        """
        response_parts = []
        for answer in answers:
            resp = answer.get("response", "")
            if resp:
                response_parts.append(resp)

        combined_response = "\n\n".join(response_parts)
        merged_citations = self._merge_citations(answers)

        return {
            "response": combined_response,
            "citations": merged_citations,
        }
