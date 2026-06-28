"""Tests for MultiHopReasoner."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.knowledge.graph import KnowledgeGraph
from src.llm.base import LLMProvider
from src.models.entity import Entity, EntityType
from src.models.relation import Relation, RelationType
from src.reasoning.rag import RAGGenerator
from src.storage.note_store import NoteStore


@pytest.fixture
def mock_graph() -> MagicMock:
    """Return a mock KnowledgeGraph."""
    return MagicMock(spec=KnowledgeGraph)


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    """Return a mock LLMProvider."""
    provider = AsyncMock(spec=LLMProvider)
    provider.complete = AsyncMock(
        return_value=(
            '["What is Python?", "What are Python\'s key features?", "Who created Python?"]'
        )
    )
    return provider


@pytest.fixture
def mock_rag() -> AsyncMock:
    """Return a mock RAGGenerator."""
    rag = AsyncMock(spec=RAGGenerator)
    rag.generate = AsyncMock(
        return_value={
            "response": "Python is a programming language.",
            "citations": [{"note_id": "note-1", "score": 0.9, "snippet": "Python info."}],
            "memories": [],
        }
    )
    return rag


@pytest.fixture
def mock_note_store(tmp_path) -> NoteStore:
    """Return a NoteStore backed by a temp directory."""
    return NoteStore(str(tmp_path / "notes"))


class TestMultiHopReasoner:
    """Test MultiHopReasoner.reason returns response with sub_questions and citations."""

    @pytest.mark.asyncio
    async def test_reason_returns_response_with_sub_questions_and_citations(
        self, mock_graph, mock_llm_provider, mock_rag, mock_note_store
    ):
        """reason() should return a dict with response, sub_questions, and citations."""
        from src.reasoning.multi_hop import MultiHopReasoner

        # Configure RAG to return different answers per call
        mock_rag.generate = AsyncMock(
            side_effect=[
                {
                    "response": "Python is a programming language.",
                    "citations": [{"note_id": "note-1", "score": 0.9, "snippet": "Python info."}],
                    "memories": [],
                },
                {
                    "response": "Python features include dynamic typing.",
                    "citations": [{"note_id": "note-2", "score": 0.8, "snippet": "Features."}],
                    "memories": [],
                },
                {
                    "response": "Python was created by Guido van Rossum.",
                    "citations": [{"note_id": "note-3", "score": 0.7, "snippet": "Creator info."}],
                    "memories": [],
                },
            ]
        )

        reasoner = MultiHopReasoner(
            graph=mock_graph,
            llm_provider=mock_llm_provider,
            rag=mock_rag,
        )

        result = await reasoner.reason("Tell me about Python", mock_note_store)

        assert "response" in result
        assert "sub_questions" in result
        assert "citations" in result
        assert isinstance(result["response"], str)
        assert isinstance(result["sub_questions"], list)
        assert isinstance(result["citations"], list)
        assert len(result["sub_questions"]) == 3
        assert len(result["citations"]) == 3
        assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_reason_extracts_memories_from_final_answer(
        self, mock_graph, mock_llm_provider, mock_rag, mock_note_store
    ):
        """Memories are extracted once from the final synthesized answer (via the
        configured memory_extractor), not per sub-answer, so the chat path can
        persist them. Sub-questions skip extraction to bound concurrent LLM load."""
        from src.reasoning.multi_hop import MultiHopReasoner
        from src.memory.extraction import MemoryExtractor
        from src.models.memory import Memory

        m1 = Memory(
            subject="Python",
            predicate="is_a",
            object="language",
            confidence=0.9,
            source="conversation",
        )
        m2 = Memory(
            subject="Python",
            predicate="created_by",
            object="Guido",
            confidence=0.8,
            source="conversation",
        )

        mock_llm_provider.complete = AsyncMock(side_effect=['["q1", "q2"]', "synthesized answer"])
        # Sub-answers carry no memories now — extraction happens once at the end.
        mock_rag.generate = AsyncMock(
            side_effect=[
                {"response": "a1", "citations": [], "memories": []},
                {"response": "a2", "citations": [], "memories": []},
            ]
        )
        extractor = AsyncMock(spec=MemoryExtractor)
        extractor.extract_from_conversation = AsyncMock(return_value=[m1, m2])

        reasoner = MultiHopReasoner(
            graph=mock_graph,
            llm_provider=mock_llm_provider,
            rag=mock_rag,
            memory_extractor=extractor,
        )

        result = await reasoner.reason("Tell me about Python", mock_note_store)

        assert result["memories"] == [m1, m2]
        # Sub-questions must NOT extract memories (load control).
        for call in mock_rag.generate.await_args_list:
            assert call.kwargs.get("extract_memories") is False
        # Extraction ran exactly once, over the synthesized final answer.
        extractor.extract_from_conversation.assert_awaited_once()
        passed = extractor.extract_from_conversation.await_args.args[0]
        assert any(msg.get("content") == "synthesized answer" for msg in passed)

    @pytest.mark.asyncio
    async def test_reason_without_extractor_returns_no_memories(
        self, mock_graph, mock_llm_provider, mock_rag, mock_note_store
    ):
        """With no memory_extractor configured, a successful answer carries no
        memories (and never crashes)."""
        from src.reasoning.multi_hop import MultiHopReasoner

        mock_llm_provider.complete = AsyncMock(side_effect=['["q1", "q2"]', "synthesized answer"])
        mock_rag.generate = AsyncMock(
            return_value={"response": "a", "citations": [], "memories": []}
        )

        reasoner = MultiHopReasoner(graph=mock_graph, llm_provider=mock_llm_provider, rag=mock_rag)

        result = await reasoner.reason("Tell me about Python", mock_note_store)

        assert result["memories"] == []

    @pytest.mark.asyncio
    async def test_fallback_carries_memories(self, mock_graph, mock_llm_provider, mock_note_store):
        """When all sub-questions fail, the fallback's memories are carried out."""
        from src.reasoning.multi_hop import MultiHopReasoner
        from src.models.memory import Memory

        fb_mem = Memory(
            subject="X", predicate="is", object="Y", confidence=0.7, source="conversation"
        )
        fallback = {
            "response": "fallback answer",
            "citations": [],
            "memories": [fb_mem],
        }

        async def _generate(sub_q, note_store, **kwargs):
            if sub_q == "complex query":
                return fallback
            raise RuntimeError("boom")

        rag = AsyncMock(spec=RAGGenerator)
        rag.generate = AsyncMock(side_effect=_generate)
        mock_llm_provider.complete = AsyncMock(return_value='["q1", "q2"]')

        reasoner = MultiHopReasoner(graph=mock_graph, llm_provider=mock_llm_provider, rag=rag)

        result = await reasoner.reason("complex query", mock_note_store)

        assert result["memories"] == [fb_mem]

    @pytest.mark.asyncio
    async def test_decompose_query_returns_list_of_sub_questions(
        self, mock_graph, mock_llm_provider, mock_rag, mock_note_store
    ):
        """_decompose_query should return a list of sub-question strings."""
        from src.reasoning.multi_hop import MultiHopReasoner

        reasoner = MultiHopReasoner(
            graph=mock_graph,
            llm_provider=mock_llm_provider,
            rag=mock_rag,
        )

        sub_questions = await reasoner._decompose_query("Tell me about Python")

        assert isinstance(sub_questions, list)
        assert len(sub_questions) == 3
        assert all(isinstance(q, str) for q in sub_questions)
        mock_llm_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_combine_answers_merges_answers(
        self, mock_graph, mock_llm_provider, mock_rag, mock_note_store
    ):
        """_combine_answers should merge multiple answer dicts into a single response."""
        from src.reasoning.multi_hop import MultiHopReasoner

        reasoner = MultiHopReasoner(
            graph=mock_graph,
            llm_provider=mock_llm_provider,
            rag=mock_rag,
        )

        answers = [
            {
                "response": "Python is a language.",
                "citations": [{"note_id": "n1", "score": 0.9}],
            },
            {
                "response": "It has dynamic typing.",
                "citations": [{"note_id": "n2", "score": 0.8}],
            },
        ]

        combined = reasoner._combine_answers(answers)

        assert "response" in combined
        assert "citations" in combined
        assert "Python is a language" in combined["response"]
        assert "dynamic typing" in combined["response"]
        assert len(combined["citations"]) == 2

    @pytest.mark.asyncio
    async def test_single_sub_question_works(
        self, mock_graph, mock_llm_provider, mock_rag, mock_note_store
    ):
        """reason() should work when decomposition yields a single sub-question."""
        from src.reasoning.multi_hop import MultiHopReasoner

        mock_llm_provider.complete = AsyncMock(return_value='["What is Python?"]')

        reasoner = MultiHopReasoner(
            graph=mock_graph,
            llm_provider=mock_llm_provider,
            rag=mock_rag,
        )

        result = await reasoner.reason("What is Python?", mock_note_store)

        assert "response" in result
        assert "sub_questions" in result
        assert "citations" in result
        assert len(result["sub_questions"]) == 1
        assert result["sub_questions"][0] == "What is Python?"
        mock_rag.generate.assert_called_once()


class TestMultiHopConcurrencyAndResilience:
    """Concurrency, capping, timeout/error handling, synthesis, and graph grounding."""

    @pytest.mark.asyncio
    async def test_sub_questions_answered_concurrently(
        self, mock_graph, mock_llm_provider, mock_rag, mock_note_store
    ):
        """Three sub-questions each sleeping 0.3s should finish in ~0.3s, not ~0.9s."""
        from src.reasoning.multi_hop import MultiHopReasoner

        async def _slow_generate(sub_q, note_store, **kwargs):
            await asyncio.sleep(0.3)
            return {"response": f"answer to {sub_q}", "citations": [], "memories": []}

        mock_rag.generate = AsyncMock(side_effect=_slow_generate)

        reasoner = MultiHopReasoner(graph=mock_graph, llm_provider=mock_llm_provider, rag=mock_rag)

        start = time.perf_counter()
        result = await reasoner.reason("Tell me about Python", mock_note_store)
        elapsed = time.perf_counter() - start

        # If serial, 3 * 0.3 = 0.9s. Concurrent should be well under 0.6s.
        assert elapsed < 0.6, f"expected concurrent execution, took {elapsed:.2f}s"
        assert mock_rag.generate.await_count == 3
        assert len(result["sub_questions"]) == 3

    @pytest.mark.asyncio
    async def test_sub_questions_capped_to_three(
        self, mock_graph, mock_llm_provider, mock_rag, mock_note_store
    ):
        """Decomposition returning 6 sub-questions should cap to 3 answered."""
        from src.reasoning.multi_hop import MultiHopReasoner

        mock_llm_provider.complete = AsyncMock(return_value='["q1", "q2", "q3", "q4", "q5", "q6"]')

        reasoner = MultiHopReasoner(graph=mock_graph, llm_provider=mock_llm_provider, rag=mock_rag)

        result = await reasoner.reason("complex query", mock_note_store)

        assert len(result["sub_questions"]) == 3
        assert mock_rag.generate.await_count == 3

    @pytest.mark.asyncio
    async def test_one_sub_question_failing_still_returns_others(
        self, mock_graph, mock_llm_provider, mock_rag, mock_note_store
    ):
        """A sub-question that raises is dropped; reasoning continues with the rest."""
        from src.reasoning.multi_hop import MultiHopReasoner

        mock_llm_provider.complete = AsyncMock(
            side_effect=[
                '["q1", "q2", "q3"]',  # decomposition
                "Synthesized final answer",  # synthesis pass
            ]
        )

        call_state = {"n": 0}

        async def _flaky_generate(sub_q, note_store, **kwargs):
            call_state["n"] += 1
            if call_state["n"] == 2:
                raise RuntimeError("sub-question failed")
            return {
                "response": f"ok {sub_q}",
                "citations": [{"note_id": f"n{call_state['n']}", "score": 0.5}],
                "memories": [],
            }

        mock_rag.generate = AsyncMock(side_effect=_flaky_generate)

        reasoner = MultiHopReasoner(graph=mock_graph, llm_provider=mock_llm_provider, rag=mock_rag)

        result = await reasoner.reason("complex query", mock_note_store)

        assert result["response"] == "Synthesized final answer"
        # Only the two successful sub-answers contribute citations.
        assert len(result["citations"]) == 2

    @pytest.mark.asyncio
    async def test_all_sub_questions_failing_falls_back_to_single_generate(
        self, mock_graph, mock_llm_provider, mock_rag, mock_note_store
    ):
        """If every sub-question fails, fall back to a single rag.generate(original)."""
        from src.reasoning.multi_hop import MultiHopReasoner

        mock_llm_provider.complete = AsyncMock(return_value='["q1", "q2", "q3"]')

        async def _always_fail(sub_q, note_store, **kwargs):
            raise RuntimeError("boom")

        fallback = {
            "response": "fallback single-pass answer",
            "citations": [{"note_id": "fb", "score": 0.4}],
            "memories": [],
        }

        async def _generate(sub_q, note_store, **kwargs):
            if sub_q == "complex query":
                return fallback
            raise RuntimeError("boom")

        mock_rag.generate = AsyncMock(side_effect=_generate)

        reasoner = MultiHopReasoner(graph=mock_graph, llm_provider=mock_llm_provider, rag=mock_rag)

        result = await reasoner.reason("complex query", mock_note_store)

        assert result["response"] == "fallback single-pass answer"
        assert result["citations"] == [{"note_id": "fb", "score": 0.4}]

    @pytest.mark.asyncio
    async def test_response_is_llm_synthesized(self, mock_graph, mock_rag, mock_note_store):
        """The final response is the LLM-synthesized text (distinct from concat)."""
        from src.reasoning.multi_hop import MultiHopReasoner

        llm = AsyncMock(spec=LLMProvider)
        llm.complete = AsyncMock(
            side_effect=[
                '["q1", "q2"]',  # decomposition
                "A SINGLE COHERENT SYNTHESIS",  # synthesis
            ]
        )

        reasoner = MultiHopReasoner(graph=mock_graph, llm_provider=llm, rag=mock_rag)

        result = await reasoner.reason("complex query", mock_note_store)

        assert result["response"] == "A SINGLE COHERENT SYNTHESIS"

    @pytest.mark.asyncio
    async def test_synthesis_failure_falls_back_to_concatenation(
        self, mock_graph, mock_rag, mock_note_store
    ):
        """If the synthesis LLM call fails, fall back to concatenated sub-answers."""
        from src.reasoning.multi_hop import MultiHopReasoner

        mock_rag.generate = AsyncMock(
            side_effect=[
                {"response": "First part.", "citations": [], "memories": []},
                {"response": "Second part.", "citations": [], "memories": []},
            ]
        )

        llm = AsyncMock(spec=LLMProvider)
        llm.complete = AsyncMock(
            side_effect=[
                '["q1", "q2"]',  # decomposition succeeds
                RuntimeError("synthesis LLM down"),  # synthesis fails
            ]
        )

        reasoner = MultiHopReasoner(graph=mock_graph, llm_provider=llm, rag=mock_rag)

        result = await reasoner.reason("complex query", mock_note_store)

        assert "First part." in result["response"]
        assert "Second part." in result["response"]

    @pytest.mark.asyncio
    async def test_graph_context_included_in_synthesis_prompt(self, mock_rag, mock_note_store):
        """A matching entity+relation should surface in the synthesis prompt."""
        from src.reasoning.multi_hop import MultiHopReasoner

        ent = Entity(id="e1", name="Python", type=EntityType.Concept, confidence=0.9)
        rel = Relation(
            id="r1",
            source_entity_id="e1",
            target_entity_id="e2",
            type=RelationType.related_to,
            confidence=0.8,
        )
        graph = MagicMock(spec=KnowledgeGraph)
        graph.find_entities = MagicMock(return_value=[ent])
        graph.get_relations = MagicMock(return_value=[rel])
        graph.get_entity = MagicMock(
            return_value=Entity(id="e2", name="FastAPI", type=EntityType.Concept, confidence=0.9)
        )

        llm = AsyncMock(spec=LLMProvider)
        llm.complete = AsyncMock(side_effect=['["What is Python?"]', "final synthesized answer"])

        reasoner = MultiHopReasoner(graph=graph, llm_provider=llm, rag=mock_rag)

        await reasoner.reason("How do Python and FastAPI relate?", mock_note_store)

        # The second complete() call is synthesis; its prompt/system must carry graph facts.
        synthesis_call = llm.complete.await_args_list[1]
        combined_text = (
            " ".join(str(a) for a in synthesis_call.args)
            + " "
            + " ".join(str(v) for v in synthesis_call.kwargs.values())
        )
        assert "Python" in combined_text
        assert "FastAPI" in combined_text

    @pytest.mark.asyncio
    async def test_empty_graph_does_not_crash(self, mock_llm_provider, mock_rag, mock_note_store):
        """An empty/raising graph must not crash reasoning; context is just empty."""
        from src.reasoning.multi_hop import MultiHopReasoner

        graph = MagicMock(spec=KnowledgeGraph)
        graph.find_entities = MagicMock(side_effect=RuntimeError("graph down"))

        reasoner = MultiHopReasoner(graph=graph, llm_provider=mock_llm_provider, rag=mock_rag)

        result = await reasoner.reason("anything", mock_note_store)
        assert isinstance(result["response"], str)

    @pytest.mark.asyncio
    async def test_decompose_uses_parse_llm_json_fenced(
        self, mock_graph, mock_rag, mock_note_store
    ):
        """A fenced JSON array from the LLM is parsed correctly by parse_llm_json."""
        from src.reasoning.multi_hop import MultiHopReasoner

        llm = AsyncMock(spec=LLMProvider)
        llm.complete = AsyncMock(return_value='```json\n["sub A", "sub B"]\n```')
        reasoner = MultiHopReasoner(graph=mock_graph, llm_provider=llm, rag=mock_rag)

        subs = await reasoner._decompose_query("q")
        assert subs == ["sub A", "sub B"]

    @pytest.mark.asyncio
    async def test_decompose_garbage_falls_back_to_query(
        self, mock_graph, mock_rag, mock_note_store
    ):
        """Unparseable decomposition output falls back to [query]."""
        from src.reasoning.multi_hop import MultiHopReasoner

        llm = AsyncMock(spec=LLMProvider)
        llm.complete = AsyncMock(return_value="I cannot help with that.")
        reasoner = MultiHopReasoner(graph=mock_graph, llm_provider=llm, rag=mock_rag)

        subs = await reasoner._decompose_query("original question")
        assert subs == ["original question"]
