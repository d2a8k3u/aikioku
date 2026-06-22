"""Tests for RAGGenerator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.base import LLMProvider
from src.memory.extraction import MemoryExtractor
from src.models.memory import Memory
from src.models.note import Note
from src.retrieval.fusion import HybridFusion
from src.retrieval.search_result import SearchResult
from src.storage.note_store import NoteStore


@pytest.fixture
def mock_fusion() -> MagicMock:
    """Return a mock HybridFusion."""
    fusion = MagicMock(spec=HybridFusion)
    fusion.search = AsyncMock(return_value=[
        SearchResult(
            note_id="note-1",
            score=0.85,
            source="fusion",
            snippet="Python is a programming language.",
        ),
        SearchResult(
            note_id="note-2",
            score=0.72,
            source="fusion",
            snippet="Machine learning uses algorithms.",
        ),
    ])
    return fusion


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    """Return a mock LLMProvider."""
    provider = AsyncMock(spec=LLMProvider)
    provider.complete = AsyncMock(
        return_value="Python is a versatile programming language used in many domains."
    )
    return provider


@pytest.fixture
def mock_memory_extractor() -> MagicMock:
    """Return a mock MemoryExtractor."""
    extractor = MagicMock(spec=MemoryExtractor)
    extractor.extract_from_conversation = AsyncMock(return_value=[
        Memory(
            subject="Python",
            predicate="is_a",
            object="versatile programming language",
            confidence=0.9,
            source="conversation",
        ),
    ])
    return extractor


@pytest.fixture
def mock_note_store(tmp_path) -> NoteStore:
    """Return a NoteStore backed by a temp directory."""
    return NoteStore(str(tmp_path / "notes"))


class TestRAGGenerator:
    """Test RAGGenerator.generate returns response with citations and memories."""

    @pytest.mark.asyncio
    async def test_generate_returns_response_with_citations(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """generate() should return a dict with response, citations, and memories."""
        from src.reasoning.rag import RAGGenerator

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        result = await generator.generate("What is Python?", mock_note_store)

        assert "response" in result
        assert "citations" in result
        assert "memories" in result
        assert isinstance(result["response"], str)
        assert isinstance(result["citations"], list)
        assert isinstance(result["memories"], list)
        assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_generate_extracts_memories(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """generate() should call memory_extractor and return memories."""
        from src.reasoning.rag import RAGGenerator

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        result = await generator.generate("What is Python?", mock_note_store)

        mock_memory_extractor.extract_from_conversation.assert_called_once()
        assert len(result["memories"]) == 1
        assert result["memories"][0].subject == "Python"

    @pytest.mark.asyncio
    async def test_generate_skips_extraction_when_disabled(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """extract_memories=False must skip the second LLM call entirely and
        return no memories, while still producing the answer + citations.

        Used by multi-hop sub-questions to halve concurrent backend load and keep
        the answer generate inside its per-sub-question timeout budget.
        """
        from src.reasoning.rag import RAGGenerator

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        result = await generator.generate(
            "What is Python?", mock_note_store, extract_memories=False
        )

        mock_memory_extractor.extract_from_conversation.assert_not_called()
        assert result["memories"] == []
        assert len(result["response"]) > 0
        assert isinstance(result["citations"], list)
        # The answer generate itself must still run exactly once.
        mock_llm_provider.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_memory_extraction_excludes_system_context(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """Memory extraction must run over the conversational exchange ONLY.

        The system prompt now embeds the full retrieved note content (Phase 1).
        Feeding it to the extractor causes it to re-derive every fact from every
        retrieved note (noise). The extractor must therefore be called with only
        the user + assistant messages: no message may have role 'system', and the
        retrieved note text (which appears in the system prompt) must not leak
        into the messages handed to the extractor.
        """
        from src.reasoning.rag import RAGGenerator

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        await generator.generate("What is Python?", mock_note_store)

        mock_memory_extractor.extract_from_conversation.assert_called_once()
        (messages,) = mock_memory_extractor.extract_from_conversation.call_args.args

        roles = [m["role"] for m in messages]
        assert "system" not in roles, f"system message leaked into extraction: {roles}"
        assert roles == ["user", "assistant"]

        # The retrieved-note context (only present in the system prompt) must not
        # appear in any message handed to the extractor.
        joined = "\n".join(m["content"] for m in messages)
        assert "Python is a programming language." not in joined
        assert "Machine learning uses algorithms." not in joined

    @pytest.mark.asyncio
    async def test_generate_system_prompt_includes_context(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """The system prompt passed to the LLM should include the retrieved context."""
        from src.reasoning.rag import RAGGenerator

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        await generator.generate("What is Python?", mock_note_store)

        mock_llm_provider.complete.assert_called_once()
        call_kwargs = mock_llm_provider.complete.call_args
        system = call_kwargs.kwargs.get("system", "")
        assert "Python is a programming language." in system
        assert "Machine learning uses algorithms." in system

    @pytest.mark.asyncio
    async def test_generate_empty_context_still_works(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """generate() should work even when no chunks are retrieved."""
        from src.reasoning.rag import RAGGenerator

        mock_fusion.search = AsyncMock(return_value=[])

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        result = await generator.generate("obscure query", mock_note_store)

        assert "response" in result
        assert "citations" in result
        assert "memories" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_generate_calls_fusion_search_with_query(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """generate() should pass the query to fusion.search()."""
        from src.reasoning.rag import RAGGenerator

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        await generator.generate("test query", mock_note_store)

        mock_fusion.search.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_build_system_prompt_includes_instructions(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """_build_system_prompt should include instructions for the LLM."""
        from src.reasoning.rag import RAGGenerator

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        context = [
            {"note_id": "note-1", "snippet": "Test snippet.", "score": 0.9},
        ]
        prompt = generator._build_system_prompt(context)
        assert "Test snippet." in prompt
        assert "note-1" in prompt

    @pytest.mark.asyncio
    async def test_build_user_prompt_formats_query(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """_build_user_prompt should include the user query."""
        from src.reasoning.rag import RAGGenerator

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        prompt = generator._build_user_prompt("What is Python?")
        assert "What is Python?" in prompt

    @pytest.mark.asyncio
    async def test_extract_citations_returns_citation_dicts(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """_extract_citations should return a list of citation dicts."""
        from src.reasoning.rag import RAGGenerator

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        context = [
            {"note_id": "note-1", "snippet": "Python is great.", "score": 0.85},
            {"note_id": "note-2", "snippet": "ML is fun.", "score": 0.72},
        ]
        citations = generator._extract_citations("Python is great.", context)

        assert isinstance(citations, list)
        assert len(citations) == 2
        assert all("note_id" in c for c in citations)
        assert all("score" in c for c in citations)
        assert citations[0]["note_id"] == "note-1"
        assert citations[0]["score"] == 0.85

    @pytest.mark.asyncio
    async def test_generate_survives_memory_extractor_failure(
        self, mock_fusion, mock_llm_provider, mock_note_store
    ):
        """If memory extraction raises (e.g. LLM ReadTimeout), the already-generated
        response and citations must still be returned and memories falls back to []."""
        from src.reasoning.rag import RAGGenerator

        failing_extractor = MagicMock(spec=MemoryExtractor)
        failing_extractor.extract_from_conversation = AsyncMock(
            side_effect=RuntimeError("ReadTimeout against remote LLM")
        )

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=failing_extractor,
        )

        result = await generator.generate("What is Python?", mock_note_store)

        assert result["response"] is not None
        assert len(result["response"]) > 0
        assert isinstance(result["citations"], list)
        assert len(result["citations"]) >= 1
        assert result["memories"] == []

    @pytest.mark.asyncio
    async def test_generate_uses_full_note_content_in_prompt(
        self, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """When fusion returns a result with an empty snippet but note_store.get
        resolves a real note, the system prompt must contain the note's content."""
        from src.reasoning.rag import RAGGenerator

        note = Note(
            title="Docker",
            content="Docker is a platform for running applications in containers.",
            path="docker.md",
        )
        mock_note_store.create(note)

        fusion = MagicMock(spec=HybridFusion)
        fusion.search = AsyncMock(
            return_value=[
                SearchResult(note_id=note.id, score=0.9, source="fusion", snippet="")
            ]
        )

        generator = RAGGenerator(
            fusion=fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        await generator.generate("What is Docker?", mock_note_store)

        mock_llm_provider.complete.assert_called_once()
        system = mock_llm_provider.complete.call_args.kwargs.get("system", "")
        assert "Docker is a platform for running applications in containers." in system

    @pytest.mark.asyncio
    async def test_extract_citations_deduplicates_by_note_id(
        self, mock_fusion, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """_extract_citations must yield at most one citation per note_id."""
        from src.reasoning.rag import RAGGenerator

        generator = RAGGenerator(
            fusion=mock_fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        context = [
            {"note_id": "note-1", "snippet": "first", "score": 0.85},
            {"note_id": "note-1", "snippet": "duplicate", "score": 0.70},
            {"note_id": "note-2", "snippet": "other", "score": 0.60},
        ]
        citations = generator._extract_citations("answer", context)

        note_ids = [c["note_id"] for c in citations]
        assert note_ids.count("note-1") == 1
        assert set(note_ids) == {"note-1", "note-2"}

    @pytest.mark.asyncio
    async def test_build_context_returns_system_prompt_and_citations(
        self, mock_llm_provider, mock_memory_extractor, mock_note_store
    ):
        """build_context() retrieves once and returns (system_prompt, citations)
        without invoking the LLM for generation (single-pass streaming support)."""
        from src.reasoning.rag import RAGGenerator

        note = Note(
            title="Docker",
            content="Docker is a platform for running applications in containers.",
            path="docker.md",
        )
        mock_note_store.create(note)

        fusion = MagicMock(spec=HybridFusion)
        fusion.search = AsyncMock(
            return_value=[
                SearchResult(note_id=note.id, score=0.9, source="fusion", snippet="")
            ]
        )

        generator = RAGGenerator(
            fusion=fusion,
            llm_provider=mock_llm_provider,
            memory_extractor=mock_memory_extractor,
        )

        system_prompt, citations = await generator.build_context(
            "What is Docker?", mock_note_store
        )

        assert isinstance(system_prompt, str)
        assert "Docker is a platform for running applications in containers." in system_prompt
        assert isinstance(citations, list)
        assert len(citations) == 1
        assert citations[0]["note_id"] == note.id
        # build_context must NOT call the LLM to generate an answer.
        mock_llm_provider.complete.assert_not_called()
