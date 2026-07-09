"""Tests for Chat API endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_rag_generator():
    """Create a mock RAGGenerator."""
    mock = AsyncMock()
    mock.generate.return_value = {
        "response": "RAG answer about Python",
        "citations": [
            {"note_id": "note-1", "score": 0.95, "snippet": "Python is a language"},
        ],
    }
    mock.build_context.return_value = (
        "SYSTEM PROMPT WITH NOTE CONTENT",
        [{"note_id": "note-1", "score": 0.95, "snippet": "Python is a language"}],
    )
    # _extract_citations is a real sync method; AsyncMock would turn it async.
    mock._extract_citations = MagicMock(
        return_value=[
            {"note_id": "note-1", "score": 0.95, "snippet": "Python is a language"},
        ]
    )
    return mock


@pytest.fixture
def mock_multi_hop_reasoner():
    """Create a mock MultiHopReasoner."""
    mock = AsyncMock()
    mock.reason.return_value = {
        "response": "Multi-hop combined answer about Python",
        "citations": [
            {"note_id": "note-2", "score": 0.88, "snippet": "Python supports OOP"},
        ],
        "sub_questions": ["What is Python?", "What are its features?"],
    }
    return mock


@pytest.fixture
def client(mock_rag_generator, mock_multi_hop_reasoner, tmp_path):
    """Create a FastAPI TestClient with mocked reasoning dependencies."""
    with (
        patch("src.api.chat._build_rag_generator", return_value=mock_rag_generator),
        patch("src.api.chat._build_multi_hop_reasoner", return_value=mock_multi_hop_reasoner),
        patch("src.api.chat.NoteStore") as mock_store_cls,
        patch("src.api.chat.settings") as mock_settings,
    ):
        mock_store_cls.return_value = MagicMock()
        mock_settings.notes_dir = str(tmp_path)
        from src.main import app

        yield TestClient(app), mock_rag_generator, mock_multi_hop_reasoner


class TestChatEndpoint:
    """Test POST /api/chat endpoint."""

    def test_returns_response_with_citations(self, client):
        cli, mock_rag, mock_multi = client
        response = cli.post(
            "/api/chat",
            json={"query": "What is Python?", "mode": "simple"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "citations" in data
        assert "mode" in data
        assert data["mode"] == "simple"
        assert isinstance(data["citations"], list)
        assert len(data["citations"]) > 0

    def test_simple_mode_uses_rag_generator(self, client):
        cli, mock_rag, mock_multi = client
        response = cli.post(
            "/api/chat",
            json={"query": "What is Python?", "mode": "simple"},
        )
        assert response.status_code == 200
        mock_rag.generate.assert_called_once()
        mock_multi.reason.assert_not_called()

    def test_multi_hop_mode_uses_multi_hop_reasoner(self, client):
        cli, mock_rag, mock_multi = client
        response = cli.post(
            "/api/chat",
            json={"query": "How does Python compare to Rust?", "mode": "multi_hop"},
        )
        assert response.status_code == 200
        mock_multi.reason.assert_called_once()
        mock_rag.generate.assert_not_called()

    def test_multi_hop_response_includes_sub_questions(self, client):
        cli, mock_rag, mock_multi = client
        response = cli.post(
            "/api/chat",
            json={"query": "How does Python compare to Rust?", "mode": "multi_hop"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sub_questions"] == [
            "What is Python?",
            "What are its features?",
        ]

    def test_missing_query_returns_422(self, client):
        cli, mock_rag, mock_multi = client
        response = cli.post(
            "/api/chat",
            json={"mode": "simple"},
        )
        assert response.status_code == 422


class TestChatPersistsConversation:
    """POST /api/chat/ must persist the user + assistant turn (conversation embedding)."""

    def test_persists_user_and_assistant_messages(self, client):
        from src.api import conversations as conv

        cli, mock_rag, mock_multi = client
        mock_rag.generate.return_value = {
            "response": "the answer",
            "citations": [],
        }

        response = cli.post(
            "/api/chat",
            json={"query": "my question", "mode": "simple"},
        )
        assert response.status_code == 200

        rows = conv.load_messages("anonymous", limit=10)
        pairs = {(r["role"], r["content"]) for r in rows}
        assert ("user", "my question") in pairs
        assert ("assistant", "the answer") in pairs


class _FakeStreamingLLM:
    """An LLM stand-in whose stream() yields fixed chunks; records stream args."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.stream_calls = []

    async def stream(self, prompt: str, system: str = "", **kwargs):
        self.stream_calls.append({"prompt": prompt, "system": system})
        for chunk in self._chunks:
            yield chunk


class TestChatStreamSinglePass:
    """POST /api/chat/stream (simple mode) must be a TRUE single grounded pass."""

    def test_simple_stream_uses_build_context_and_streams_once(self, client):
        cli, mock_rag, mock_multi = client

        fake_llm = _FakeStreamingLLM(["Hello ", "world."])
        cli.app.state.llm_provider = fake_llm

        with cli.stream(
            "POST",
            "/api/chat/stream",
            json={"query": "What is Docker?", "mode": "simple"},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

        # build_context used exactly once; generate() NOT used in the stream path.
        mock_rag.build_context.assert_called_once()
        mock_rag.generate.assert_not_called()

        # The stream was invoked once, grounded on the build_context system prompt.
        # (A tone preamble may be prepended, so assert containment, not equality.)
        assert len(fake_llm.stream_calls) == 1
        assert "SYSTEM PROMPT WITH NOTE CONTENT" in fake_llm.stream_calls[0]["system"]
        assert fake_llm.stream_calls[0]["prompt"] == "What is Docker?"

        # SSE protocol: citations event, non-empty message chunks, done.
        assert "event: citations" in body
        assert "event: message" in body
        assert "event: done" in body
        assert "Hello " in body
        assert "world." in body

    def test_get_stream_uses_build_context(self, client):
        cli, mock_rag, mock_multi = client

        fake_llm = _FakeStreamingLLM(["chunk1", "chunk2"])
        cli.app.state.llm_provider = fake_llm

        with cli.stream(
            "GET",
            "/api/chat/stream?query=What+is+Docker&mode=simple",
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

        mock_rag.build_context.assert_called_once()
        mock_rag.generate.assert_not_called()
        assert len(fake_llm.stream_calls) == 1
        assert "event: citations" in body
        assert "chunk1" in body

    def test_multi_hop_stream_does_not_rerun_llm_on_answer(self, client):
        cli, mock_rag, mock_multi = client

        fake_llm = _FakeStreamingLLM(["SHOULD NOT BE USED"])
        cli.app.state.llm_provider = fake_llm

        with cli.stream(
            "POST",
            "/api/chat/stream",
            json={"query": "How do Python and FastAPI relate?", "mode": "multi_hop"},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

        mock_multi.reason.assert_called_once()
        # The synthesized multi-hop answer is streamed verbatim, not re-generated.
        assert "Multi-hop combined answer about Python" in body
        assert len(fake_llm.stream_calls) == 0
        assert "event: citations" in body
        assert "event: done" in body