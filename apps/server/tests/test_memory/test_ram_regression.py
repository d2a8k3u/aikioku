"""RAM regression guard for the chat pipeline.

Measures Python-side allocation growth over 100 unique chat turns with the
reasoning stack mocked. tracemalloc is the primary gate; RSS is a secondary
canary. The semantic cache is disabled here on purpose: its per-turn cost is
dominated by ONNX model inference (a constant ~100 MB of resident model
memory, not a leak), and its entry-growth behavior is already pinned by
tests/test_cache/test_semantic_cache_bounds.py.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_WARMUP_TURNS = 5
_MEASURED_TURNS = 100
_SRC_GROWTH_BYTES = 2 * 1024 * 1024
_UNFILTERED_GROWTH_BYTES = 15 * 1024 * 1024
_RSS_CANARY_BYTES = 200 * 1024 * 1024
_BG_DRAIN_SECONDS = 15.0


@pytest.fixture
def mock_rag_generator() -> AsyncMock:
    mock = AsyncMock()
    mock.generate.return_value = {
        "response": "RAG answer",
        "citations": [{"note_id": "note-1", "score": 0.95, "snippet": "snippet"}],
    }
    mock.build_context.return_value = (
        "SYSTEM PROMPT",
        [{"note_id": "note-1", "score": 0.95, "snippet": "snippet"}],
    )
    mock._extract_citations = MagicMock(
        return_value=[{"note_id": "note-1", "score": 0.95, "snippet": "snippet"}]
    )
    return mock


@pytest.fixture
def mock_multi_hop_reasoner() -> AsyncMock:
    mock = AsyncMock()
    mock.reason.return_value = {
        "response": "Multi-hop answer",
        "citations": [{"note_id": "note-2", "score": 0.88, "snippet": "snippet"}],
        "sub_questions": ["q1"],
    }
    return mock


@pytest.fixture
def client(
    mock_rag_generator: AsyncMock, mock_multi_hop_reasoner: AsyncMock, tmp_path: Any
) -> Iterator[tuple[TestClient, Any]]:
    with (
        patch("src.api.chat._build_rag_generator", return_value=mock_rag_generator),
        patch("src.api.chat._build_multi_hop_reasoner", return_value=mock_multi_hop_reasoner),
        patch("src.api.chat.NoteStore") as mock_store_cls,
        patch("src.api.chat.settings") as mock_settings,
    ):
        mock_store_cls.return_value = MagicMock()
        mock_settings.notes_dir = str(tmp_path)
        from src.main import app

        yield TestClient(app), app


def _drain_background_tasks(app: Any) -> None:
    deadline = time.monotonic() + _BG_DRAIN_SECONDS
    while time.monotonic() < deadline:
        if not getattr(app.state, "_bg_tasks", None):
            return
        time.sleep(0.1)


def _rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return 0
    return int(psutil.Process().memory_info().rss)


def _snapshot_bytes(snapshot: Any, path_fragment: str | None = None) -> int:
    if path_fragment:
        filtered = snapshot.filter_traces([tracemalloc.Filter(True, f"*{path_fragment}*")])
        return sum(stat.size for stat in filtered.statistics("filename"))
    return sum(stat.size for stat in snapshot.statistics("filename"))


@pytest.mark.slow
def test_chat_pipeline_ram_regression(client: tuple[TestClient, Any], monkeypatch: Any) -> None:
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
    cli, app = client
    limiter = getattr(app.state, "limiter", None)
    if limiter is not None:
        limiter.enabled = False

    for i in range(_WARMUP_TURNS):
        response = cli.post("/api/chat", json={"query": f"warmup query {i}", "mode": "simple"})
        assert response.status_code == 200
    _drain_background_tasks(app)

    rss_before = _rss_bytes()

    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for i in range(_MEASURED_TURNS):
        response = cli.post(
            "/api/chat", json={"query": f"unique query number {i}", "mode": "simple"}
        )
        assert response.status_code == 200
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    _drain_background_tasks(app)

    src_growth = _snapshot_bytes(after, "/src/") - _snapshot_bytes(before, "/src/")
    unfiltered_growth = _snapshot_bytes(after) - _snapshot_bytes(before)
    rss_growth = _rss_bytes() - rss_before

    assert len(getattr(app.state, "_bg_tasks", set())) == 0
    assert src_growth < _SRC_GROWTH_BYTES, f"src allocation growth {src_growth} bytes"
    assert unfiltered_growth < _UNFILTERED_GROWTH_BYTES, (
        f"unfiltered allocation growth {unfiltered_growth} bytes"
    )
    assert rss_growth < _RSS_CANARY_BYTES, f"RSS growth {rss_growth} bytes"
