"""Tests for build_runtime() old-provider cleanup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _setup_minimal_app_state(app):
    """Set up the minimum app.state attributes that build_runtime() needs."""
    app.state.embedding_store = MagicMock()
    app.state.knowledge_graph = MagicMock()
    app.state.event_bus = MagicMock()
    app.state.conversation_embedding_store = None


class TestBuildRuntimeCleanup:
    """Verify build_runtime() closes old providers before building new ones."""

    def test_build_runtime_closes_old_providers(self):
        """Old providers on app.state have aclose() called during build_runtime()."""
        from src.main import app, build_runtime

        _setup_minimal_app_state(app)

        old_llm = MagicMock()
        old_llm.aclose = AsyncMock()
        old_emb = MagicMock()
        old_emb.aclose = AsyncMock()

        app.state.llm_provider = old_llm
        app.state.embedding_provider = old_emb

        with (
            patch("src.llm.factory.build_llm_provider", return_value=MagicMock()),
            patch("src.llm.factory.build_embedding_provider", return_value=MagicMock()),
        ):
            build_runtime(app)

        old_llm.aclose.assert_called_once()
        old_emb.aclose.assert_called_once()

    def test_build_runtime_handles_close_failure(self):
        """When old provider's aclose() raises synchronously, new provider is still assigned."""
        from src.main import app, build_runtime

        _setup_minimal_app_state(app)

        old_llm = MagicMock()
        old_llm.aclose.side_effect = RuntimeError("close boom")

        app.state.llm_provider = old_llm
        app.state.embedding_provider = None

        new_llm = MagicMock()
        new_emb = MagicMock()

        with (
            patch("src.llm.factory.build_llm_provider", return_value=new_llm),
            patch("src.llm.factory.build_embedding_provider", return_value=new_emb),
            patch("src.main.logger.warning") as mock_warning,
        ):
            build_runtime(app)

        assert app.state.llm_provider is new_llm
        assert app.state.embedding_provider is new_emb
        mock_warning.assert_called_once_with(
            "build_runtime.old_llm_close_failed", exc_info=True
        )

    def test_build_runtime_no_old_providers(self):
        """When app.state has no old providers, build_runtime() succeeds."""
        from src.main import app, build_runtime

        _setup_minimal_app_state(app)

        app.state.llm_provider = None
        app.state.embedding_provider = None

        new_llm = MagicMock()
        new_emb = MagicMock()

        with (
            patch("src.llm.factory.build_llm_provider", return_value=new_llm),
            patch("src.llm.factory.build_embedding_provider", return_value=new_emb),
        ):
            build_runtime(app)

        assert app.state.llm_provider is new_llm
        assert app.state.embedding_provider is new_emb

    def test_build_runtime_same_provider_not_double_closed(self):
        """Same provider on both llm_provider and embedding_provider: aclose() called once."""
        from src.main import app, build_runtime

        _setup_minimal_app_state(app)

        shared = MagicMock()
        shared.aclose = AsyncMock()

        app.state.llm_provider = shared
        app.state.embedding_provider = shared

        with (
            patch("src.llm.factory.build_llm_provider", return_value=MagicMock()),
            patch("src.llm.factory.build_embedding_provider", return_value=MagicMock()),
        ):
            build_runtime(app)

        shared.aclose.assert_called_once()
