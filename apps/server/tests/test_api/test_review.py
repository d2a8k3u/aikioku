"""Tests for Review API endpoints."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import httpx

from src.models.card import Card, CardStatus, CardType
from src.models.note import Note


@pytest.fixture
def sample_note() -> Note:
    return Note(
        id="12345678-1234-5678-1234-567812345678",
        title="Python Basics",
        content="Python is a high-level programming language created by Guido van Rossum.",
        frontmatter={"tags": ["python"]},
        links=[],
        path="/notes/python-basics.md",
    )


@pytest.fixture
def temp_db_path() -> str:
    d = tempfile.mkdtemp()
    yield str(Path(d) / "test_cards.db")


@pytest.fixture
def note_store(temp_db_path: str):
    from src.storage.note_store import NoteStore
    notes_dir = str(Path(temp_db_path).parent / "notes")
    return NoteStore(notes_dir=notes_dir)


class TestGetDueCards:
    """Test GET /api/review/due endpoint."""

    def test_returns_due_cards(self, temp_db_path, note_store):
        now = datetime.utcnow()
        due_cards = [
            Card(
                id="card-1",
                note_id="note-1",
                type=CardType.qa,
                front="Due question",
                back="Due answer",
                next_review=now,
                status=CardStatus.review,
            )
        ]
        with patch("src.api.review._get_spaced_repetition") as mock_get_sr:
            mock_sr = MagicMock()
            mock_sr.get_due_cards = AsyncMock(return_value=due_cards)
            mock_get_sr.return_value = mock_sr

            with patch("src.api.review.settings") as mock_settings:
                mock_settings.sqlite_db_path = temp_db_path
                from src.main import app
                cli = TestClient(app)
                response = cli.get("/api/review/due")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == "card-1"
        assert data[0]["front"] == "Due question"


class TestCreateCards:
    """Test POST /api/review/cards endpoint."""

    def test_creates_cards_from_note(self, temp_db_path, note_store, sample_note):
        created_cards = [
            Card(
                id="card-1",
                note_id=sample_note.id,
                type=CardType.qa,
                front="Who created Python?",
                back="Guido van Rossum",
                next_review=datetime.utcnow(),
                status=CardStatus.new,
            ),
            Card(
                id="card-2",
                note_id=sample_note.id,
                type=CardType.cloze,
                front="Python was created by {{Guido van Rossum}}.",
                back="Python was created by Guido van Rossum.",
                next_review=datetime.utcnow(),
                status=CardStatus.new,
            ),
        ]
        with patch("src.api.review._get_note_store") as mock_get_store, \
             patch("src.api.review._get_spaced_repetition") as mock_get_sr:
            mock_get_store.return_value = note_store
            note_store.get = MagicMock(return_value=sample_note)
            mock_sr = MagicMock()
            mock_sr.generate_cards = AsyncMock(return_value=created_cards)
            mock_get_sr.return_value = mock_sr

            with patch("src.api.review.settings") as mock_settings:
                mock_settings.sqlite_db_path = temp_db_path
                from src.main import app
                cli = TestClient(app)
                response = cli.post("/api/review/cards", json={"note_id": sample_note.id})

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["front"] == "Who created Python?"
        assert data[1]["front"] == "Python was created by {{Guido van Rossum}}."


class TestReviewCard:
    """Test POST /api/review/cards/{card_id}/review endpoint."""

    def test_rating_3_updates_interval(self, temp_db_path):
        now = datetime.utcnow()
        reviewed_card = Card(
            id="card-1",
            note_id="note-1",
            type=CardType.qa,
            front="Q",
            back="A",
            ease_factor=2.5,
            interval=10,
            repetitions=3,
            next_review=now,
            status=CardStatus.review,
        )
        with patch("src.api.review._get_spaced_repetition") as mock_get_sr:
            mock_sr = MagicMock()
            mock_sr.review_card = AsyncMock(return_value=reviewed_card)
            # The endpoint awaits get_stats() in its broadcast path.
            mock_sr.get_stats = AsyncMock(return_value={"due": 0, "total": 0})
            mock_get_sr.return_value = mock_sr

            with patch("src.api.review.settings") as mock_settings:
                mock_settings.sqlite_db_path = temp_db_path
                from src.main import app
                cli = TestClient(app)
                response = cli.post(
                    "/api/review/cards/card-1/review",
                    json={"rating": 3},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "card-1"
        assert data["interval"] == 10
        assert data["repetitions"] == 3
        mock_sr.review_card.assert_called_once_with("card-1", 3)

    def test_rating_1_resets(self, temp_db_path):
        now = datetime.utcnow()
        reviewed_card = Card(
            id="card-1",
            note_id="note-1",
            type=CardType.qa,
            front="Q",
            back="A",
            ease_factor=2.5,
            interval=1,
            repetitions=0,
            next_review=now,
            status=CardStatus.learning,
        )
        with patch("src.api.review._get_spaced_repetition") as mock_get_sr:
            mock_sr = MagicMock()
            mock_sr.review_card = AsyncMock(return_value=reviewed_card)
            # The endpoint awaits get_stats() in its broadcast path.
            mock_sr.get_stats = AsyncMock(return_value={"due": 0, "total": 0})
            mock_get_sr.return_value = mock_sr

            with patch("src.api.review.settings") as mock_settings:
                mock_settings.sqlite_db_path = temp_db_path
                from src.main import app
                cli = TestClient(app)
                response = cli.post(
                    "/api/review/cards/card-1/review",
                    json={"rating": 1},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "card-1"
        assert data["interval"] == 1
        assert data["repetitions"] == 0
        assert data["status"] == "learning"
        mock_sr.review_card.assert_called_once_with("card-1", 1)


class TestSpacedRepetitionProviderInjection:
    """_get_spaced_repetition must use app.state.llm_provider, never localhost."""

    def test_uses_app_state_llm_provider(self, temp_db_path):
        from src.main import app
        from src.api.review import _get_spaced_repetition

        fake_provider = MagicMock(name="fake_llm_provider")
        app.state.llm_provider = fake_provider
        # Ensure no cached instance.
        if hasattr(app.state, "spaced_repetition"):
            delattr(app.state, "spaced_repetition")

        try:
            with patch("src.api.review.settings") as mock_settings:
                mock_settings.sqlite_db_path = temp_db_path
                mock_settings.notes_dir = str(Path(temp_db_path).parent)
                # If a bare OllamaProvider were constructed, importing it would be
                # the only way; assert the SR uses our injected provider instead.
                request = MagicMock()
                request.app = app
                sr = _get_spaced_repetition(request)
        finally:
            if hasattr(app.state, "spaced_repetition"):
                delattr(app.state, "spaced_repetition")
            if hasattr(app.state, "llm_provider"):
                delattr(app.state, "llm_provider")

        assert sr._llm is fake_provider

    def test_missing_provider_raises_503(self, temp_db_path):
        from fastapi import HTTPException

        from src.main import app
        from src.api.review import _get_spaced_repetition

        if hasattr(app.state, "spaced_repetition"):
            delattr(app.state, "spaced_repetition")
        if hasattr(app.state, "llm_provider"):
            delattr(app.state, "llm_provider")

        with patch("src.api.review.settings") as mock_settings:
            mock_settings.sqlite_db_path = temp_db_path
            mock_settings.notes_dir = str(Path(temp_db_path).parent)
            request = MagicMock()
            request.app = app
            request.app.state = app.state
            with pytest.raises(HTTPException) as exc_info:
                _get_spaced_repetition(request)

        assert exc_info.value.status_code == 503


class TestGenerateCardsGracefulErrors:
    """POST /api/review/cards maps LLM-output / network failures to 502 / 503."""

    def _client_with_provider(self, temp_db_path, note_store, sample_note, sr):
        from src.main import app

        cli_ctx = patch("src.api.review._get_note_store", return_value=note_store)
        sr_ctx = patch("src.api.review._get_spaced_repetition", return_value=sr)
        return cli_ctx, sr_ctx, app

    def test_fenced_json_returns_200(self, temp_db_path, note_store, sample_note):
        from src.augmentation.spaced_repetition import SpacedRepetition

        note_store.get = MagicMock(return_value=sample_note)
        provider = MagicMock()
        provider.complete = AsyncMock(
            return_value=(
                "```json\n"
                '[{"type": "qa", "front": "Who created Python?", '
                '"back": "Guido van Rossum"}]\n```'
            )
        )
        sr = SpacedRepetition(
            note_store=note_store, llm_provider=provider, db_path=temp_db_path
        )

        with patch("src.api.review._get_note_store", return_value=note_store), \
             patch("src.api.review._get_spaced_repetition", return_value=sr):
            from src.main import app
            cli = TestClient(app)
            response = cli.post("/api/review/cards", json={"note_id": sample_note.id})

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["front"] == "Who created Python?"

    def test_garbage_returns_502(self, temp_db_path, note_store, sample_note):
        from src.augmentation.spaced_repetition import SpacedRepetition

        note_store.get = MagicMock(return_value=sample_note)
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="I cannot make cards.")
        sr = SpacedRepetition(
            note_store=note_store, llm_provider=provider, db_path=temp_db_path
        )

        with patch("src.api.review._get_note_store", return_value=note_store), \
             patch("src.api.review._get_spaced_repetition", return_value=sr):
            from src.main import app
            cli = TestClient(app)
            response = cli.post("/api/review/cards", json={"note_id": sample_note.id})

        assert response.status_code == 502

    def test_connect_error_returns_503(self, temp_db_path, note_store, sample_note):
        note_store.get = MagicMock(return_value=sample_note)
        mock_sr = MagicMock()
        mock_sr.generate_cards = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with patch("src.api.review._get_note_store", return_value=note_store), \
             patch("src.api.review._get_spaced_repetition", return_value=mock_sr):
            from src.main import app
            cli = TestClient(app)
            response = cli.post("/api/review/cards", json={"note_id": sample_note.id})

        assert response.status_code == 503

    def test_missing_note_returns_404(self, temp_db_path, note_store):
        note_store.get = MagicMock(return_value=None)
        mock_sr = MagicMock()
        mock_sr.generate_cards = AsyncMock()

        with patch("src.api.review._get_note_store", return_value=note_store), \
             patch("src.api.review._get_spaced_repetition", return_value=mock_sr):
            from src.main import app
            cli = TestClient(app)
            response = cli.post("/api/review/cards", json={"note_id": "missing-id"})

        assert response.status_code == 404


class TestGetStats:
    """Test GET /api/review/stats endpoint."""

    def test_returns_correct_counts(self, temp_db_path):
        with patch("src.api.review._get_spaced_repetition") as mock_get_sr:
            mock_sr = MagicMock()
            mock_sr.get_stats = AsyncMock(return_value={
                "total": 10,
                "due": 3,
                "new": 2,
                "learning": 1,
                "review": 6,
            })
            mock_sr.get_suspended_count = AsyncMock(return_value=0)
            mock_get_sr.return_value = mock_sr

            with patch("src.api.review.settings") as mock_settings:
                mock_settings.sqlite_db_path = temp_db_path
                from src.main import app
                with TestClient(app) as client:
                    response = client.get("/api/review/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 10
        assert data["due"] == 3
        assert data["new"] == 2
        assert data["learning"] == 1
        assert data["review"] == 6
        assert data["suspended"] == 0
