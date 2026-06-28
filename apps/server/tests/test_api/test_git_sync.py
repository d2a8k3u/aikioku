"""Tests for Git Sync API endpoints."""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI TestClient with a fresh GitSync in a temp repo."""
    from src.main import app

    with tempfile.TemporaryDirectory() as tmpdir:
        from src.storage.git_sync import GitSync

        gs = GitSync(tmpdir)
        app.state.git_sync = gs
        yield TestClient(app), tmpdir


class TestGitCommit:
    def test_commit_with_changes(self, client):
        cli, tmpdir = client
        note_path = os.path.join(tmpdir, "note1.md")
        with open(note_path, "w") as f:
            f.write("Hello\n")
        response = cli.post("/api/sync/git/commit?message=Add+note1")
        assert response.status_code == 200
        data = response.json()
        assert data["committed"] is True
        assert data["message"] == "Add note1"

    def test_commit_without_changes(self, client):
        cli, tmpdir = client
        response = cli.post("/api/sync/git/commit?message=Nothing+to+do")
        assert response.status_code == 200
        data = response.json()
        assert data["committed"] is False

    def test_missing_message_returns_422(self, client):
        cli, tmpdir = client
        response = cli.post("/api/sync/git/commit")
        assert response.status_code == 422


class TestGitHistory:
    def test_returns_list(self, client):
        cli, tmpdir = client
        response = cli.get("/api/sync/git/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert set(data[0].keys()) >= {"hash", "date", "message"}

    def test_limit_param(self, client):
        cli, tmpdir = client
        # Add a few commits
        for i in range(3):
            with open(os.path.join(tmpdir, f"f{i}.md"), "w") as f:
                f.write(f"{i}\n")
            cli.post(f"/api/sync/git/commit?message=commit+{i}")
        response = cli.get("/api/sync/git/history?limit=2")
        data = response.json()
        assert len(data) == 2

    def test_history_contains_commits(self, client):
        cli, tmpdir = client
        with open(os.path.join(tmpdir, "note.md"), "w") as f:
            f.write("v1\n")
        cli.post("/api/sync/git/commit?message=feat:+add+note")
        response = cli.get("/api/sync/git/history?limit=1")
        data = response.json()
        assert data[0]["message"] == "feat: add note"


class TestGitDiff:
    def test_diff_for_new_file(self, client):
        cli, tmpdir = client
        with open(os.path.join(tmpdir, "note.md"), "w") as f:
            f.write("v1\n")
        cli.post("/api/sync/git/commit?message=add+note")
        response = cli.get("/api/sync/git/diff/note")
        assert response.status_code == 200
        data = response.json()
        assert data["note_id"] == "note"
        assert "diff" in data
        # diff may contain text because file was added in last commit
        assert isinstance(data["diff"], str)

    def test_diff_empty_for_no_history(self, client):
        cli, tmpdir = client
        response = cli.get("/api/sync/git/diff/nonexistent")
        assert response.status_code == 200
        data = response.json()
        assert data["diff"] == ""

    def test_diff_shows_changes(self, client):
        cli, tmpdir = client
        note_path = os.path.join(tmpdir, "note.md")
        with open(note_path, "w") as f:
            f.write("v1\n")
        cli.post("/api/sync/git/commit?message=add+note")
        with open(note_path, "w") as f:
            f.write("v2\n")
        cli.post("/api/sync/git/commit?message=update+note")
        response = cli.get("/api/sync/git/diff/note")
        data = response.json()
        assert (
            "v1" in data["diff"]
            or "v2" in data["diff"]
            or "-" in data["diff"]
            or "+" in data["diff"]
        )
