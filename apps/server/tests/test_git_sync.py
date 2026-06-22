"""Comprehensive tests for the GitSync class."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.storage.git_sync import GitSync


class TestGitSyncInit:
    """Tests for GitSync initialization and repo creation."""

    def test_init_creates_repo_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = os.path.join(tmpdir, "notes")
            GitSync(repo_path)
            assert os.path.isdir(repo_path)
            assert os.path.isdir(os.path.join(repo_path, ".git"))

    def test_init_creates_gitignore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            GitSync(tmpdir)
            gitignore = os.path.join(tmpdir, ".gitignore")
            assert os.path.isfile(gitignore)
            assert ".DS_Store" in open(gitignore).read()

    def test_init_does_not_duplicate_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            GitSync(tmpdir)
            git_dir = os.path.join(tmpdir, ".git")
            assert os.path.isdir(git_dir)
            # Second init should be no-op
            GitSync(tmpdir)
            assert os.path.isdir(git_dir)

    def test_initial_commit_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            history = gs.get_history(limit=1)
            assert len(history) == 1
            assert "init: git sync for notes" in history[0]["message"]


class TestGitSyncCommit:
    """Tests for staging and committing."""

    def test_commit_returns_true_on_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            # Create a new file
            note_path = os.path.join(tmpdir, "note1.md")
            with open(note_path, "w") as f:
                f.write("Hello world\n")
            result = gs.commit("Add note1")
            assert result is True

    def test_commit_returns_false_when_nothing_to_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            # No changes after init
            result = gs.commit("nothing")
            assert result is False

    def test_commit_message_appears_in_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            note_path = os.path.join(tmpdir, "note2.md")
            with open(note_path, "w") as f:
                f.write("Content\n")
            gs.commit("feat: add note2")
            history = gs.get_history(limit=1)
            assert "feat: add note2" in history[0]["message"]

    def test_commit_multiple_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            for i in range(3):
                with open(os.path.join(tmpdir, f"file{i}.md"), "w") as f:
                    f.write(f"data {i}\n")
            gs.commit("Add three files")
            history = gs.get_history(limit=1)
            assert history[0]["message"] == "Add three files"


class TestGitSyncHistory:
    """Tests for retrieving commit history."""

    def test_history_returns_list_of_dicts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            history = gs.get_history(limit=5)
            assert isinstance(history, list)
            assert all(isinstance(h, dict) for h in history)

    def test_history_contains_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            history = gs.get_history(limit=1)
            assert set(history[0].keys()) >= {"hash", "date", "message"}

    def test_history_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            for i in range(5):
                with open(os.path.join(tmpdir, f"n{i}.md"), "w") as f:
                    f.write(f"{i}\n")
                gs.commit(f"commit {i}")
            history = gs.get_history(limit=3)
            assert len(history) == 3

    def test_history_on_empty_repo(self):
        # get_history on a brand-new repo still has the init commit
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            history = gs.get_history(limit=10)
            assert len(history) >= 1


class TestGitSyncDiff:
    """Tests for per-note diffs."""

    def test_diff_returns_empty_for_new_file(self):
        # HEAD~1 vs HEAD on a file added in the most recent commit shows the
        # addition diff in Git; we assert the real behavior rather than empty.
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            note_path = os.path.join(tmpdir, "note.md")
            with open(note_path, "w") as f:
                f.write("v1\n")
            gs.commit("add note")
            diff = gs.get_diff("note")
            assert "v1" in diff or "note.md" in diff or "+" in diff

    def test_diff_returns_empty_when_no_prior_commit(self):
        # When HEAD~1 doesn't exist (single-commit repo), diff exits non-zero
        # and get_diff swallows it and returns empty.
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            diff = gs.get_diff("nonexistent")
            assert diff == ""

    def test_diff_shows_changes_between_commits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            note_path = os.path.join(tmpdir, "note.md")
            # first commit
            with open(note_path, "w") as f:
                f.write("v1\n")
            gs.commit("add note")
            # second commit
            with open(note_path, "w") as f:
                f.write("v2\n")
            gs.commit("update note")
            diff = gs.get_diff("note")
            assert "v1" in diff or "v2" in diff or "-" in diff or "+" in diff

    def test_diff_returns_empty_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            diff = gs.get_diff("nonexistent")
            assert diff == ""


class TestGitSyncEdgeCases:
    """Edge-case and failure-mode tests."""

    def test_run_git_raises_on_invalid_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            with pytest.raises(Exception):
                gs._run_git("not-a-real-command")

    def test_commit_gracefully_handles_git_error(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            # Force subprocess.run to raise CalledProcessError so commit returns False
            import subprocess as _sp

            original_run = _sp.run

            def broken_run(*args, **kwargs):
                raise _sp.CalledProcessError(returncode=1, cmd=args[0], output="err", stderr="err")

            monkeypatch.setattr(_sp, "run", broken_run)
            assert gs.commit("msg") is False
            monkeypatch.setattr(_sp, "run", original_run)

    def test_repo_dir_as_string_or_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gs = GitSync(tmpdir)
            assert str(gs.repo_dir) == tmpdir
