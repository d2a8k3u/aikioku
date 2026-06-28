"""GitSync: Git-based version control for notes."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitSync:
    """Git-based version control for the notes directory."""

    # Patterns that must never be versioned: derived caches that live in the
    # notes dir (e.g. the NoteStore metadata index). The index is rebuildable
    # from the markdown files, so committing it would only add churn/bloat.
    _IGNORE_PATTERNS = (".DS_Store", ".note_index.db", ".note_index.db-*")

    def __init__(self, repo_dir: str) -> None:
        """Initialize git repo if needed."""
        self.repo_dir = Path(repo_dir)
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_repo()

    def _ensure_repo(self) -> None:
        """Initialize git repo if .git doesn't exist."""
        git_dir = self.repo_dir / ".git"
        if not git_dir.exists():
            self._run_git("init")
            # Configure identity so commits work in containers/CI without global git config
            self._run_git("config", "user.email", "git-sync@aikioku.local")
            self._run_git("config", "user.name", "Git Sync")
            # Create .gitignore
            gitignore = self.repo_dir / ".gitignore"
            gitignore.write_text("\n".join(self._IGNORE_PATTERNS) + "\n", encoding="utf-8")
            self._run_git("add", ".gitignore")
            self._run_git("commit", "-m", "init: git sync for notes")
        else:
            self._ensure_gitignore_patterns()

    def _ensure_gitignore_patterns(self) -> None:
        """Idempotently ensure derived-cache patterns are gitignored.

        Handles pre-existing repos created before the index cache existed, so a
        later ``git add -A`` never stages the index file.
        """
        gitignore = self.repo_dir / ".gitignore"
        try:
            existing = (
                gitignore.read_text(encoding="utf-8").splitlines() if gitignore.exists() else []
            )
        except OSError:
            return
        present = {line.strip() for line in existing}
        missing = [p for p in self._IGNORE_PATTERNS if p not in present]
        if not missing:
            return
        lines = existing + missing
        gitignore.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _run_git(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a git command in the repo directory."""
        return subprocess.run(
            ["git", *args],
            cwd=str(self.repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )

    def commit(self, message: str) -> bool:
        """Stage all changes and commit. Returns True if committed, False if nothing to commit."""
        try:
            # Check if there are changes
            status = self._run_git("status", "--porcelain")
            if not status.stdout.strip():
                return False
            self._run_git("add", "-A")
            self._run_git("commit", "-m", message)
            return True
        except subprocess.CalledProcessError:
            return False

    def get_history(self, limit: int = 50) -> list[dict[str, str]]:
        """Get commit history. Returns list of {hash, date, message}."""
        try:
            result = self._run_git(
                "log",
                f"--max-count={limit}",
                "--format=%H|%aI|%s",
            )
            history = []
            for line in result.stdout.strip().split("\n"):
                if "|" not in line:
                    continue
                parts = line.split("|", 2)
                if len(parts) == 3:
                    history.append(
                        {
                            "hash": parts[0],
                            "date": parts[1],
                            "message": parts[2],
                        }
                    )
            return history
        except subprocess.CalledProcessError:
            return []

    def get_diff(self, note_id: str) -> str:
        """Show the latest diff for a specific note file."""
        file_path = f"{note_id}.md"
        try:
            result = self._run_git("diff", "HEAD~1", "HEAD", "--", file_path)
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    def get_note_history(self, note_id: str, limit: int = 50) -> list[dict[str, str]]:
        """Get commit history that touched a specific note file."""
        file_path = f"{note_id}.md"
        try:
            result = self._run_git(
                "log",
                f"--max-count={limit}",
                "--format=%H|%aI|%s",
                "--",
                file_path,
            )
            history = []
            for line in result.stdout.strip().split("\n"):
                if "|" not in line:
                    continue
                parts = line.split("|", 2)
                if len(parts) == 3:
                    history.append(
                        {
                            "hash": parts[0],
                            "date": parts[1],
                            "message": parts[2],
                        }
                    )
            return history
        except subprocess.CalledProcessError:
            return []

    def get_note_versions(self, note_id: str, limit: int = 50) -> list[dict[str, str]]:
        """Return the content of a note at each commit (HEAD, HEAD~1, ...).
        This is expensive; use with small limits."""
        file_path = f"{note_id}.md"
        versions = []
        try:
            # Get commit hashes that touched the file
            log_result = self._run_git(
                "log",
                f"--max-count={limit}",
                "--format=%H",
                "--",
                file_path,
            )
            hashes = [h for h in log_result.stdout.strip().split("\n") if h]
            for commit_hash in hashes:
                try:
                    show_result = self._run_git("show", f"{commit_hash}:{file_path}")
                    versions.append(
                        {
                            "commit": commit_hash,
                            "content": show_result.stdout,
                        }
                    )
                except subprocess.CalledProcessError:
                    continue
            return versions
        except subprocess.CalledProcessError:
            return []
