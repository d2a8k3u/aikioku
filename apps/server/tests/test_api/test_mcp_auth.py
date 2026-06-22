"""Tests for the mounted MCP surface: PAT gating + tool scope enforcement.

The MCP streamable-HTTP session manager can only be ``.run()`` once per instance
(and ``mcp`` is a module singleton built at import), so all tests share a single
module-scoped TestClient whose lifespan runs once. Its lifespan is patched onto a
temp DB so it never opens a second Kuzu writer on the live ``/data`` graph. Each
test still creates its tokens under the per-test temp DB installed by the autouse
``reset_app_state`` fixture; the request middleware reads that same path at call
time, so token create + verify stay consistent within a test.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src import access_tokens

_MCP = "/mcp/"  # trailing slash served by the mount; bare /mcp served by a shim (no redirect)
_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


@pytest.fixture(scope="module")
def mcp_client(tmp_path_factory):
    from src.config import settings
    from src.main import app

    d = tmp_path_factory.mktemp("mcp_mod")
    with (
        patch.object(settings, "sqlite_db_path", str(d / "test.db")),
        patch.object(settings, "notes_dir", str(d)),
    ):
        # base_url sets the Host header to an allowed value: the MCP transport
        # has DNS-rebinding protection that rejects the default "testserver" host.
        with TestClient(app, base_url="http://127.0.0.1:8869") as client:
            yield client


def _tools_list() -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


def _tools_call(name: str, arguments: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def _result_text(resp_json: dict) -> str:
    return "".join(c.get("text", "") for c in resp_json["result"]["content"])


class TestMcpAuthGate:
    def test_missing_token_is_401(self, mcp_client):
        resp = mcp_client.post(_MCP, headers=_HEADERS, json=_tools_list())
        assert resp.status_code == 401

    def test_invalid_token_is_401(self, mcp_client):
        resp = mcp_client.post(
            _MCP,
            headers={**_HEADERS, "Authorization": "Bearer sbk_bogus"},
            json=_tools_list(),
        )
        assert resp.status_code == 401

    def test_valid_token_lists_tools(self, mcp_client):
        _, plaintext = access_tokens.create_token("ci", "alice", "full")
        resp = mcp_client.post(
            _MCP,
            headers={**_HEADERS, "Authorization": f"Bearer {plaintext}"},
            json=_tools_list(),
        )
        assert resp.status_code == 200
        names = {t["name"] for t in resp.json()["result"]["tools"]}
        assert {"search_notes", "hybrid_search", "create_note", "call_api"} <= names


class TestMcpBarePathNoRedirect:
    """The bare ``/mcp`` (no trailing slash) must serve directly, not 307-redirect.

    Most MCP clients are configured with the no-slash URL; a redirect cost every
    call an extra round-trip and doubled the request log.
    """

    def test_bare_path_serves_without_redirect(self, mcp_client):
        _, plaintext = access_tokens.create_token("ci", "alice", "full")
        resp = mcp_client.post(
            "/mcp",
            headers={**_HEADERS, "Authorization": f"Bearer {plaintext}"},
            json=_tools_list(),
            follow_redirects=False,
        )
        assert resp.status_code == 200, f"expected direct 200, got {resp.status_code}"
        names = {t["name"] for t in resp.json()["result"]["tools"]}
        assert "search_notes" in names

    def test_bare_path_still_pat_gated(self, mcp_client):
        resp = mcp_client.post("/mcp", headers=_HEADERS, json=_tools_list(), follow_redirects=False)
        assert resp.status_code == 401


class TestMcpScopeEnforcement:
    def test_read_only_token_cannot_write(self, mcp_client):
        _, plaintext = access_tokens.create_token("ci", "alice", "read")
        resp = mcp_client.post(
            _MCP,
            headers={**_HEADERS, "Authorization": f"Bearer {plaintext}"},
            json=_tools_call("create_note", {"title": "x", "content": "y"}),
        )
        body = resp.json()
        assert body["result"]["isError"] is True
        assert "read-only" in _result_text(body).lower()

    def test_call_api_denylist_blocks_token_routes(self, mcp_client):
        _, plaintext = access_tokens.create_token("ci", "alice", "full")
        resp = mcp_client.post(
            _MCP,
            headers={**_HEADERS, "Authorization": f"Bearer {plaintext}"},
            json=_tools_call("call_api", {"method": "GET", "path": "/api/settings/tokens/"}),
        )
        body = resp.json()
        assert body["result"]["isError"] is True
        assert "not permitted" in _result_text(body).lower()


class TestMcpCreateNoteSourceType:
    """MCP create_note hides notes by default; hidden=False makes them visible."""

    def test_mcp_create_note_defaults_to_hidden(self, mcp_client):
        _, plaintext = access_tokens.create_token("ci", "alice", "full")
        resp = mcp_client.post(
            _MCP,
            headers={**_HEADERS, "Authorization": f"Bearer {plaintext}"},
            json=_tools_call("create_note", {"title": "MCP Note", "content": "MCP body"}),
        )
        body = resp.json()
        assert not body["result"].get("isError"), f"unexpected error: {_result_text(body)}"
        import json

        note = json.loads(_result_text(body))
        assert note["source_type"] == "hidden", f"expected 'hidden', got {note.get('source_type')}"

    def test_mcp_create_note_visible_when_hidden_false(self, mcp_client):
        _, plaintext = access_tokens.create_token("ci", "alice", "full")
        resp = mcp_client.post(
            _MCP,
            headers={**_HEADERS, "Authorization": f"Bearer {plaintext}"},
            json=_tools_call(
                "create_note",
                {"title": "Visible Note", "content": "Visible body", "hidden": False},
            ),
        )
        body = resp.json()
        assert not body["result"].get("isError"), f"unexpected error: {_result_text(body)}"
        import json

        note = json.loads(_result_text(body))
        assert note["source_type"] == "note", f"expected 'note', got {note.get('source_type')}"


class TestMcpCreateMemoryHiddenNote:
    """create_memory stores a hidden, fully-processed note (retrievable, not listed)."""

    def test_create_memory_creates_hidden_note(self, mcp_client):
        import json

        _, plaintext = access_tokens.create_token("ci", "alice", "full")
        text = "Paris is the capital of France"
        resp = mcp_client.post(
            _MCP,
            headers={**_HEADERS, "Authorization": f"Bearer {plaintext}"},
            json=_tools_call("create_memory", {"text": text}),
        )
        body = resp.json()
        assert not body["result"].get("isError"), f"unexpected error: {_result_text(body)}"
        note = json.loads(_result_text(body))
        assert note["source_type"] == "hidden", f"expected 'hidden', got {note.get('source_type')}"
        assert note["content"] == text
        assert note["title"] == text  # short single-line text becomes the title verbatim

    def test_read_only_token_cannot_create_memory(self, mcp_client):
        _, plaintext = access_tokens.create_token("ci", "alice", "read")
        resp = mcp_client.post(
            _MCP,
            headers={**_HEADERS, "Authorization": f"Bearer {plaintext}"},
            json=_tools_call("create_memory", {"text": "x"}),
        )
        body = resp.json()
        assert body["result"]["isError"] is True
        assert "read-only" in _result_text(body).lower()
