"""A benign client disconnect mid-POST to /mcp must not be logged as a server error.

The MCP SDK's ``_handle_post_request`` reads the request body straight off the
live receive channel (``await request.body()``); if the client hangs up first
(cancellation, idle/LB timeout, network blip — all normal), Starlette raises
``ClientDisconnect``, which the SDK's broad ``except Exception`` turns into a full
ERROR traceback ("Error handling POST request") plus a doomed HTTP 500. The
``_MCPClientDisconnectGuard`` wrapper buffers the body up front and handles the
disconnect quietly, so the SDK's error path is never entered.

These drive the raw ASGI app (through the real middleware stack) because a
``TestClient`` cannot inject an ``http.disconnect`` mid-body. The autouse
``reset_app_state`` fixture points settings at a temp DB, so entering the lifespan
never opens a second Kuzu writer on the live ``/data`` graph.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable

from src import access_tokens

Message = dict
Receive = Callable[[], Awaitable[Message]]


def _scope(path: str, pat: str) -> dict:
    # host must be 127.0.0.1 — the streamable transport rejects other hosts
    # (DNS-rebinding protection).
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "scheme": "http",
        "server": ("127.0.0.1", 8869),
        "client": ("127.0.0.1", 54321),
        "headers": [
            (b"host", b"127.0.0.1:8869"),
            (b"content-type", b"application/json"),
            (b"accept", b"application/json, text/event-stream"),
            (b"authorization", b"Bearer " + pat.encode()),
        ],
    }


async def _drive(scope: dict, receive: Receive) -> list[Message]:
    """Run one request through the full ASGI app under a live lifespan."""
    from src.main import app  # import after reset_app_state patches settings

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    # One lifespan at a time: session_manager is a session-wide singleton (the
    # lifespan resets _has_started so re-entry re-runs its task group cleanly).
    async with app.router.lifespan_context(app):
        await app(scope, receive, send)
    return sent


_TOOLS_LIST = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()


async def test_midbody_disconnect_is_quiet(caplog: logging.LogRecord) -> None:
    _, pat = access_tokens.create_token("ci", "alice", "full")
    messages = iter(
        [
            {"type": "http.request", "body": _TOOLS_LIST[:6], "more_body": True},
            {"type": "http.disconnect"},
        ]
    )

    async def receive() -> Message:
        return next(messages)

    with caplog.at_level(logging.ERROR, logger="mcp.server.streamable_http"):
        sent = await _drive(_scope("/mcp/", pat), receive)

    assert "Error handling POST request" not in caplog.text
    statuses = [m["status"] for m in sent if m["type"] == "http.response.start"]
    assert 500 not in statuses  # a departed client must not yield a 500


async def test_normal_post_returns_jsonrpc() -> None:
    _, pat = access_tokens.create_token("ci", "alice", "full")
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": _TOOLS_LIST, "more_body": False}
        return {"type": "http.disconnect"}

    sent = await _drive(_scope("/mcp/", pat), receive)

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 200
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert "search_notes" in body.decode()
