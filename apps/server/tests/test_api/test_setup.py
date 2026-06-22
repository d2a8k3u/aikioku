"""Tests for the first-run setup wizard API and the central auth wall."""

from __future__ import annotations

from fastapi.testclient import TestClient

_PAYLOAD = {
    "llm_provider": "ollama",
    "llm_model": "llama3",
    "ollama_base_url": "http://localhost:11434",
    "embedding_dimension": 1024,
    "hf_api_key": "hf-secret",
    "account": {"username": "admin", "password": "supersecret1"},
}


def test_full_setup_flow():
    from src.main import app

    with TestClient(app) as c:
        # Fresh install -> unconfigured, runtime deferred.
        r = c.get("/api/setup/status")
        assert r.json() == {"configured": False, "auth_required": False}
        assert app.state.configured is False
        assert app.state.llm_provider is None

        # Run the wizard -> configured, runtime built without restart.
        r = c.post("/api/setup/", json=_PAYLOAD)
        assert r.status_code == 201, r.text
        assert app.state.configured is True
        assert app.state.llm_provider is not None
        assert app.state.hybrid_fusion is not None

        # Status reflects the auth wall now being on.
        assert c.get("/api/setup/status").json() == {
            "configured": True,
            "auth_required": True,
        }

        # Re-running setup is blocked.
        assert c.post("/api/setup/", json=_PAYLOAD).status_code == 409

        # Auth wall: protected route without a token -> 401.
        assert c.get("/api/settings/").status_code == 401

        # Login, then the protected route succeeds and leaks no secret values.
        r = c.post("/api/auth/login", data={"username": "admin", "password": "supersecret1"})
        assert r.status_code == 200, r.text
        token = r.json()["access_token"]
        r = c.get("/api/settings/", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert "hf_api_key" in r.json()["secret_keys"]
        assert "hf-secret" not in r.text


def test_setup_test_endpoint_unreachable_host():
    from src.main import app

    with TestClient(app) as c:
        r = c.post("/api/setup/test", json={
            "llm_provider": "ollama",
            "ollama_base_url": "http://127.0.0.1:1",
        })
        assert r.status_code == 200
        assert r.json()["ok"] is False
