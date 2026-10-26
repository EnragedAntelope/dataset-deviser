"""The launch settings that keep a localhost, no-auth app from leaking files."""
from __future__ import annotations

from fastapi.testclient import TestClient
from gradio.routes import App

import app as A


def _client() -> TestClient:
    kw = A.LAUNCH_KWARGS
    A.demo.allowed_paths = kw["allowed_paths"]
    A.demo.blocked_paths = kw["blocked_paths"]
    return TestClient(App.create_app(A.demo, app_kwargs=dict(kw["app_kwargs"])))


def test_binds_to_localhost_only() -> None:
    assert A.LAUNCH_KWARGS["server_name"] == "127.0.0.1"
    assert "share" not in A.LAUNCH_KWARGS
    assert A.LAUNCH_KWARGS["run_history"] is False


def test_foreign_host_header_is_refused() -> None:
    # DNS rebinding: a page on evil.example re-pointed at 127.0.0.1 sends its own Host.
    c = _client()
    assert c.get("/config", headers={"host": "evil.example:7861"}).status_code == 400
    assert c.get("/config", headers={"host": "127.0.0.1:7861"}).status_code == 200
    assert c.get("/config", headers={"host": "localhost:7861"}).status_code == 200


def test_env_file_is_never_served() -> None:
    env = A.REPO_ROOT / ".env"
    r = _client().get(f"/gradio_api/file={env}", headers={"host": "127.0.0.1"})
    assert r.status_code in (403, 404)
