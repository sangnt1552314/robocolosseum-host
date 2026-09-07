"""Authentication tests."""

from __future__ import annotations

from .conftest import auth_headers


def test_missing_token_is_401(client):
    resp = client.get("/health")
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_wrong_token_is_401(client):
    resp = client.get("/health", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_malformed_header_is_401(client):
    resp = client.get("/health", headers={"Authorization": "Basic abc"})
    assert resp.status_code == 401


def test_valid_token_allows_health(client):
    resp = client.get("/health", headers=auth_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["slurm_available"] is True


def test_missing_env_token_is_server_error(config, backend, monkeypatch):
    from fastapi.testclient import TestClient

    from slurm_job_launcher_server.app import create_app

    monkeypatch.delenv("LAUNCHER_API_TOKEN", raising=False)
    app = create_app(config=config, backend=backend)
    client = TestClient(app)
    resp = client.get("/health", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 500
    assert resp.json()["error"] == "server_not_configured"
