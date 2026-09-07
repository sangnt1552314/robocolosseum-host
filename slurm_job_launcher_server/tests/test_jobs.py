"""API-level job tests -- all Slurm CLIs are mocked (no real submissions)."""

from __future__ import annotations

from .conftest import auth_headers


def test_unsupported_model_is_422(client):
    resp = client.post("/jobs", json={"model": "does-not-exist"}, headers=auth_headers())
    assert resp.status_code == 422
    assert resp.json()["error"] == "unsupported_model"


def test_submit_returns_202_and_job_id(client, fake_slurm):
    resp = client.post("/jobs", json={"model": "molmoact2-droid"}, headers=auth_headers())
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] == "825501"
    assert body["model"] == "molmoact2-droid"
    assert body["state"] == "PENDING"


def test_multiple_jobs_allowed_below_limits(client):
    for _ in range(3):
        resp = client.post("/jobs", json={"model": "molmoact2-droid"}, headers=auth_headers())
        assert resp.status_code == 202
    listed = client.get("/jobs", headers=auth_headers()).json()["jobs"]
    assert len(listed) == 3


def test_global_limit_enforced(client, config, fake_slurm):
    config_max = config.max_active_jobs
    # Pre-fill with unrelated-but-owned policy jobs up to the global cap.
    for _ in range(config_max):
        fake_slurm.add_job(name="colosseum-molmo", user="testuser")
    resp = client.post("/jobs", json={"model": "molmoact2-droid"}, headers=auth_headers())
    assert resp.status_code == 409
    assert resp.json()["error"] == "global_limit_reached"


def test_per_model_replica_limit_enforced(client, fake_slurm):
    # max_replicas is 3 in the test config.
    for _ in range(3):
        fake_slurm.add_job(name="colosseum-molmo", user="testuser")
    resp = client.post("/jobs", json={"model": "molmoact2-droid"}, headers=auth_headers())
    assert resp.status_code == 429
    assert resp.json()["error"] == "model_replica_limit_reached"


def test_request_id_is_idempotent(client, fake_slurm):
    first = client.post(
        "/jobs",
        json={"model": "molmoact2-droid", "request_id": "match-123"},
        headers=auth_headers(),
    )
    assert first.status_code == 202
    job_id = first.json()["job_id"]

    second = client.post(
        "/jobs",
        json={"model": "molmoact2-droid", "request_id": "match-123"},
        headers=auth_headers(),
    )
    assert second.status_code == 202
    assert second.json()["job_id"] == job_id
    # only one real submission happened
    assert len(fake_slurm.submitted) == 1


def test_status_endpoint(client):
    submit = client.post("/jobs", json={"model": "molmoact2-droid"}, headers=auth_headers())
    job_id = submit.json()["job_id"]
    resp = client.get(f"/jobs/{job_id}", headers=auth_headers())
    assert resp.status_code == 200
    assert resp.json()["job_id"] == job_id


def test_status_unknown_job_is_404(client):
    resp = client.get("/jobs/999999", headers=auth_headers())
    assert resp.status_code == 404
    assert resp.json()["error"] == "job_not_found"


def test_cancel_owned_policy_job(client, fake_slurm):
    submit = client.post("/jobs", json={"model": "molmoact2-droid"}, headers=auth_headers())
    job_id = submit.json()["job_id"]
    resp = client.post(f"/jobs/{job_id}/cancel", headers=auth_headers())
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancel_requested"
    assert job_id in fake_slurm.cancelled


def test_cancel_refuses_non_launcher_job(client, fake_slurm):
    jid = fake_slurm.add_job(name="some-other-job", user="testuser")
    resp = client.post(f"/jobs/{jid}/cancel", headers=auth_headers())
    assert resp.status_code in (403, 404)
    assert jid not in fake_slurm.cancelled


def test_slurm_error_returns_sanitized_502(client, fake_slurm):
    fake_slurm.sbatch_fails = True
    resp = client.post("/jobs", json={"model": "molmoact2-droid"}, headers=auth_headers())
    assert resp.status_code == 502
    assert resp.json()["error"] == "slurm_error"
    assert "sbatch: error" not in resp.text
