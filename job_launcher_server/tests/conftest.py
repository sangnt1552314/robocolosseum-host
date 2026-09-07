"""Shared test fixtures. All Slurm calls are mocked -- no real jobs submitted."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from job_launcher_server.app import create_app
from job_launcher_server.config import LauncherConfig, PolicyEntry
from job_launcher_server.registry import PolicyRegistry
from job_launcher_server.slurm_backend import SlurmBackend

TOKEN = "test-token-value"


def make_config(tmp_path: Path, *, max_active_jobs: int = 10, max_replicas: int = 3) -> LauncherConfig:
    return LauncherConfig(
        host="127.0.0.1",
        port=8000,
        max_active_jobs=max_active_jobs,
        project_root=tmp_path,
        ngrok_bin="ngrok",
        policies={
            "molmoact2-droid": PolicyEntry(
                model="molmoact2-droid",
                script="slurm/molmoact2.sh",
                slurm_job_name="colosseum-molmo",
                max_replicas=max_replicas,
            )
        },
    )


class FakeSlurm:
    """Deterministic stand-in for the Slurm CLIs used by SlurmBackend.

    Holds an in-memory list of active jobs and records submit/cancel calls.
    """

    def __init__(self) -> None:
        self.jobs: list[dict] = []
        self.submitted: list[list[str]] = []
        self.cancelled: list[str] = []
        self.next_job_id = 825501
        self.sbatch_fails = False

    def add_job(self, *, job_id=None, name="colosseum-molmo", state="RUNNING",
                comment=None, user="testuser") -> str:
        job_id = str(job_id or self.next_job_id)
        self.next_job_id = int(job_id) + 1
        self.jobs.append(
            {
                "job_id": int(job_id),
                "name": name,
                "job_state": [state],
                "comment": comment,
                "user_name": user,
            }
        )
        return job_id

    def run(self, cmd, **kwargs):  # signature mirrors subprocess.run
        prog = Path(cmd[0]).name
        if prog == "sbatch":
            return self._sbatch(cmd)
        if prog == "squeue":
            if "--version" in cmd:
                return _cp(0, "slurm 26.05.2\n")
            return _cp(0, json.dumps({"jobs": self.jobs}))
        if prog == "sacct":
            job_id = cmd[cmd.index("--jobs") + 1]
            match = [j for j in self.jobs if str(j["job_id"]) == str(job_id)]
            return _cp(0, json.dumps({"jobs": match}))
        if prog == "scancel":
            self.cancelled.append(cmd[1])
            return _cp(0, "")
        raise AssertionError(f"unexpected command: {cmd}")

    def _sbatch(self, cmd):
        self.submitted.append(cmd)
        if self.sbatch_fails:
            return _cp(1, "", "sbatch: error: something failed")
        comment = None
        if "--comment" in cmd:
            comment = cmd[cmd.index("--comment") + 1]
        job_id = self.add_job(comment=comment, state="PENDING")
        return _cp(0, f"{job_id};soc\n")


def _cp(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def fake_slurm() -> FakeSlurm:
    return FakeSlurm()


@pytest.fixture
def config(tmp_path) -> LauncherConfig:
    return make_config(tmp_path)


@pytest.fixture
def backend(config, fake_slurm) -> SlurmBackend:
    registry = PolicyRegistry(config)
    return SlurmBackend(config, registry, user="testuser", runner=fake_slurm.run)


@pytest.fixture
def client(config, backend, monkeypatch) -> TestClient:
    monkeypatch.setenv("LAUNCHER_API_TOKEN", TOKEN)
    app = create_app(config=config, backend=backend)
    return TestClient(app)


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}
