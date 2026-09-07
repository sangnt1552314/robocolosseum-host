"""Slurm control-plane backend.

Only machine-readable Slurm interfaces are used (``sbatch --parsable``,
``squeue --json``, ``sacct --json``, ``scancel``). Human table output is never
parsed. All commands are run in list form -- never through a shell -- so an
HTTP caller can never inject arguments.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import LauncherConfig, PolicyEntry
from .registry import PolicyRegistry

logger = logging.getLogger("launcher.slurm")

# Slurm states that mean the job still occupies (or is queued for) resources.
_TERMINAL_STATES = {
    "COMPLETED",
    "CANCELLED",
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
    "SPECIAL_EXIT",
    "REVOKED",
}

_DEFAULT_TIMEOUT = 30.0


class SlurmError(Exception):
    """A Slurm command failed. The message is safe to surface (sanitized)."""


@dataclass
class JobInfo:
    job_id: str
    model: str | None
    state: str
    name: str
    comment: str | None


Runner = Callable[..., subprocess.CompletedProcess]


class SlurmBackend:
    def __init__(
        self,
        config: LauncherConfig,
        registry: PolicyRegistry,
        *,
        user: str | None = None,
        sbatch: str = "sbatch",
        squeue: str = "squeue",
        sacct: str = "sacct",
        scancel: str = "scancel",
        runner: Runner = subprocess.run,
    ) -> None:
        self._config = config
        self._registry = registry
        self._user = user or os.environ.get("USER") or ""
        self._sbatch = sbatch
        self._squeue = squeue
        self._sacct = sacct
        self._scancel = scancel
        self._run = runner

    # -- low-level -----------------------------------------------------------

    def _exec(self, cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
        try:
            result = self._run(
                cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=_DEFAULT_TIMEOUT,
                check=False,
            )
        except FileNotFoundError as exc:
            logger.error("Slurm command not found: %s", cmd[0])
            raise SlurmError("Slurm command not available.") from exc
        except subprocess.TimeoutExpired as exc:
            logger.error("Slurm command timed out: %s", cmd[0])
            raise SlurmError("Slurm command timed out.") from exc
        if result.returncode != 0:
            # Log the real stderr for operators (launcher logs only); return a
            # generic message to the client so paths/usernames never leak.
            logger.error("Command %s failed (rc=%s): %s", cmd[0], result.returncode, result.stderr)
            raise SlurmError(f"Slurm command '{cmd[0]}' failed.")
        return result

    # -- availability --------------------------------------------------------

    def available(self) -> bool:
        try:
            result = self._run(
                [self._squeue, "--version"],
                capture_output=True,
                text=True,
                timeout=_DEFAULT_TIMEOUT,
                check=False,
            )
            return result.returncode == 0
        except Exception:  # noqa: BLE001 -- health probe must never raise
            return False

    # -- submit --------------------------------------------------------------

    def submit(self, entry: PolicyEntry, request_id: str | None) -> str:
        script = self._registry.resolved_script(entry)
        cmd = [self._sbatch, "--parsable", "--export=NONE"]
        if request_id:
            # Durable, restart-proof idempotency backstop; also lets list/status
            # recover the request id straight from Slurm.
            cmd += ["--comment", request_id]
        cmd.append(script)

        result = self._exec(cmd, cwd=self._config.project_root)
        raw = (result.stdout or "").strip()
        # --parsable yields "<jobid>" or "<jobid>;<cluster>".
        job_id = raw.split(";", 1)[0].strip()
        if not job_id.isdigit():
            logger.error("Unexpected sbatch output: %r", raw)
            raise SlurmError("Could not parse sbatch output.")
        return job_id

    # -- query ---------------------------------------------------------------

    def _squeue_jobs(self) -> list[dict]:
        result = self._exec([self._squeue, "--json"])
        data = json.loads(result.stdout or "{}")
        return data.get("jobs", []) or []

    def _sacct_job(self, job_id: str) -> dict | None:
        result = self._exec([self._sacct, "--json", "--jobs", job_id])
        data = json.loads(result.stdout or "{}")
        jobs = data.get("jobs", []) or []
        return jobs[0] if jobs else None

    def _to_info(self, raw: dict) -> JobInfo:
        job_id = str(raw.get("job_id", "")).strip()
        name = str(raw.get("name", "") or "")
        state = _normalize_state(raw.get("job_state") or raw.get("state"))
        comment = _normalize_comment(raw.get("comment"))
        model = self._registry.model_for_job_name(name)
        return JobInfo(job_id=job_id, model=model, state=state, name=name, comment=comment)

    def _owned(self, raw: dict) -> bool:
        name = str(raw.get("name", "") or "")
        if not self._registry.is_launcher_job_name(name):
            return False
        user = str(raw.get("user_name", "") or "")
        return (not self._user) or user == self._user

    def list_jobs(self) -> list[JobInfo]:
        """Active launcher-owned jobs only (present in squeue)."""

        return [self._to_info(j) for j in self._squeue_jobs() if self._owned(j)]

    def get_job(self, job_id: str) -> JobInfo | None:
        for raw in self._squeue_jobs():
            if str(raw.get("job_id", "")).strip() == str(job_id):
                return self._to_info(raw)
        raw = self._sacct_job(str(job_id))
        if raw is not None:
            return self._to_info(raw)
        return None

    def count_active_jobs(self) -> int:
        return len(self.list_jobs())

    def count_active_model_jobs(self, model: str) -> int:
        return sum(1 for j in self.list_jobs() if j.model == model)

    def find_by_request_id(self, request_id: str) -> JobInfo | None:
        if not request_id:
            return None
        for job in self.list_jobs():
            if job.comment == request_id:
                return job
        return None

    # -- cancel --------------------------------------------------------------

    def cancel(self, job_id: str) -> None:
        self._exec([self._scancel, str(job_id)])


def _normalize_state(value: object) -> str:
    """Slurm reports state as a list (newer) or string (older); sacct nests it
    under {"current": [...]}. Reduce all forms to a single upper-case string."""

    if value is None:
        return "UNKNOWN"
    if isinstance(value, dict):
        value = value.get("current")
    if isinstance(value, list):
        return str(value[0]).upper() if value else "UNKNOWN"
    return str(value).upper()


def _normalize_comment(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        # sacct nests the user comment under "job".
        job_comment = value.get("job")
        return str(job_comment) if job_comment else None
    return None


def is_terminal(state: str) -> bool:
    return state.upper() in _TERMINAL_STATES
