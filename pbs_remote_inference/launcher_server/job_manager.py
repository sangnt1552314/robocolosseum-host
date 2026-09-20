"""PBS control-plane backend and in-memory session store.

Only machine-readable PBS interfaces are used (``qsub``, ``qstat -f -F json``,
``qdel``). All commands run in list form -- never through a shell -- so an HTTP
caller can never inject arguments. Secrets are passed to the child job via
``qsub -v`` (validated values only) and are never logged.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pbs_remote_inference.models import ModelSpec

from .model_registry import ModelRegistry

logger = logging.getLogger("pbs_launcher.backend")

_DEFAULT_TIMEOUT = 30.0
# PBS job id, e.g. "1234567" or "1234567.pbs-server".
_JOB_ID_RE = re.compile(r"^\d+(\.[A-Za-z0-9._\-]+)?$")

# PBS single-character job states -> our session status vocabulary.
_STATE_MAP = {
    "Q": "queued",
    "H": "queued",
    "W": "queued",
    "T": "queued",
    "R": "running",
    "E": "running",
    "S": "running",
    "B": "running",
    "F": "finished",
    "X": "finished",
}

Runner = Callable[..., subprocess.CompletedProcess]


class PBSError(Exception):
    """A PBS command failed. The message is safe to surface (sanitized)."""


@dataclass
class Session:
    session_id: str
    job_id: str
    model: str
    callback_url: str
    status: str
    created: float = field(default_factory=time.time)


class SessionStore:
    """Thread-safe in-memory session table (no database in v1)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def create(self, model: str, callback_url: str) -> str:
        session_id = f"session-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._sessions[session_id] = Session(
                session_id=session_id,
                job_id="",
                model=model,
                callback_url=callback_url,
                status="submitting",
            )
        return session_id

    def set(self, session_id: str, **fields) -> None:
        with self._lock:
            session = self._sessions[session_id]
            for key, value in fields.items():
                setattr(session, key, value)

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return None if session is None else Session(**vars(session))

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


class PBSBackend:
    def __init__(
        self,
        registry: ModelRegistry,
        *,
        project_root: Path,
        qsub: str = "qsub",
        qstat: str = "qstat",
        qdel: str = "qdel",
        runner: Runner = subprocess.run,
    ) -> None:
        self._registry = registry
        self._project_root = project_root
        self._qsub = qsub
        self._qstat = qstat
        self._qdel = qdel
        self._run = runner

    # -- low-level -----------------------------------------------------------

    def _exec(self, cmd: list[str]) -> subprocess.CompletedProcess:
        try:
            result = self._run(
                cmd,
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=_DEFAULT_TIMEOUT,
                check=False,
            )
        except FileNotFoundError as exc:
            logger.error("PBS command not found: %s", cmd[0])
            raise PBSError("PBS command not available.") from exc
        except subprocess.TimeoutExpired as exc:
            logger.error("PBS command timed out: %s", cmd[0])
            raise PBSError("PBS command timed out.") from exc
        if result.returncode != 0:
            # Log real stderr for operators only; return a generic message.
            logger.error(
                "Command %s failed (rc=%s): %s", cmd[0], result.returncode, result.stderr
            )
            raise PBSError(f"PBS command '{cmd[0]}' failed.")
        return result

    def available(self) -> bool:
        try:
            result = self._run(
                [self._qstat, "--version"],
                capture_output=True,
                text=True,
                timeout=_DEFAULT_TIMEOUT,
                check=False,
            )
            return result.returncode == 0
        except Exception:  # noqa: BLE001 -- health probe must never raise
            return False

    # -- submit --------------------------------------------------------------

    def submit(
        self,
        spec: ModelSpec,
        *,
        session_id: str,
        callback_url: str,
        callback_token: str,
    ) -> str:
        script = self._registry.resolved_script(spec)
        # Values are validated (session/model internal; url+token schema-checked)
        # and contain no commas, so the -v list stays well-formed. subprocess
        # list form means no shell can interpret them.
        exports = ",".join(
            [
                f"SESSION_ID={session_id}",
                f"MODEL={spec.public_name}",
                f"CALLBACK_URL={callback_url}",
                f"CALLBACK_TOKEN={callback_token}",
            ]
        )
        cmd = [self._qsub, "-v", exports, script]
        result = self._exec(cmd)
        raw = (result.stdout or "").strip().splitlines()
        job_id = raw[-1].strip() if raw else ""
        if not _JOB_ID_RE.match(job_id):
            logger.error("Unexpected qsub output: %r", result.stdout)
            raise PBSError("Could not parse qsub output.")
        return job_id

    # -- status --------------------------------------------------------------

    def job_status(self, job_id: str) -> str | None:
        """Return our status vocabulary for a job id, or None if PBS lost it."""
        if not _JOB_ID_RE.match(job_id):
            raise PBSError("invalid job id")
        try:
            result = self._exec([self._qstat, "-f", "-F", "json", "-x", job_id])
        except PBSError:
            return None
        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return None
        jobs = data.get("Jobs", {}) or {}
        record = jobs.get(job_id)
        if record is None and jobs:
            record = next(iter(jobs.values()))
        if not record:
            return None
        state = str(record.get("job_state", "")).strip().upper()
        exit_status = record.get("Exit_status")
        if state in {"F", "X"} and exit_status not in (None, 0, "0"):
            return "failed"
        return _STATE_MAP.get(state, "running")

    # -- cancel --------------------------------------------------------------

    def cancel(self, job_id: str) -> None:
        if not _JOB_ID_RE.match(job_id):
            raise PBSError("invalid job id")
        self._exec([self._qdel, job_id])
