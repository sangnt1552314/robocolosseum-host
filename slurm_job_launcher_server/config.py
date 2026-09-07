"""Launcher configuration.

Two deliberately separate sources, mirroring the main project:

* The committed YAML file (``launcher.yaml`` / ``launcher.example.yaml``) holds
  non-secret configuration: admission limits and the strict policy registry.
* Secrets (``LAUNCHER_API_TOKEN``) come from the environment only and are never
  read from, or written to, any file here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Repository root (parent of this package), so relative Slurm script paths in
# the config resolve against the real project checkout.
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PolicyEntry:
    """A single registered policy. Everything the caller may trigger is fixed
    here in configuration -- the HTTP caller only ever supplies the model name."""

    model: str
    script: str
    slurm_job_name: str
    max_replicas: int


@dataclass(frozen=True)
class LauncherConfig:
    host: str
    port: int
    max_active_jobs: int
    project_root: Path
    ngrok_bin: str
    policies: dict[str, PolicyEntry] = field(default_factory=dict)


def _config_path() -> Path:
    """Resolve which YAML to load.

    Precedence: ``LAUNCHER_CONFIG`` env var, then a git-ignored
    ``launcher.yaml`` next to this package, then the committed example.
    """

    override = os.environ.get("LAUNCHER_CONFIG")
    if override:
        return Path(override)
    real = _PACKAGE_DIR / "launcher.yaml"
    if real.exists():
        return real
    return _PACKAGE_DIR / "launcher.example.yaml"


def load_config(path: str | Path | None = None) -> LauncherConfig:
    """Load and validate the launcher YAML."""

    cfg_path = Path(path) if path is not None else _config_path()
    data = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}

    launcher_raw = data.get("launcher", {}) or {}
    host = str(launcher_raw.get("host", "127.0.0.1"))
    port = int(launcher_raw.get("port", 8000))
    max_active_jobs = int(launcher_raw.get("max_active_jobs", 10))
    if max_active_jobs < 1:
        raise ValueError("launcher.max_active_jobs must be >= 1")

    project_root = Path(
        os.environ.get("PROJECT_ROOT")
        or launcher_raw.get("project_root")
        or _DEFAULT_PROJECT_ROOT
    ).resolve()

    ngrok_bin = str(
        os.environ.get("NGROK_BIN")
        or launcher_raw.get("ngrok_bin")
        or "ngrok"
    )

    policies_raw = data.get("policies", {}) or {}
    if not policies_raw:
        raise ValueError("config must define at least one policy under 'policies'")

    policies: dict[str, PolicyEntry] = {}
    for model, entry in policies_raw.items():
        entry = entry or {}
        try:
            script = str(entry["script"])
            slurm_job_name = str(entry["slurm_job_name"])
        except KeyError as exc:
            raise ValueError(
                f"policy '{model}' is missing required key: {exc}"
            ) from exc
        max_replicas = int(entry.get("max_replicas", 1))
        if max_replicas < 1:
            raise ValueError(f"policy '{model}' max_replicas must be >= 1")
        policies[str(model)] = PolicyEntry(
            model=str(model),
            script=script,
            slurm_job_name=slurm_job_name,
            max_replicas=max_replicas,
        )

    return LauncherConfig(
        host=host,
        port=port,
        max_active_jobs=max_active_jobs,
        project_root=project_root,
        ngrok_bin=ngrok_bin,
        policies=policies,
    )
