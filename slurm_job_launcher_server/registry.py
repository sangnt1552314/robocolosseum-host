"""Strict model registry.

The registry is the single place that maps an HTTP-supplied model *name* to a
fixed, pre-approved Slurm script. The caller can never specify a script path,
shell command, or arbitrary Slurm argument -- only a model name that must
already exist here (loaded from the committed config).
"""

from __future__ import annotations

from pathlib import Path

from .config import LauncherConfig, PolicyEntry


class UnknownModelError(KeyError):
    """Raised when a caller asks for a model that is not registered."""


class PolicyRegistry:
    def __init__(self, config: LauncherConfig) -> None:
        self._config = config
        self._policies = dict(config.policies)
        # Reverse map: Slurm job name -> model, used to recognise which jobs
        # this launcher is allowed to manage.
        self._by_job_name: dict[str, str] = {
            entry.slurm_job_name: model for model, entry in self._policies.items()
        }

    def available_models(self) -> list[str]:
        return sorted(self._policies)

    def get(self, model: str) -> PolicyEntry:
        try:
            return self._policies[model]
        except KeyError as exc:
            raise UnknownModelError(model) from exc

    def is_launcher_job_name(self, slurm_job_name: str) -> bool:
        return slurm_job_name in self._by_job_name

    def model_for_job_name(self, slurm_job_name: str) -> str | None:
        return self._by_job_name.get(slurm_job_name)

    def managed_job_names(self) -> set[str]:
        return set(self._by_job_name)

    def resolved_script(self, entry: PolicyEntry) -> str:
        """Return the script path exactly as registered (relative to the project
        root). Defense in depth: refuse anything that escapes the project."""

        script = Path(entry.script)
        if script.is_absolute() or ".." in script.parts:
            raise ValueError(f"registered script path is not project-relative: {script}")
        return entry.script
