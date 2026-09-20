"""Strict model registry over the shared ``models.py`` table.

The registry is the single place that maps an HTTP-supplied model *name* to a
fixed, pre-approved PBS script. A caller can never specify a script path, shell
command, or arbitrary PBS argument -- only a public model name that must already
exist here.
"""

from __future__ import annotations

from pathlib import Path

from pbs_remote_inference.models import MODELS, ModelSpec, available_models


class UnknownModelError(KeyError):
    """Raised when a caller asks for a model that is not registered."""


class ModelRegistry:
    def __init__(self) -> None:
        self._by_name = dict(MODELS)
        self._by_job_name = {spec.job_name: name for name, spec in MODELS.items()}

    def available_models(self) -> list[str]:
        return available_models()

    def get(self, model: str) -> ModelSpec:
        try:
            return self._by_name[model]
        except KeyError as exc:
            raise UnknownModelError(model) from exc

    def is_launcher_job_name(self, job_name: str) -> bool:
        return job_name in self._by_job_name

    def resolved_script(self, spec: ModelSpec) -> str:
        """Return the registered script path, refusing anything that escapes the
        project (defense in depth even though users never supply paths)."""
        script = Path(spec.script)
        if script.is_absolute() or ".." in script.parts:
            raise ValueError(f"registered script path is not project-relative: {script}")
        return spec.script
