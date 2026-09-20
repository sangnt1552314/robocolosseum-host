"""Single source of truth mapping public model names to everything they need.

The launcher uses ``script`` / ``job_name``; the GPU worker uses ``policy`` /
``config``. Keeping one table here prevents the launcher and worker from
drifting out of sync. HTTP callers only ever supply a *public name* that must
already be a key here -- never a script path or policy class.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    public_name: str
    policy: str  # registry key in robocolosseum.policies.registry
    script: str  # project-relative PBS script path
    job_name: str  # PBS -N job name (used to recognise our jobs)
    config: str  # default policy YAML for the worker


MODELS: dict[str, ModelSpec] = {
    "molmoact2-droid": ModelSpec(
        "molmoact2-droid",
        policy="molmoact2",
        script="pbs_remote_inference/pbs/molmoact2_droid.pbs",
        job_name="pbsri-molmoact2",
        config="configs/molmoact2.yaml",
    ),
    "pi05-droid": ModelSpec(
        "pi05-droid",
        policy="pi05_droid",
        script="pbs_remote_inference/pbs/pi05_droid.pbs",
        job_name="pbsri-pi05",
        config="configs/pi05_droid.yaml",
    ),
    "groot-droid": ModelSpec(
        "groot-droid",
        policy="gr00t_n17_droid",
        script="pbs_remote_inference/pbs/groot_droid.pbs",
        job_name="pbsri-groot",
        config="configs/gr00t_n17_droid.yaml",
    ),
    "g05-droid": ModelSpec(
        "g05-droid",
        policy="galaxea_g05",
        script="pbs_remote_inference/pbs/g05_droid.pbs",
        job_name="pbsri-g05",
        config="configs/galaxea_g05.yaml",
    ),
    "lingbot": ModelSpec(
        "lingbot",
        policy="lingbot_vla_v2",
        script="pbs_remote_inference/pbs/lingbot.pbs",
        job_name="pbsri-lingbot",
        config="configs/lingbot_vla_v2.yaml",
    ),
}


def available_models() -> list[str]:
    return sorted(MODELS)
