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
    "molmoact2": ModelSpec(
        "molmoact2",
        policy="molmoact2",
        script="pbs_remote_inference/pbs/molmoact2.pbs",
        job_name="pbsri-molmoact2",
        config="pbs_remote_inference/configs/so101/stack_white_bowls_100episodes/molmoact2.yaml",
    ),
    "pi05": ModelSpec(
        "pi05",
        policy="pi05",
        script="pbs_remote_inference/pbs/pi05.pbs",
        job_name="pbsri-pi05",
        config="pbs_remote_inference/configs/so101/stack_white_bowls_100episodes/pi05.yaml",
    ),
    "groot": ModelSpec(
        "groot",
        policy="gr00t_n17",
        script="pbs_remote_inference/pbs/groot.pbs",
        job_name="pbsri-groot",
        config="pbs_remote_inference/configs/so101/stack_white_bowls_100episodes/gr00t_n17.yaml",
    ),
    "g05": ModelSpec(
        "g05",
        policy="galaxea_g05",
        script="pbs_remote_inference/pbs/g05.pbs",
        job_name="pbsri-g05",
        config="pbs_remote_inference/configs/so101/stack_white_bowls_100episodes/galaxea_g05.yaml",
    ),
    "lingbot": ModelSpec(
        "lingbot",
        policy="lingbot_vla_v2",
        script="pbs_remote_inference/pbs/lingbot.pbs",
        job_name="pbsri-lingbot",
        config="pbs_remote_inference/configs/so101/stack_white_bowls_100episodes/lingbot_vla_v2.yaml",
    ),
}


def available_models() -> list[str]:
    return sorted(MODELS)
