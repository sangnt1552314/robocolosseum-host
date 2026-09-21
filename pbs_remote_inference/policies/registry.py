"""Worker-local policy registry.

Maps a policy name (the ``policy.name`` in the worker YAML, same key used by the
shared ``robocolosseum`` registry) to an adapter defined under
``pbs_remote_inference.policies``. The worker consults this registry first and
falls back to ``robocolosseum.policies.registry`` when a name is absent here, so
a local adapter transparently overrides the shared one for that model.

To route a model through a worker-local adapter, add one line::

    "<policy_name>": "pbs_remote_inference.policies.<module>:<ClassName>",
"""

from __future__ import annotations

import importlib
from typing import Any

from robocolosseum.policies.base import BasePolicyAdapter

LOCAL_POLICY_REGISTRY: dict[str, str] = {
    # LeRobot fine-tuned molmoact2 checkpoints (overrides the shared
    # transformers-based robocolosseum.policies.molmoact2 adapter).
    "molmoact2": "pbs_remote_inference.policies.molmoact2:MolmoAct2LeRobotAdapter",
}


def has_local_policy(name: str) -> bool:
    return name in LOCAL_POLICY_REGISTRY


def create_local_policy(name: str, config: Any) -> BasePolicyAdapter:
    spec = LOCAL_POLICY_REGISTRY[name]
    module_path, _, class_name = spec.partition(":")
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(config)
