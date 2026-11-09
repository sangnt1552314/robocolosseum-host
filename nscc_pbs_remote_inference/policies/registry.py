"""Local policy registry for colosseum_server.py.

Maps a policy name (the ``policy.name`` in the config YAML, same key used by the
shared ``robocolosseum`` registry) to an adapter defined under
``nscc_pbs_remote_inference.policies``. The server consults this registry first and
falls back to ``robocolosseum.policies.registry`` when a name is absent here, so
a local adapter transparently overrides the shared one for that model.

To route a model through a local adapter, add one line::

    "<policy_name>": "nscc_pbs_remote_inference.policies.<module>:<ClassName>",
"""

from __future__ import annotations

import importlib
from typing import Any

from robocolosseum.policies.base import BasePolicyAdapter

LOCAL_POLICY_REGISTRY: dict[str, str] = {
    # allenai/MolmoAct2-* for any embodiment (extends the shared DROID adapter).
    "molmoact2": "nscc_pbs_remote_inference.policies.molmoact2:MolmoAct2Adapter",
    "lingbot": "nscc_pbs_remote_inference.policies.lingbot:LingbotAdapter",  # LingBot-VLA v2 fine-tunes (upstream deploy class)
    "g05": "nscc_pbs_remote_inference.policies.g05:G05Adapter",
    "pi05": "nscc_pbs_remote_inference.policies.pi05:Pi05Adapter",  # LeRobot pi0.5 fine-tunes with saved processors (so101, yam)
    "groot": "nscc_pbs_remote_inference.policies.groot:GrootLeRobotAdapter",  # LeRobot GR00T N1.7 fine-tunes (so101, yam)
}


def has_local_policy(name: str) -> bool:
    return name in LOCAL_POLICY_REGISTRY


def create_local_policy(name: str, config: Any) -> BasePolicyAdapter:
    spec = LOCAL_POLICY_REGISTRY[name]
    module_path, _, class_name = spec.partition(":")
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(config)
