"""Policy registry.

Adapters are registered by name and imported lazily so that heavy model
dependencies (torch, transformers, ...) are only loaded when that specific
policy is actually created. This keeps each model's environment isolated: the
read-only router test, for example, needs no torch at all.

To add a new policy, add one line mapping ``name -> "module.path:ClassName"``.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import BasePolicyAdapter

POLICY_REGISTRY: dict[str, str] = {
    "molmoact2": "robocolosseum.policies.molmoact2:MolmoAct2Adapter",
}


def available_policies() -> list[str]:
    return sorted(POLICY_REGISTRY)


def create_policy(name: str, config: Any) -> BasePolicyAdapter:
    try:
        spec = POLICY_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown policy '{name}'. Available: {available_policies()}"
        ) from exc

    module_path, _, class_name = spec.partition(":")
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(config)
