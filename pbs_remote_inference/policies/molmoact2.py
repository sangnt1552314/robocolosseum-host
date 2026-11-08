"""MolmoAct2 adapter for every embodiment (allenai/MolmoAct2-* checkpoints).

Extends the shared DROID adapter (``robocolosseum.policies.molmoact2``) without
changing it. Two optional ``policy`` options in the YAML:

* ``image_fields``: observation images in the order the checkpoint expects. The
  shared adapter always sends ``[external, external, wrist]`` (the DROID
  layout); MolmoAct2-SO100_101 was trained on two views and the yam checkpoint
  on three (top, left, right).
* ``action_dim``: keep the first N action columns. The SO100/101 checkpoint
  returns actions padded past the robot's 6 values.

With neither option set this behaves exactly like the shared DROID adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robocolosseum.config import AppConfig
from robocolosseum.policies.molmoact2 import MolmoAct2Adapter as SharedMolmoAct2Adapter


class MolmoAct2Adapter(SharedMolmoAct2Adapter):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        opts = self.policy.options
        self.image_fields = list(opts.get("image_fields") or [])
        action_dim = opts.get("action_dim")
        self.action_dim = int(action_dim) if action_dim is not None else None

    def _cameras(self, observation: Any) -> list[np.ndarray]:
        if not self.image_fields:
            return super()._cameras(observation)
        return [np.asarray(getattr(observation.state, name)) for name in self.image_fields]

    def predict(self, observation: Any) -> np.ndarray:
        actions = super().predict(observation)
        return actions[..., : self.action_dim] if self.action_dim else actions
