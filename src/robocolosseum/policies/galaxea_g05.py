"""OpenGalaxea/G05 (G0.5 DROID) policy adapter.

Wraps the Galaxea G0.5 autoregressive VLA (``g05-droid`` checkpoint). Unlike the
Hugging Face ``AutoModel`` checkpoints, G0.5 ships as a Hydra-configured
``model_state_dict.pt`` with sidecar ``dataset_stats.json`` and a shared action
tokenizer, and is normally served through the repo's policy server. To keep this
worker single-process, the adapter builds the policy in-process through the
installed ``g05`` package.

Because the in-process factory is repo-version specific, the construction is
driven entirely by ``policy.options`` instead of hard-coding symbols that may
move between releases:

* ``builder`` — dotted ``"module:callable"`` that returns a ready policy object
  given the checkpoint path, device and action-step count (see
  ``scripts/serve_policy.py`` in the GalaxeaVLA repo);
* ``infer_method`` — the policy's inference entrypoint (takes the observation
  dict below, returns an action chunk);
* the multi-view / state / language keys and the DROID action width.

VERIFY every option against the installed GalaxeaVLA (``g05``) version and the
``g05-droid`` checkpoint before trusting the output. This mirrors the project's
documented "adapter owns model specifics + VERIFY" approach.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

import numpy as np

from ..config import AppConfig
from .base import BasePolicyAdapter

log = logging.getLogger(__name__)


class GalaxeaG05Adapter(BasePolicyAdapter):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.policy_cfg = config.policy
        self.camera = config.camera_mapping

        opts = self.policy_cfg.options
        # In-process policy factory: "module.path:callable". Must match the
        # installed g05 package (see GalaxeaVLA scripts/serve_policy.py).
        self.builder_spec = opts.get("builder")
        self.infer_method = opts.get("infer_method", "infer")
        self.embodiment = opts.get("embodiment", "galaxea_droid")
        self.action_steps = int(opts.get("action_steps", 16))
        # Extra keyword arguments forwarded verbatim to the builder callable.
        self.builder_kwargs = dict(opts.get("builder_kwargs", {}) or {})

        # Observation dict keys expected by the g05 DROID policy.
        image_keys = opts.get("image_keys", {}) or {}
        self.exterior_key = image_keys.get("exterior", "exterior_image")
        self.wrist_key = image_keys.get("wrist", "wrist_image")
        self.state_key = opts.get("state_key", "state")
        self.task_key = opts.get("task_key", "instruction")
        # DROID: 7 joints + 1 gripper. Set null to disable slicing.
        action_dim = opts.get("action_dim", 8)
        self.action_dim = int(action_dim) if action_dim is not None else None

        self._policy: Any = None

    def _resolve(self, spec: str) -> Callable[..., Any]:
        module_path, _, attr = spec.partition(":")
        if not attr:
            raise ValueError(
                f"builder must be 'module.path:callable', got '{spec}'"
            )
        module = importlib.import_module(module_path)
        return getattr(module, attr)

    def load(self) -> None:
        if not self.builder_spec:
            raise ValueError(
                "galaxea_g05 requires policy.options.builder "
                "(e.g. 'g05.policy:build_policy'); set it to match the installed "
                "g05 package."
            )
        build = self._resolve(self.builder_spec)
        log.info("Building G0.5 policy from: %s", self.policy_cfg.checkpoint)
        self._policy = build(
            ckpt_path=self.policy_cfg.checkpoint,
            device=self.policy_cfg.device,
            action_steps=self.action_steps,
            embodiment=self.embodiment,
            **self.builder_kwargs,
        )
        log.info("G0.5 policy loaded on %s", self.policy_cfg.device)

    def _image(self, arr: Any) -> np.ndarray:
        img = np.asarray(arr)
        if img.ndim != 3:
            raise ValueError(f"Expected an HWC image, got shape {img.shape}")
        return img.astype(np.uint8)

    def _state(self, observation: Any) -> np.ndarray:
        joints = observation.state.joints
        if joints is None:
            raise ValueError("Observation is missing joint state")
        joints = np.asarray(joints, dtype=np.float32).reshape(-1)

        gripper = observation.state.gripper
        if gripper is None:
            state = joints
        else:
            gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
            state = np.concatenate([joints, gripper])
        return state.astype(np.float32)

    def predict(self, observation: Any) -> np.ndarray:
        if self._policy is None:
            raise RuntimeError("predict() called before load()")

        external = getattr(observation.state, self.camera.external)
        wrist = getattr(observation.state, self.camera.wrist)
        if external is None:
            raise ValueError(
                f"External camera '{self.camera.external}' is missing from the observation"
            )
        if wrist is None:
            raise ValueError(
                f"Wrist camera '{self.camera.wrist}' is missing from the observation"
            )

        obs = {
            self.exterior_key: self._image(external),
            self.wrist_key: self._image(wrist),
            self.state_key: self._state(observation),
            self.task_key: observation.instruction or "",
        }

        infer_fn = getattr(self._policy, self.infer_method)
        actions = infer_fn(obs)

        if hasattr(actions, "detach"):
            actions = actions.detach().to("cpu").numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if self.action_dim is not None and actions.shape[-1] > self.action_dim:
            actions = actions[..., : self.action_dim]
        return actions

    def reset(self) -> None:
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()

    def close(self) -> None:
        if self._policy is not None:
            del self._policy
            self._policy = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
