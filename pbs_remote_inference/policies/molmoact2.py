"""MolmoAct2 policy adapter for LeRobot fine-tuned checkpoints.

The checkpoints deployed here are produced by robocolosseum-training as LeRobot
policies (``config.json`` with ``type: molmoact2`` plus ``model.safetensors`` and
a ``policy_preprocessor`` / ``policy_postprocessor`` pipeline) -- NOT a plain
``transformers`` model. They must be loaded through LeRobot, the same way the
shared pi05 adapter loads its checkpoint. This module keeps that LeRobot-specific
loading path out of the shared ``robocolosseum.policies.molmoact2`` adapter,
which targets the raw ``allenai/MolmoAct2-DROID`` transformers checkpoint.

Verified against the SO101 stack_white_bowls checkpoint config: image keys
``observation.images.front`` (external) + ``observation.images.wrist``,
``observation.state`` (6 = 5 arm joints + gripper), action shape 6,
``n_action_steps=30``, continuous flow-matching inference.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from robocolosseum.config import AppConfig
from robocolosseum.policies.base import BasePolicyAdapter

log = logging.getLogger(__name__)


class MolmoAct2LeRobotAdapter(BasePolicyAdapter):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.policy_cfg = config.policy
        self.camera = config.camera_mapping

        opts = self.policy_cfg.options
        # Batch keys expected by the LeRobot molmoact2 checkpoint. The exterior
        # camera feeds ``front``, the wrist feeds ``wrist``.
        image_keys = opts.get("image_keys", {}) or {}
        self.front_key = image_keys.get("front", "observation.images.front")
        self.wrist_key = image_keys.get("wrist", "observation.images.wrist")
        self.state_key = opts.get("state_key", "observation.state")
        self.task_key = opts.get("task_key", "task")
        self.inference_action_mode = opts.get("inference_action_mode", "continuous")
        # Optional: number of flow-matching steps; None falls back to the config.
        num_steps = opts.get("num_steps")
        self.num_steps = int(num_steps) if num_steps is not None else None
        # SO101 action is 6-dim; None = keep whatever the policy returns.
        action_dim = opts.get("action_dim")
        self.action_dim = int(action_dim) if action_dim is not None else None

        self._torch: Any = None
        self.policy: Any = None
        self.preprocessor: Any = None
        self.postprocessor: Any = None

    def load(self) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

        self._torch = torch

        ckpt = self.policy_cfg.checkpoint
        log.info("Loading LeRobot MolmoAct2 checkpoint: %s", ckpt)
        self.policy = (
            MolmoAct2Policy.from_pretrained(ckpt).to(self.policy_cfg.device).eval()
        )
        # The processor pipeline ships with the checkpoint; it normalises inputs
        # and un-normalises the predicted actions.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=ckpt
        )
        log.info("MolmoAct2 policy loaded on %s", self.policy_cfg.device)

    def _image(self, arr: Any) -> Any:
        torch = self._torch
        # copy=True: SDK-decoded frames may be read-only, which torch rejects.
        img = np.array(arr, dtype=np.float32, copy=True)
        if img.ndim != 3:
            raise ValueError(f"Expected an HWC image, got shape {img.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        # LeRobot expects images normalised to [0, 1]; scale 8-bit input.
        if float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        # HWC -> CHW, add a leading batch dimension.
        return tensor.permute(2, 0, 1).unsqueeze(0)

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        torch = self._torch
        device = self.policy_cfg.device
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _state(self, observation: Any) -> Any:
        torch = self._torch
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
        return torch.from_numpy(state.astype(np.float32)).unsqueeze(0)

    def predict(self, observation: Any) -> np.ndarray:
        torch = self._torch
        if self.policy is None:
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

        batch = {
            self.front_key: self._image(external),
            self.wrist_key: self._image(wrist),
            self.state_key: self._state(observation),
            self.task_key: observation.instruction or "",
        }

        kwargs: dict[str, Any] = {"inference_action_mode": self.inference_action_mode}
        if self.num_steps is not None:
            kwargs["num_steps"] = self.num_steps

        with torch.inference_mode():
            processed = self.preprocessor(batch)
            processed = self._to_device(processed)
            actions = self.policy.predict_action_chunk(processed, **kwargs)
            actions = self.postprocessor(actions)

        if hasattr(actions, "detach"):
            actions = actions.detach().to("cpu").numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if self.action_dim is not None and actions.shape[-1] > self.action_dim:
            actions = actions[..., : self.action_dim]
        return actions

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()

    def close(self) -> None:
        if self.policy is not None and self._torch is not None:
            del self.policy
            self.policy = None
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
