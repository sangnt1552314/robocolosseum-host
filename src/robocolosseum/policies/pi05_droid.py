"""lerobot/pi05_droid (pi0.5) policy adapter.

Wraps the LeRobot ``PI0Policy``. All model-specific knowledge (camera batch
keys, state layout, processor pipeline) lives here and nowhere else.

Verified against the LeRobot PI0 API:

* ``PI0Policy.from_pretrained`` loads the checkpoint;
* ``make_pre_post_processors`` builds the processor pipeline that normalises
  inputs and tokenises the language instruction;
* ``predict_action_chunk`` returns an action chunk shaped
  ``(batch, chunk, action_dim)`` in the model's normalised space;
* the post-processor un-normalises those actions;
* ``reset`` clears the policy's internal action queue between rollouts.

VERIFY against the installed lerobot version and the checkpoint config before
trusting the output: the camera batch keys (``policy.options.image_keys``) and
the DROID state layout. This mirrors the project's documented approach for
MolmoAct2.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..config import AppConfig
from .base import BasePolicyAdapter

log = logging.getLogger(__name__)


class Pi05Adapter(BasePolicyAdapter):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.policy_cfg = config.policy
        self.camera = config.camera_mapping

        opts = self.policy_cfg.options
        image_keys = opts.get("image_keys", {}) or {}
        self.exterior_1_key = image_keys.get(
            "exterior_1", "observation.images.exterior_image_1_left"
        )
        self.exterior_2_key = image_keys.get(
            "exterior_2", "observation.images.exterior_image_2_left"
        )
        self.wrist_key = image_keys.get("wrist", "observation.images.wrist_image_left")
        self.state_key = opts.get("state_key", "observation.state")
        self.task_key = opts.get("task_key", "task")

        self._torch: Any = None
        self.policy: Any = None
        self.preprocessor: Any = None
        self.postprocessor: Any = None

    def load(self) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        self._torch = torch

        ckpt = self.policy_cfg.checkpoint
        log.info("Loading LeRobot PI0 checkpoint: %s", ckpt)
        self.policy = PI0Policy.from_pretrained(ckpt).to(self.policy_cfg.device).eval()
        # The processor pipeline ships with the checkpoint; it handles input
        # normalisation, language tokenisation and device placement, and the
        # post-processor un-normalises the predicted actions.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=ckpt
        )
        log.info("PI0 policy loaded on %s", self.policy_cfg.device)

    def _image(self, arr: Any) -> Any:
        torch = self._torch
        img = np.asarray(arr)
        if img.ndim != 3:
            raise ValueError(f"Expected an HWC image, got shape {img.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(img)).float()
        # LeRobot expects images normalised to [0, 1]; scale 8-bit input.
        if float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        # HWC -> CHW, add a leading batch dimension.
        return tensor.permute(2, 0, 1).unsqueeze(0)

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

        exterior_img = self._image(external)
        # Single exterior camera duplicated into both DROID exterior slots.
        batch = {
            self.exterior_1_key: exterior_img,
            self.exterior_2_key: exterior_img,
            self.wrist_key: self._image(wrist),
            self.state_key: self._state(observation),
            self.task_key: observation.instruction or "",
        }

        with torch.inference_mode():
            processed = self.preprocessor(batch)
            actions = self.policy.predict_action_chunk(processed)
            actions = self.postprocessor(actions)

        if hasattr(actions, "detach"):
            actions = actions.detach().to("cpu").numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
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
