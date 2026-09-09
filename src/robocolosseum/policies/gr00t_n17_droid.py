"""nvidia/GR00T-N1.7-DROID policy adapter.

Wraps the NVIDIA Isaac GR00T ``Gr00tPolicy`` (N1.7) finetuned on DROID. All
model-specific knowledge (nested observation layout, embodiment tag, action
keys) lives here and nowhere else.

Verified against the GR00T N1.7 Policy API guide
(``getting_started/policy.md`` / ``examples/DROID/README.md``):

* ``Gr00tPolicy(model_path, embodiment_tag, device)`` loads the checkpoint; the
  embodiment tag selects the modality config (state/action keys, normalisation).
* ``get_action(observation)`` takes a nested dict
  ``{"video": {...}, "state": {...}, "language": {...}}`` where video is
  ``uint8`` ``(B, T, H, W, 3)`` and state is ``float32`` ``(B, T, D)``; it
  returns ``(action_dict, info)`` with each action stream shaped
  ``(B, T, D)`` in physical units (already un-normalised).
* the DROID embodiment (``OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT``) exposes the
  action streams ``joint_position`` (7, relative) and ``gripper_position`` (1).

VERIFY against the installed ``gr00t`` version and the checkpoint before
trusting the output: the video/state keys (``policy.options``), that DROID
``joint_position`` is a *relative* joint action, and the camera mapping. This
mirrors the project's documented approach for MolmoAct2 / pi05_droid.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..config import AppConfig
from .base import BasePolicyAdapter

log = logging.getLogger(__name__)


class GR00TN17DroidAdapter(BasePolicyAdapter):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.policy_cfg = config.policy
        self.camera = config.camera_mapping

        opts = self.policy_cfg.options
        self.embodiment_tag = opts.get(
            "embodiment_tag", "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
        )
        # Nested-observation keys expected by the DROID modality config. The
        # Colosseum exterior slot feeds ``exterior_image_1_left``, the wrist
        # feeds ``wrist_image_left``.
        video_keys = opts.get("video_keys", {}) or {}
        self.exterior_key = video_keys.get("exterior", "exterior_image_1_left")
        self.wrist_key = video_keys.get("wrist", "wrist_image_left")
        # State streams built from the live observation. DROID also defines an
        # ``eef_9d`` stream; it is dropped here because the Colosseum observation
        # exposes joints + gripper (state dropout during training tolerates the
        # missing stream). Override via options if your observation provides it.
        self.joint_state_key = opts.get("joint_state_key", "joint_position")
        self.gripper_state_key = opts.get("gripper_state_key", "gripper_position")
        # Language key for the DROID modality config.
        self.language_key = opts.get(
            "language_key", "annotation.language.language_instruction"
        )
        # Action streams concatenated (in order) into the emitted chunk. Default
        # matches the Colosseum ``joint_position`` action space: 7 joints + 1
        # gripper = 8 dims.
        self.action_keys = list(
            opts.get("action_keys", ["joint_position", "gripper_position"])
        )

        self._policy: Any = None

    def load(self) -> None:
        try:
            from gr00t.policy import Gr00tPolicy
        except ImportError:  # older layout
            from gr00t.model.policy import Gr00tPolicy

        log.info("Loading GR00T policy: %s", self.policy_cfg.checkpoint)
        self._policy = Gr00tPolicy(
            model_path=self.policy_cfg.checkpoint,
            embodiment_tag=self.embodiment_tag,
            device=self.policy_cfg.device,
        )
        log.info(
            "GR00T policy loaded on %s (embodiment=%s)",
            self.policy_cfg.device,
            self.embodiment_tag,
        )

    def _video(self, arr: Any) -> np.ndarray:
        # GR00T expects uint8 RGB frames with a leading (batch, time) pair:
        # (B, T, H, W, 3).
        img = np.asarray(arr)
        if img.ndim != 3:
            raise ValueError(f"Expected an HWC image, got shape {img.shape}")
        return img.astype(np.uint8)[np.newaxis, np.newaxis, ...]

    def _state(self, observation: Any) -> dict[str, np.ndarray]:
        joints = observation.state.joints
        if joints is None:
            raise ValueError("Observation is missing joint state")
        joints = np.asarray(joints, dtype=np.float32).reshape(1, 1, -1)

        state = {self.joint_state_key: joints}
        gripper = observation.state.gripper
        if gripper is not None:
            gripper = np.asarray(gripper, dtype=np.float32).reshape(1, 1, -1)
            state[self.gripper_state_key] = gripper
        return state

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
            "video": {
                self.exterior_key: self._video(external),
                self.wrist_key: self._video(wrist),
            },
            "state": self._state(observation),
            "language": {self.language_key: [[observation.instruction or ""]]},
        }

        result = self._policy.get_action(obs)
        # N1.7 returns (action_dict, info); older versions return the dict only.
        action_dict = result[0] if isinstance(result, tuple) else result

        chunks = []
        for key in self.action_keys:
            if key not in action_dict:
                raise KeyError(
                    f"Action stream '{key}' not in GR00T output {list(action_dict)}"
                )
            part = np.asarray(action_dict[key], dtype=np.float32)
            if part.ndim == 3 and part.shape[0] == 1:
                part = part[0]  # drop batch -> (horizon, dim)
            if part.ndim == 1:
                part = part[:, np.newaxis]
            chunks.append(part)

        return np.concatenate(chunks, axis=-1).astype(np.float32)

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
