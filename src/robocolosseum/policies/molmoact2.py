"""MolmoAct2-DROID policy adapter.

Wraps the official ``allenai/MolmoAct2-DROID`` checkpoint. All model-specific
knowledge (camera layout, state layout, ``predict_action`` arguments) lives here
and nowhere else. Verified against the checkpoint's ``inference.py`` / README and
``norm_stats.json`` (action horizon 15, action_dim 8, absolute joint pose).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import numpy as np

from ..config import AppConfig
from .base import BasePolicyAdapter

log = logging.getLogger(__name__)


class MolmoAct2Adapter(BasePolicyAdapter):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.policy = config.policy
        self.camera = config.camera_mapping

        opts = self.policy.options
        self.norm_tag = opts.get("norm_tag", "franka_droid")
        self.inference_action_mode = opts.get("inference_action_mode", "continuous")
        self.num_steps = int(opts.get("num_steps", 10))
        self.normalize_language = bool(opts.get("normalize_language", True))
        self.enable_cuda_graph = bool(opts.get("enable_cuda_graph", True))

        self._torch: Any = None
        self._dtype: Any = None
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._dtype = getattr(torch, self.policy.dtype)

        log.info("Loading checkpoint from cache: %s", self.policy.checkpoint)
        self.processor = AutoProcessor.from_pretrained(
            self.policy.checkpoint, trust_remote_code=True
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                self.policy.checkpoint,
                trust_remote_code=True,
                dtype=self._dtype,
            )
            .to(self.policy.device)
            .eval()
        )
        log.info(
            "Model loaded on %s (dtype=%s)", self.policy.device, self.policy.dtype
        )

    def _cameras(self, observation: Any) -> list[np.ndarray]:
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
        external = np.asarray(external)
        wrist = np.asarray(wrist)
        # DROID checkpoint expects [exterior_1, exterior_2, wrist]; with a single
        # exterior camera we duplicate it to match the training layout.
        return [external, external, wrist]

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
        torch = self._torch
        if self.model is None:
            raise RuntimeError("predict() called before load()")

        images = self._cameras(observation)
        state = self._state(observation)
        task = observation.instruction or ""

        device_type = self.policy.device.split(":")[0]
        use_autocast = self._dtype != torch.float32
        autocast_ctx = (
            torch.autocast(device_type, dtype=self._dtype)
            if use_autocast
            else nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=task,
                state=state,
                norm_tag=self.norm_tag,
                inference_action_mode=self.inference_action_mode,
                enable_depth_reasoning=False,
                num_steps=self.num_steps,
                normalize_language=self.normalize_language,
                enable_cuda_graph=self.enable_cuda_graph,
            )

        actions = out.actions
        if hasattr(actions, "detach"):
            actions = actions.detach().to("cpu").numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions

    def reset(self) -> None:
        log.debug("MolmoAct2 reset (model is stateless between rollouts)")

    def close(self) -> None:
        if self.model is not None and self._torch is not None:
            del self.model
            self.model = None
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
