"""Robbyant/lingbot-vla-v2 policy adapter.

Wraps the ``Robbyant/lingbot-vla-v2`` checkpoint through the Hugging Face
``AutoModel`` / ``AutoProcessor`` ``trust_remote_code`` path. All model-specific
knowledge (camera layout, state layout, the ``predict`` entrypoint) lives here
and nowhere else.

VERIFY before trusting the output: this checkpoint is gated and its exact
inference API is not publicly documented, so the specifics below are configured
through ``policy.options`` and must be confirmed against the checkpoint's model
card / ``modeling_*.py`` on the deployment machine:

* the loader class (``AutoModel`` vs ``AutoModelForVision2Seq`` etc.);
* the inference method name (``predict_action`` by default) and its signature
  (images / state / task arguments);
* the camera slots and state layout (DROID: 7 joints + 1 gripper).

This mirrors the project's documented "adapter owns model specifics + VERIFY"
approach used for MolmoAct2 and pi05_droid.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import numpy as np

from ..config import AppConfig
from .base import BasePolicyAdapter

log = logging.getLogger(__name__)


class LingbotVLAv2Adapter(BasePolicyAdapter):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.policy_cfg = config.policy
        self.camera = config.camera_mapping

        opts = self.policy_cfg.options
        # Name of the inference method exposed by the checkpoint's model class.
        self.predict_method = opts.get("predict_method", "predict_action")
        # Whether the wrist frame is duplicated into a second exterior slot
        # (DROID training layout [exterior_1, exterior_2, wrist]).
        self.duplicate_exterior = bool(opts.get("duplicate_exterior", False))
        # Extra keyword arguments forwarded verbatim to the inference method.
        self.predict_kwargs = dict(opts.get("predict_kwargs", {}) or {})
        # Keep only the real robot action dims (DROID: 7 joints + 1 gripper).
        action_dim = opts.get("action_dim", 8)
        self.action_dim = int(action_dim) if action_dim is not None else None

        self._torch: Any = None
        self._dtype: Any = None
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._dtype = getattr(torch, self.policy_cfg.dtype)

        log.info("Loading checkpoint: %s", self.policy_cfg.checkpoint)
        self.processor = AutoProcessor.from_pretrained(
            self.policy_cfg.checkpoint, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(
                self.policy_cfg.checkpoint,
                trust_remote_code=True,
                dtype=self._dtype,
            )
            .to(self.policy_cfg.device)
            .eval()
        )
        log.info(
            "Model loaded on %s (dtype=%s)",
            self.policy_cfg.device,
            self.policy_cfg.dtype,
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
        if self.duplicate_exterior:
            return [external, external, wrist]
        return [external, wrist]

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

        predict_fn = getattr(self.model, self.predict_method)

        device_type = self.policy_cfg.device.split(":")[0]
        use_autocast = self._dtype != torch.float32
        autocast_ctx = (
            torch.autocast(device_type, dtype=self._dtype)
            if use_autocast
            else nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            out = predict_fn(
                processor=self.processor,
                images=images,
                task=task,
                state=state,
                **self.predict_kwargs,
            )

        actions = getattr(out, "actions", out)
        if hasattr(actions, "detach"):
            actions = actions.detach().to("cpu").numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if self.action_dim is not None and actions.shape[-1] > self.action_dim:
            actions = actions[..., : self.action_dim]
        return actions

    def reset(self) -> None:
        if self.model is not None and hasattr(self.model, "reset"):
            self.model.reset()

    def close(self) -> None:
        if self.model is not None and self._torch is not None:
            del self.model
            self.model = None
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
