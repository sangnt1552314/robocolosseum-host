"""GR00T N1.7 adapter for LeRobot policy checkpoints (so101, yam).

For fine-tuned LeRobot GR00T N1.7 checkpoints such as ``hqfang/groot-so100_101``
and ``NUSMAGIC/GR00T-N1.7-MolmoAct2-YAM``, not raw Isaac-GR00T checkpoints (the
shared ``robocolosseum.policies.gr00t_n17`` adapter targets those). Mirrors the
Colosseum Policy Server workers ``so101_groot.py`` / ``yam_groot.py``: load the
policy and its saved pre/post processors, which own normalization, camera
packing, VLM tokenization and action unnormalization, then return one absolute
action chunk per call. Runs offline from local snapshots. ``policy`` options:

* ``base_model``: local ``nvidia/GR00T-N1.7-3B`` snapshot. The loader builds the
  model from it, then loads the fine-tuned weights.
* ``processor``: local Qwen3-VL processor directory for VLM tokenization
  (the checkpoint's bundled ``vlm_processor/`` or a Qwen3-VL-2B-Instruct snapshot).
* ``image_fields``: request image field -> checkpoint camera key. Cameras the
  checkpoint has but that are not listed are left out, not zero-padded.
* ``clip_columns``: action columns clipped to [0, 1] (YAM grippers).

State is ``joints`` followed by ``gripper`` (if any), in the checkpoint's units.
The model runs with float32 weights and bfloat16 autocast, as it was trained.
``num_steps`` from the request is ignored: GR00T keeps the checkpoint's
denoising steps (4), like the Policy Server workers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robocolosseum.config import AppConfig
from robocolosseum.policies.base import BasePolicyAdapter


class GrootLeRobotAdapter(BasePolicyAdapter):
    def __init__(self, config: AppConfig) -> None:
        self.policy_cfg = config.policy
        if self.policy_cfg.dtype != "float32":
            raise ValueError("groot runs float32 weights with bfloat16 autocast; set policy.dtype: float32")
        opts = self.policy_cfg.options
        self.base_model = str(Path(opts["base_model"]).resolve())
        self.processor = str(Path(opts["processor"]).resolve())
        self.image_fields = dict(opts["image_fields"])
        self.clip_columns = list(opts.get("clip_columns") or [])
        self._torch: Any = None
        self._policy: Any = None

    def load(self) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        checkpoint, device = self.policy_cfg.checkpoint, self.policy_cfg.device
        config = GrootConfig.from_pretrained(checkpoint, local_files_only=True)
        if config.use_relative_actions or config.n_obs_steps != 1:
            raise ValueError("expected an absolute-action, single-observation GR00T checkpoint")
        cameras = set(self.image_fields.values())
        if not cameras <= set(config.input_features):
            raise ValueError(f"cameras {sorted(cameras)} are not all in {sorted(config.input_features)}")
        config.device = device
        config.base_model_path = self.base_model
        # The cuDNN attention backend builds a plan per new sequence length (~1.5 s on the
        # first call with each new instruction length); the other SDPA kernels do not.
        torch.backends.cuda.enable_cudnn_sdp(False)
        self._torch = torch
        self._policy = GrootPolicy.from_pretrained(checkpoint, config=config, local_files_only=True)
        self._policy.to(device).eval()
        self._state_dim = config.input_features["observation.state"].shape[0]
        self._pre, self._post = make_pre_post_processors(
            config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "groot_n1_7_vlm_encode_v1": {"model_name": self.processor, "device": device},
                "device_processor": {"device": device},
            },
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def predict(self, observation: Any) -> np.ndarray:
        torch = self._torch
        state = observation.state
        parts = [state.joints] if state.gripper is None else [state.joints, state.gripper]
        vector = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])
        if vector.shape != (self._state_dim,):
            raise ValueError(f"state has {vector.size} values, checkpoint expects {self._state_dim}")
        if not (observation.instruction or "").strip():
            raise ValueError("instruction must be nonempty text")
        batch = {"observation.state": torch.from_numpy(vector), "task": observation.instruction}
        for field, key in self.image_fields.items():
            frame = np.asarray(getattr(state, field))
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"{field} must be a uint8 (H, W, 3) RGB image")
            # Same float [0, 1] CHW input as LeRobot datasets during training.
            batch[key] = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1), dtype=np.float32) / 255)
        with torch.inference_mode():
            actions = self._post(self._policy.predict_action_chunk(self._pre(batch)))
        actions = actions.detach().float().cpu().numpy()[0]
        if self.clip_columns:
            actions[:, self.clip_columns] = np.clip(actions[:, self.clip_columns], 0.0, 1.0)
        return actions
