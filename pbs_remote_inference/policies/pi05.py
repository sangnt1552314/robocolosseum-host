"""LeRobot pi0.5 adapter for fine-tunes that ship saved processors (so101, yam).

The shared ``robocolosseum.policies.pi05`` adapter loads ``PI0Policy`` with the
DROID camera keys. pi0.5 fine-tunes need ``PI05Policy``, their own image slots
and their saved pre/post processors. This adapter mirrors the Colosseum Policy
Server workers (``so101_pi05.py``, ``yam_pi05.py``):

* builds ``PI05Policy`` from the checkpoint config and loads ``model.safetensors``
  strictly (upstream ``from_pretrained`` only prints weight errors);
* the saved processors do the quantile state/action normalisation, the
  state-in-prompt tokens and the tokenizer; actions come back as absolute
  targets in the checkpoint's units and the state layout;
* state = ``joints`` (+ ``gripper``) exactly as received.

``policy`` options in the YAML:

* ``image_keys``: checkpoint image feature -> request image field. Image
  features left out are sent as zero images flagged ``<feature>_is_pad`` (the
  checkpoint's camera padding; needs the hqfang/pi05-so100_101 LeRobot patch).
* ``tokenizer``: local tokenizer for a saved ``tokenizer_processor`` step that
  names a Hub repo (jobs run offline).
* ``num_steps``: flow-matching steps (default 10); the server overrides it per request.
* ``clip_columns`` / ``clip_range``: clip these action columns (e.g. YAM grippers to [0, 1]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robocolosseum.config import AppConfig
from robocolosseum.policies.base import BasePolicyAdapter


class Pi05Adapter(BasePolicyAdapter):
    def __init__(self, config: AppConfig) -> None:
        self.policy_cfg = config.policy
        opts = self.policy_cfg.options
        self.image_keys = dict(opts["image_keys"])
        self.tokenizer = opts.get("tokenizer")
        self.num_steps = int(opts.get("num_steps", 10))
        self.clip_columns = list(opts.get("clip_columns") or [])
        self.clip_range = tuple(opts.get("clip_range") or (0.0, 1.0))
        self.policy: Any = None

    def load(self) -> None:
        import torch
        from safetensors.torch import load_file
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        import lerobot.policies.pi05.processor_pi05  # noqa: F401  registers the saved processor steps

        self._torch = torch
        ckpt = Path(self.policy_cfg.checkpoint)
        device = self.policy_cfg.device
        cfg = PI05Config.from_pretrained(str(ckpt), local_files_only=True)
        dtype = self.policy_cfg.dtype  # LeRobot >= Oct 2026 stores torch.dtype, older a string
        cfg.device, cfg.dtype = device, getattr(torch, dtype) if isinstance(cfg.dtype, torch.dtype) else dtype
        cfg.compile_model = cfg.gradient_checkpointing = False
        cfg.num_inference_steps = self.num_steps
        if (ckpt / "tokenizer/tokenizer.model").is_file():
            cfg.text_tokenizer_name = str(ckpt / "tokenizer/tokenizer.model")
        unknown = set(self.image_keys) - set(cfg.image_features)
        if unknown:
            raise ValueError(f"image_keys {sorted(unknown)} not in checkpoint {list(cfg.image_features)}")
        self.pad_keys = [key for key in cfg.image_features if key not in self.image_keys]
        self.pad_shapes = {key: tuple(cfg.image_features[key].shape) for key in self.pad_keys}

        policy = PI05Policy(cfg)
        weights = policy._fix_pytorch_state_dict_keys(load_file(str(ckpt / "model.safetensors")), cfg)
        weights = {k if k.startswith("model.") else f"model.{k}": v for k, v in weights.items()}
        missing, unexpected = policy.load_state_dict(policy._prepare_pretrained_state_dict(weights), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"weights do not match PI05Policy: missing {missing[:5]}, unexpected {unexpected[:5]}")
        del weights
        self.policy = policy.to(device).eval()

        overrides = {"device_processor": {"device": device}}
        if self.tokenizer:
            overrides["tokenizer_processor"] = {"tokenizer_name": str(self.tokenizer)}
        self.pre, self.post = make_pre_post_processors(
            cfg,
            pretrained_path=str(ckpt),
            preprocessor_overrides=overrides,
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def predict(self, observation: Any) -> np.ndarray:
        torch = self._torch
        obs = observation.state
        state = np.asarray(obs.joints, dtype=np.float32).reshape(-1)
        if obs.gripper is not None:
            state = np.concatenate([state, np.asarray(obs.gripper, dtype=np.float32).reshape(-1)])
        batch = {"observation.state": torch.tensor(state), "task": observation.instruction}
        for key, field in self.image_keys.items():
            image = np.asarray(getattr(obs, field))
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{field} must be uint8 RGB (H, W, 3), got {image.dtype} {image.shape}")
            batch[key] = torch.tensor(image).permute(2, 0, 1).float() / 255  # CHW in [0, 1]
        for key in self.pad_keys:
            batch[key] = torch.zeros(self.pad_shapes[key])
        if self.pad_keys:
            for key in [*self.image_keys, *self.pad_keys]:
                batch[f"{key}_is_pad"] = torch.tensor([key in self.pad_keys])  # already batched

        self.policy.reset()  # one observation per request, no action queue
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(batch), num_steps=self.num_steps)
            actions = self.post(chunk)
        actions = actions.detach().float().cpu().numpy()[0]
        if self.clip_columns:
            actions[:, self.clip_columns] = np.clip(actions[:, self.clip_columns], *self.clip_range)
        return actions
