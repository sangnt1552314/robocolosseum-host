"""LingBot-VLA v2 adapter (fine-tunes of robbyant/lingbot-vla-v2-6b) for any embodiment.

The shared ``robocolosseum.policies.lingbot_vla_v2`` adapter guesses a Hugging
Face ``AutoModel`` API that these checkpoints do not have. This adapter instead
runs upstream's own inference class, ``deploy.lingbot_vla_v2_policy.
LingbotVLAv2Server`` (github.com/robbyant/lingbot-vla-v2), in process, like the
Colosseum Policy Server workers ``yam_lingbot.py`` and ``so101_lingbot.py``.
Upstream normalizes the state, resizes the images, runs the model and
un-normalizes the actions; the robot config decides whether actions are
absolute or relative to the state. Options under ``policy:`` in the YAML:

* ``source_root``: upstream source checkout; ``processor``: local
  Qwen3-VL-4B-Instruct config/tokenizer files (weights are not needed).
* ``training_config``: the checkpoint's ``lingbotvla_cli.yaml`` (default: in
  the checkpoint folder or three levels above it, upstream's layout).
* ``robot_config``: upstream robot config (path or inline mapping) that maps the
  raw ``observation.state`` / ``action`` columns and camera keys to the model's
  features; ``norm_stats``: its normalization statistics JSON.
* ``image_fields``: request image field -> camera key used in ``robot_config``.
* ``chunk_size``: actions per call (at most the trained chunk).
* ``normalize_instruction``: lowercase and drop trailing ``.?!`` (training did).
* ``clip_columns`` / ``clip_range``: clip these action columns (grippers).
* ``use_compile``: upstream's optional ``torch.compile`` (default off, as in
  the Policy Server).

The model is built in the serving dtype and loaded one shard at a time (as in
``so101_lingbot.py``) instead of upstream's float32 build plus an in-memory
merge of all shards, which needs about 50 GB of host RAM. The resulting weights
are the same as upstream's ``.to(bfloat16)``.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robocolosseum.config import AppConfig
from robocolosseum.policies.base import BasePolicyAdapter

ROBOT = "colosseum"  # name of the staged robot config


class LingbotAdapter(BasePolicyAdapter):
    def __init__(self, config: AppConfig) -> None:
        self.policy = config.policy
        opts = self.policy.options
        self.checkpoint = Path(self.policy.checkpoint).resolve()
        self.source_root = Path(opts["source_root"]).resolve()
        self.processor = Path(opts["processor"]).resolve()
        self.norm_stats = Path(opts["norm_stats"]).resolve()
        self.robot_config = opts["robot_config"]
        self.training_config = opts.get("training_config")
        self.image_fields = dict(opts["image_fields"])
        self.chunk_size = int(opts["chunk_size"])
        self.normalize_instruction = bool(opts.get("normalize_instruction", False))
        self.clip_columns = list(opts.get("clip_columns") or [])
        self.clip_range = tuple(opts.get("clip_range") or (0.0, 1.0))
        self.use_compile = bool(opts.get("use_compile", False))
        self.server: Any = None

    def _training_config(self) -> dict:
        candidates = [self.training_config] if self.training_config else [
            self.checkpoint / "lingbotvla_cli.yaml",
            self.checkpoint.parent.parent.parent / "lingbotvla_cli.yaml",
        ]
        for path in candidates:
            if Path(path).is_file():
                return yaml.safe_load(Path(path).read_text())
        raise FileNotFoundError(f"lingbotvla_cli.yaml not found: {candidates}")

    def _robot_config(self) -> dict:
        robot = self.robot_config
        if not isinstance(robot, dict):
            robot = yaml.safe_load(Path(robot).read_text())
        return {**robot, "norm_stats": str(self.norm_stats)}

    def load(self) -> None:
        import torch
        from safetensors import safe_open

        device = torch.device(self.policy.device)
        if device.type != "cuda":
            raise ValueError("upstream LingBot runs on CUDA only")
        if device.index is not None:
            torch.cuda.set_device(device)  # upstream calls .cuda() / device="cuda"
        dtype = getattr(torch, self.policy.dtype)
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("LingBot dtype must be bfloat16 or float32")

        os.environ["QWEN3VL_PATH"] = str(self.processor)
        sys.path.insert(0, str(self.source_root))
        upstream = importlib.import_module("deploy.lingbot_vla_v2_policy")

        class Server(upstream.LingbotVLAv2Server):
            def load_vla(self, path_to_pi_model):
                previous = torch.get_default_dtype()
                torch.set_default_dtype(dtype)
                try:
                    return super().load_vla(path_to_pi_model)
                finally:
                    torch.set_default_dtype(previous)

            def load_model_weights(self, path_to_pi_model, strict=True):
                expected, loaded = set(self.vla.state_dict()), set()
                for shard in sorted(Path(path_to_pi_model).glob("*.safetensors")):
                    with safe_open(str(shard), framework="pt", device="cpu") as f:
                        weights = {key: f.get_tensor(key).to(dtype) for key in f.keys()}
                    if set(weights) - expected:
                        raise RuntimeError(f"unexpected weights in {shard.name}")
                    self.vla.load_state_dict(weights, strict=False)
                    loaded |= set(weights)
                    del weights
                if loaded != expected:
                    raise RuntimeError(f"{len(expected - loaded)} weights missing from {path_to_pi_model}")

        # Upstream reads lingbotvla_cli.yaml three levels above the weights and
        # configs/robot_configs/<robot>.yaml relative to the working directory:
        # stage both in a temporary directory, leaving the snapshot untouched.
        training = self._training_config()
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="lingbot-") as staging:
            staging = Path(staging)
            weights = staging / "run/checkpoints/step/hf_ckpt"
            weights.parent.mkdir(parents=True)
            weights.symlink_to(self.checkpoint, target_is_directory=True)
            (staging / "run/lingbotvla_cli.yaml").write_text(yaml.safe_dump(training))
            robot = staging / f"configs/robot_configs/{ROBOT}.yaml"
            robot.parent.mkdir(parents=True)
            robot.write_text(yaml.safe_dump(self._robot_config()))
            os.chdir(staging)
            try:
                self.server = Server(
                    path_to_pi_model=str(weights),
                    robot_norm_path=str(self.norm_stats),
                    use_length=self.chunk_size,
                    chunk_ret=True,  # a fresh full chunk on every call
                    use_bf16=dtype == torch.bfloat16,
                    use_fp32=dtype == torch.float32,
                    use_compile=self.use_compile,
                )
                self.server.reset(robo_name=ROBOT)
            finally:
                os.chdir(previous_cwd)

    def predict(self, observation: Any) -> np.ndarray:
        if self.server is None:
            raise RuntimeError("predict() called before load()")
        state = np.array(observation.state.joints, dtype=np.float32).reshape(-1)  # writable copy
        if observation.state.gripper is not None:
            state = np.concatenate([state, np.asarray(observation.state.gripper, np.float32).reshape(-1)])
        task = observation.instruction or ""
        if self.normalize_instruction:
            task = re.sub(r"[.?!]+$", "", task.strip().lower()).strip()
        obs = {"observation.state": state, "task": task}
        for field, key in self.image_fields.items():
            obs[key] = np.array(getattr(observation.state, field), dtype=np.uint8)  # writable copy
        actions = np.array(self.server.infer(obs)["action"], dtype=np.float32)
        if self.clip_columns:
            actions[:, self.clip_columns] = np.clip(actions[:, self.clip_columns], *self.clip_range)
        return actions

    def close(self) -> None:
        self.server = None
