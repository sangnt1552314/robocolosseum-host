"""Galaxea G0.5 adapter for so101 and yam (OpenGalaxea/G05 g05-so101, G05 YAM fine-tunes).

Extends the shared DROID adapter (``robocolosseum.policies.galaxea_g05``)
without changing it. Like the Policy Server's G05 workers, it runs upstream
GalaxeaVLA ``scripts/serve_policy.py`` in-process (``load_config_from_run_dir``,
``filter_embodiment``, ``setup``, ``build_obs_dict``, ``PolicyInferencer``), so
normalization, the relative-joint transform and its inverse stay upstream: the
returned parts are already absolute joint targets.

Differences from the shared adapter, all set through ``policy`` options:

* The saved run config is read from a temporary copy, so nothing is written
  next to the checkpoint. ``checkpoint`` is the weights file
  (``checkpoints/model_state_dict.pt`` or ``model.pt``); ``.hydra/config.yaml``
  and ``dataset_stats.json`` are found above it.
* ``action_tokenizer`` / ``processor``: absolute paths of the shared assets
  (``action_tokenizer.pt``, ``qwen3_5_2b_base_processor/``). Upstream resolves
  them relative to the GalaxeaVLA root. ``action_tokenizer`` defaults to the
  run directory's own copy when it has one.
* ``parts_meta``: a ``configs/data/parts_meta/<name>.yaml`` file that the saved
  config loads but upstream does not ship (yam). Its path replaces that
  relative path in the config copy.
* ``native_attention: false`` (default) disables flash-attention in the vision
  tower (SDPA), as the Policy Server does.
* ``state_features`` sources may be sliced, e.g. ``joints[7:13]``.
* ``padded_images``: G05 camera keys filled with zero images (so101 has no
  left wrist camera).
* ``clip_columns``: ``{column: [low, high]}`` applied to the returned actions
  (yam grippers to [0, 1]).
* The AR head's ``_absent_keys`` do not describe the flow-matching actions the
  checkpoint returns when ``continuous_action`` is on, so they are ignored then.
"""

from __future__ import annotations

import importlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from robocolosseum.config import AppConfig
from robocolosseum.policies.galaxea_g05 import GalaxeaG05Adapter, _chdir

log = logging.getLogger(__name__)

_SLICE = re.compile(r"^(\w+)\[(-?\d*):(-?\d*)\]$")


class G05Adapter(GalaxeaG05Adapter):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        opts = self.policy_cfg.options
        self.action_tokenizer = opts.get("action_tokenizer")
        self.processor_dir = opts.get("processor")
        self.parts_meta = opts.get("parts_meta")
        self.native_attention = bool(opts.get("native_attention", False))
        self.padded_images = list(opts.get("padded_images") or [])
        self.clip_columns = {int(k): tuple(v) for k, v in (opts.get("clip_columns") or {}).items()}
        self._stage: tempfile.TemporaryDirectory | None = None
        self._continuous = False

    def _stage_run_dir(self, ckpt: Path) -> Path:
        run_dir = next((p for p in ckpt.parents if (p / ".hydra" / "config.yaml").is_file()), None)
        if run_dir is None:
            raise FileNotFoundError(f"No .hydra/config.yaml above {ckpt}")
        text = (run_dir / ".hydra" / "config.yaml").read_text()
        if self.parts_meta:
            parts_meta = Path(self.parts_meta).expanduser().absolute()
            if not parts_meta.is_file():
                raise FileNotFoundError(f"policy.parts_meta '{parts_meta}' not found")
            text = text.replace(f"configs/data/parts_meta/{parts_meta.name}", str(parts_meta))
        self._stage = tempfile.TemporaryDirectory(prefix="colosseum-g05-")
        stage = Path(self._stage.name)
        (stage / ".hydra").mkdir()
        (stage / ".hydra" / "config.yaml").write_text(text)
        (stage / "dataset_stats.json").symlink_to(run_dir / "dataset_stats.json")
        (stage / ckpt.name).symlink_to(ckpt)
        if not self.action_tokenizer and (run_dir / "action_tokenizer.pt").is_file():
            self.action_tokenizer = str(run_dir / "action_tokenizer.pt")
        return stage / ckpt.name

    def load(self) -> None:
        if not self.policy_cfg.device.startswith("cuda"):
            raise ValueError("G0.5 inference runs on CUDA; set policy.device: cuda.")
        repo = Path(self.g05_repo or "").expanduser()
        if not (repo / "scripts" / "serve_policy.py").is_file():
            raise FileNotFoundError(f"policy.g05_repo '{repo}' has no scripts/serve_policy.py")
        ckpt = Path(self.policy_cfg.checkpoint).expanduser().absolute()
        if not ckpt.is_file():
            raise FileNotFoundError(f"G0.5 weights file '{ckpt}' not found")
        staged_ckpt = self._stage_run_dir(ckpt)
        assets = {
            "action_tokenizer": (self.action_tokenizer, Path.is_file),
            "processor": (self.processor_dir, Path.is_dir),
        }
        for name, (path, check) in assets.items():
            if not path or not check(Path(path)):
                raise FileNotFoundError(f"policy.{name} '{path}' not found")

        with _chdir(str(repo)):  # the saved config loads upstream files by relative path
            serve = self._load_serve_module(repo)
            from omegaconf import OmegaConf

            if not self.native_attention:
                vision = importlib.import_module("g05.models.g05.qwen35.vision")
                vision._flash_attn_varlen = None
                vision._flash_attn_backend = "sdpa"
            overrides = [f"eval_embodiment={self.eval_embodiment}", *self.overrides]
            cfg = serve.load_config_from_run_dir(staged_ckpt.parent, str(staged_ckpt), overrides)
            OmegaConf.set_struct(cfg, False)
            for key, value in (
                ("tokenizer.vq_config.ckpt_dir", self.action_tokenizer),
                ("model.tokenizer.vq_config.ckpt_dir", self.action_tokenizer),
                ("model.model_arch.AT_CONFIG.ckpt_dir", self.action_tokenizer),
                ("model.model_arch.hf_processor_path", self.processor_dir),
                ("model.processor.tokenizer_params.pretrained_model_name_or_path", self.processor_dir),
            ):
                if OmegaConf.select(cfg, key) is not None:
                    OmegaConf.update(cfg, key, str(Path(value).expanduser().absolute()), merge=False)
            if "embodiment_datasets" in cfg.data:
                serve.filter_embodiment(cfg, self.eval_embodiment)
            policy, processor = serve.setup(cfg, device=self.policy_cfg.device)
            self._inferencer = serve.PolicyInferencer(policy, processor, device=self.policy_cfg.device)

        self._serve, self._processor, self._repo_str = serve, processor, str(repo)
        self._continuous = getattr(policy, "continuous_action", False) is True
        self._log_shape_meta()
        log.info("G0.5 ready (embodiment=%s, continuous=%s)", self.eval_embodiment, self._continuous)

    def _state_part(self, observation: Any, source: Any) -> np.ndarray:
        sources = source if isinstance(source, (list, tuple)) else [source]
        pieces = []
        for name in sources:
            match = _SLICE.match(name)
            if match is None:
                pieces.append(super()._state_part(observation, name))
                continue
            start, stop = (int(g) if g else None for g in match.group(2, 3))
            pieces.append(super()._state_part(observation, match.group(1))[start:stop])
        return np.concatenate(pieces).astype(np.float32)

    def _build_g05_observation(self, observation: Any) -> dict[str, Any]:
        raw = super()._build_g05_observation(observation)
        first = next(iter(raw["images"].values()))
        for key in self.padded_images:
            raw["images"].setdefault(key, np.zeros_like(first))
        raw["embodiment_type"] = self.eval_embodiment
        return raw

    def _convert_action(self, action: Any) -> np.ndarray:
        action = {k: v.detach().float() if hasattr(v, "detach") else v for k, v in action.items()}
        absent = set(action.pop("_absent_keys", None) or ())
        if absent and not self._continuous and absent & set(self.action_keys or action):
            raise ValueError(f"G0.5 AR output is missing action parts {sorted(absent)}")
        actions = super()._convert_action(action)
        for column, (low, high) in self.clip_columns.items():
            values = actions[:, column]
            if np.any((values < low) | (values > high)):
                log.warning("Clipping action column %d (range %.4f..%.4f) to [%g, %g]",
                            column, values.min(), values.max(), low, high)
            actions[:, column] = np.clip(values, low, high)
        return actions

    def close(self) -> None:
        super().close()
        if self._stage is not None:
            self._stage.cleanup()
            self._stage = None
