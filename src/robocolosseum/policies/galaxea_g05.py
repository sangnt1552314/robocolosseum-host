"""OpenGalaxea/G05 (G0.5 DROID) policy adapter.

Thin adapter between the RoboColosseum runner and the OFFICIAL GalaxeaVLA G0.5
inference path. G0.5 is normally served over websocket+msgpack by
``scripts/serve_policy.py``; this adapter reuses that script's *own* in-process
building blocks (so behaviour stays identical to the reference server, minus the
socket):

* ``serve_policy.setup(cfg, device)`` -> ``(policy, processor)`` -- loads the
  ``model_state_dict.pt`` via ``load_model_from_checkpoint``, casts to bf16,
  builds the processor and sets its normalizer from ``dataset_stats.json``,
  neutralises the idle action filter, and sets the action horizon;
* ``PolicyInferencer(policy, processor, device)`` runs inference;
* ``serve_policy.build_obs_dict(raw_obs, processor)`` turns a raw observation
  (``{"images": {key: CHW uint8}, "state": {key: 1D}, "task": str}``) into the
  model input, validated against the processor's ``shape_meta``;
* ``inferencer.infer([obs_dict])`` returns a per-part action dict (DROID:
  ``right_arm[7]`` + ``right_gripper[1]``).

Unlike LingBot, G0.5 ships a DROID checkpoint (``g05-droid``) and a
``Droid_Franka`` embodiment, so this runs zero-shot once the repo + checkpoint +
shared resources are present. Prerequisites are validated with clear errors.

Everything embodiment-specific is options-driven (``image_features`` /
``state_features`` map the Colosseum observation to the shape_meta keys), so a
different G0.5 checkpoint/embodiment can be wired via config only.

VERIFY the action semantics against your robot: the DROID action parts may be
EEF deltas rather than absolute joint positions -- confirm before enabling
actions.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AppConfig
from .base import BasePolicyAdapter

log = logging.getLogger(__name__)


@contextmanager
def _chdir(path: str):
    """Temporarily change cwd (G0.5 resolves shared resources by relative path)."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class GalaxeaG05Adapter(BasePolicyAdapter):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.policy_cfg = config.policy
        self.camera = config.camera_mapping

        opts = self.policy_cfg.options
        # Path to the cloned https://github.com/OpenGalaxea/GalaxeaVLA checkout
        # (src-layout `g05` package; shared resources under <repo>/checkpoints).
        self.g05_repo = opts.get("g05_repo")
        # Hydra override selecting the embodiment (DROID default).
        self.eval_embodiment = opts.get("eval_embodiment", "Droid_Franka")
        # Extra Hydra overrides, e.g. ["model.model_arch.discrete_action=true",
        # "model.model_arch.continuous_action=false"] for the discrete checkpoint.
        self.overrides = list(opts.get("overrides", []) or [])
        self.frequency = float(opts.get("frequency", config.runtime.control_hz))

        # Colosseum -> G0.5 raw-observation mapping. `image_features` maps each
        # shape_meta image key to a Colosseum source ("external"/"wrist" via
        # camera_mapping, else a literal observation.state attribute).
        # `state_features` maps each shape_meta state key to a source or a list
        # of sources ("joints"/"gripper"/attr) to concatenate into a 1D vector.
        self.image_features: dict[str, str] = dict(opts.get("image_features", {}) or {})
        self.state_features: dict[str, Any] = dict(opts.get("state_features", {}) or {})
        # Ordered action-part keys to concatenate (None = all predicted parts).
        action_keys = opts.get("action_keys")
        self.action_keys = list(action_keys) if action_keys else None

        self._serve: Any = None          # loaded serve_policy module
        self._processor: Any = None
        self._inferencer: Any = None
        self._repo_str: str = ""

    # ---- loading & validation ------------------------------------------------

    def _validate_prereqs(self) -> tuple[Path, Path]:
        if not self.g05_repo:
            raise ValueError(
                "galaxea_g05 requires policy.options.g05_repo (path to the cloned "
                "https://github.com/OpenGalaxea/GalaxeaVLA checkout with its uv "
                ".venv installed)."
            )
        repo = Path(self.g05_repo).expanduser()
        if not (repo / "src" / "g05").is_dir():
            raise FileNotFoundError(
                f"g05_repo '{repo}' does not contain src/g05. Clone GalaxeaVLA "
                "and install it (uv) there."
            )
        serve_py = repo / "scripts" / "serve_policy.py"
        if not serve_py.is_file():
            raise FileNotFoundError(f"Missing {serve_py} in the GalaxeaVLA repo.")

        ckpt = Path(self.policy_cfg.checkpoint).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"G0.5 checkpoint '{ckpt}' not found. Point policy.checkpoint at "
                "the g05-droid model_state_dict.pt, e.g. "
                "<repo>/checkpoints/g05-droid/checkpoints/model_state_dict.pt "
                "(download with `huggingface-cli download OpenGalaxea/G05 "
                "--local-dir <repo>/checkpoints`)."
            )
        # Checkpoint dir must carry .hydra/config.yaml + dataset_stats.json.
        run_dir = ckpt.parent.parent
        if not (run_dir / ".hydra" / "config.yaml").is_file():
            raise FileNotFoundError(
                f"Missing '{run_dir / '.hydra' / 'config.yaml'}'. A g05 checkpoint "
                "dir must contain .hydra/config.yaml, checkpoints/"
                "model_state_dict.pt and dataset_stats.json."
            )
        # Shared resources at the repo root (loaded by the config).
        for shared in (
            repo / "checkpoints" / "action_tokenizer.pt",
            repo / "checkpoints" / "qwen3_5_2b_base_processor",
        ):
            if not shared.exists():
                log.warning(
                    "Shared resource '%s' not found; download it into "
                    "<repo>/checkpoints (see GalaxeaVLA README).",
                    shared,
                )
        return repo, ckpt

    def _load_serve_module(self, repo: Path) -> Any:
        # g05 is a src-layout package; put src (and repo root for `scripts`/
        # rootutils) on the path before importing the server module, which
        # imports `from g05...` and registers OmegaConf resolvers at import time.
        for p in (str(repo / "src"), str(repo)):
            if p not in sys.path:
                sys.path.insert(0, p)
        serve_py = repo / "scripts" / "serve_policy.py"
        spec = importlib.util.spec_from_file_location("g05_serve_policy", serve_py)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def load(self) -> None:
        if not self.policy_cfg.device.startswith("cuda"):
            raise ValueError("G0.5 inference runs on CUDA; set policy.device: cuda.")
        repo, ckpt = self._validate_prereqs()
        repo_str = str(repo)

        log.info("Loading OpenGalaxea/G0.5")
        log.info("  checkpoint:  %s", ckpt)
        log.info("  embodiment:  %s", self.eval_embodiment)
        log.info("  overrides:   %s", self.overrides or "<none>")
        log.info("  device:      %s", self.policy_cfg.device)

        # setup() loads the qwen processor / action tokenizer / dataset stats by
        # paths that are relative to the repo root, so run it from there.
        with _chdir(repo_str):
            serve = self._load_serve_module(repo)
            overrides = list(self.overrides)
            overrides.append(f"eval_embodiment={self.eval_embodiment}")
            run_dir = serve.find_run_dir(str(ckpt))
            cfg = serve.load_config_from_run_dir(run_dir, str(ckpt), overrides)
            eval_embodiment = cfg.get("eval_embodiment", None)
            if eval_embodiment and "embodiment_datasets" in cfg.data:
                serve.filter_embodiment(cfg, eval_embodiment)
            policy, processor = serve.setup(cfg, device=self.policy_cfg.device)
            inferencer = serve.PolicyInferencer(
                policy, processor, device=self.policy_cfg.device
            )

        self._serve = serve
        self._processor = processor
        self._inferencer = inferencer
        self._repo_str = repo_str

        self._log_shape_meta()
        log.info("G0.5 ready (embodiment=%s)", self.eval_embodiment)

    def _shape_meta(self) -> dict[str, Any] | None:
        p = self._processor
        sm = getattr(p, "shape_meta", None)
        if sm is None and getattr(p, "processors", None):
            sm = getattr(next(iter(p.processors.values())), "shape_meta", None)
        return sm

    def _log_shape_meta(self) -> None:
        sm = self._shape_meta()
        if not sm:
            return
        imgs = [m.get("key") for m in sm.get("images", [])]
        states = [(m.get("key"), m.get("raw_shape")) for m in sm.get("state", [])]
        actions = [m.get("key") for m in sm.get("action", [])]
        log.info("  expected image keys:  %s", imgs)
        log.info("  expected state keys:  %s", states)
        log.info("  action parts:         %s", actions)
        if not self.image_features:
            log.warning(
                "policy.options.image_features is empty; map the image keys above "
                "to Colosseum sources (e.g. {%s: external}).",
                imgs[0] if imgs else "cam",
            )
        if not self.state_features:
            log.warning(
                "policy.options.state_features is empty; map the state keys above "
                "to Colosseum sources (joints/gripper/attr)."
            )

    # ---- observation / action conversion ------------------------------------

    def _resolve_image(self, observation: Any, source: str) -> np.ndarray:
        if source == "external":
            source = self.camera.external
        elif source == "wrist":
            source = self.camera.wrist
        img = getattr(observation.state, source, None)
        if img is None:
            raise ValueError(f"Camera field '{source}' missing from the observation")
        img = np.asarray(img)
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"Expected an HWC RGB image for '{source}', got shape {img.shape}"
            )
        # G0.5 expects [C, H, W] uint8.
        return np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.uint8)

    def _state_part(self, observation: Any, source: Any) -> np.ndarray:
        sources = source if isinstance(source, (list, tuple)) else [source]
        pieces = []
        for name in sources:
            if name == "joints":
                value = observation.state.joints
            elif name == "gripper":
                value = observation.state.gripper
            else:
                value = getattr(observation.state, name, None)
            if value is None:
                raise ValueError(f"State source '{name}' missing from the observation")
            pieces.append(np.asarray(value, dtype=np.float32).reshape(-1))
        return np.concatenate(pieces).astype(np.float32)

    def _build_g05_observation(self, observation: Any) -> dict[str, Any]:
        if not self.image_features:
            raise ValueError(
                "policy.options.image_features is empty. Map each shape_meta image "
                "key (logged at load) to a Colosseum source, e.g. "
                "{exterior_image_1_left: external, wrist_image_left: wrist}."
            )
        if not self.state_features:
            raise ValueError(
                "policy.options.state_features is empty. Map each shape_meta state "
                "key (logged at load) to a Colosseum source, e.g. "
                "{joint_position: joints, gripper_position: gripper}."
            )
        images = {
            key: self._resolve_image(observation, source)
            for key, source in self.image_features.items()
        }
        state = {
            key: self._state_part(observation, source)
            for key, source in self.state_features.items()
        }
        return {
            "images": images,
            "state": state,
            "task": observation.instruction or "",
            "frequency": self.frequency,
        }

    def _convert_action(self, action: Any) -> np.ndarray:
        if not isinstance(action, dict):
            actions = np.asarray(action, dtype=np.float32)
            return actions[np.newaxis, :] if actions.ndim == 1 else actions
        action = dict(action)
        action.pop("_cot_text", None)
        for absent in action.pop("_absent_keys", set()) or set():
            action.pop(absent, None)
        keys = self.action_keys or [k for k in action if not str(k).startswith("_")]
        parts = []
        for key in keys:
            if key not in action:
                raise KeyError(
                    f"Action part '{key}' not in G0.5 output {list(action)}"
                )
            value = action[key]
            if hasattr(value, "detach"):
                value = value.detach().to("cpu").numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.ndim == 3 and value.shape[0] == 1:  # (1, chunk, dim)
                value = value[0]
            if value.ndim == 1:
                value = value[np.newaxis, :]
            parts.append(value)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def predict(self, observation: Any) -> np.ndarray:
        if self._inferencer is None:
            raise RuntimeError("predict() called before load()")
        raw_obs = self._build_g05_observation(observation)
        obs_dict = self._serve.build_obs_dict(raw_obs, self._processor)
        actions = self._inferencer.infer([obs_dict])
        return self._convert_action(actions[0])

    def reset(self) -> None:
        # G0.5's chunk caching lives in the (unused) websocket wrapper; in-process
        # we recompute each predict, so there is no per-rollout state to clear.
        log.debug("G0.5 reset (stateless in-process inference)")

    def close(self) -> None:
        self._inferencer = None
        self._processor = None
        self._serve = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
