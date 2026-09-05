"""Configuration loading for the policy worker.

Two sources are kept deliberately separate:

* ``AppConfig`` comes from a committed YAML file (policy, camera mapping,
  runtime timeouts). It never contains secrets.
* ``RouterConfig`` (router URL + token) comes from environment variables or a
  private, git-ignored YAML file. It is never committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PolicyConfig:
    name: str
    checkpoint: str
    dtype: str = "bfloat16"
    device: str = "cuda"
    # Adapter-specific extras (e.g. norm_tag, num_steps) forwarded to the adapter.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraMapping:
    """Maps model camera slots to Colosseum observation field names."""

    external: str = "left_image"
    wrist: str = "right_image"


@dataclass
class RuntimeConfig:
    startup_timeout_seconds: float = 600.0
    idle_timeout_seconds: float = 600.0
    action_spaces: tuple[str, ...] = ("joint_position",)
    control_hz: int = 15
    max_horizon: int = 16


@dataclass
class AppConfig:
    policy: PolicyConfig
    camera_mapping: CameraMapping
    runtime: RuntimeConfig


@dataclass
class RouterConfig:
    url: str
    token: str


def load_app_config(path: str | Path) -> AppConfig:
    """Load the committed policy/runtime YAML (no secrets)."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    policy_raw = dict(data.get("policy", {}))
    try:
        name = policy_raw.pop("name")
        checkpoint = policy_raw.pop("checkpoint")
    except KeyError as exc:
        raise ValueError(f"policy config is missing required key: {exc}") from exc
    dtype = policy_raw.pop("dtype", "bfloat16")
    device = policy_raw.pop("device", "cuda")
    policy = PolicyConfig(
        name=name,
        checkpoint=checkpoint,
        dtype=dtype,
        device=device,
        options=policy_raw,
    )

    cam = data.get("camera_mapping", {}) or {}
    camera = CameraMapping(
        external=cam.get("external", "left_image"),
        wrist=cam.get("wrist", "right_image"),
    )

    rt = data.get("runtime", {}) or {}
    runtime = RuntimeConfig(
        startup_timeout_seconds=float(rt.get("startup_timeout_seconds", 600)),
        idle_timeout_seconds=float(rt.get("idle_timeout_seconds", 600)),
        action_spaces=tuple(rt.get("action_spaces", ["joint_position"])),
        control_hz=int(rt.get("control_hz", 15)),
        max_horizon=int(rt.get("max_horizon", 16)),
    )

    return AppConfig(policy=policy, camera_mapping=camera, runtime=runtime)


def load_router_config(path: str | Path | None = None) -> RouterConfig:
    """Load router credentials from env vars (preferred) or a private YAML file.

    Environment variables ``COLOSSEUM_ROUTER_URL`` and ``COLOSSEUM_TOKEN`` take
    precedence. The optional file must contain only ``url`` and ``token``.
    """

    url = os.environ.get("COLOSSEUM_ROUTER_URL")
    token = os.environ.get("COLOSSEUM_TOKEN")

    if (not url or not token) and path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        url = url or data.get("url")
        token = token or data.get("token")

    if not url or not token:
        raise ValueError(
            "Router credentials not found. Set COLOSSEUM_ROUTER_URL and "
            "COLOSSEUM_TOKEN, or pass a private YAML file containing 'url' and "
            "'token'."
        )
    return RouterConfig(url=str(url), token=str(token))
