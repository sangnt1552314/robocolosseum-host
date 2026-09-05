#!/usr/bin/env python3
"""Mock MolmoAct2 inference test (no router, no robot).

Loads ``allenai/MolmoAct2-DROID`` from the local cache, feeds a realistic fake
DROID-style observation through the real adapter, and validates the resulting
action chunk. Answers: "Can Hopper load the checkpoint and produce an action?"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robocolosseum.config import load_app_config  # noqa: E402
from robocolosseum.policies.registry import create_policy  # noqa: E402


class _FakeState:
    def __init__(self, left, right, joints, gripper) -> None:
        self.left_image = left
        self.right_image = right
        self.head_image = None
        self.joints = joints
        self.gripper = gripper


class _FakeObservation:
    def __init__(self, state, instruction) -> None:
        self.state = state
        self.instruction = instruction
        self.session_id = "mock"
        self.sequence = 0
        self.control_step = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/molmoact2.example.yaml")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO", format="%(asctime)s [%(levelname)s] %(message)s"
    )
    log = logging.getLogger("mock")

    config = load_app_config(args.config)
    adapter = create_policy(config.policy.name, config)

    log.info("Loading model...")
    adapter.load()
    log.info("Model loaded.")

    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    right = rng.integers(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    # Sample DROID state from the checkpoint README (7 joints + gripper).
    joints = np.array(
        [-0.12726949, -0.30641943, 0.09134164, -2.4143615, -0.26460838, 2.068765, 0.123698],
        dtype=np.float32,
    )
    gripper = np.array([0.0], dtype=np.float32)
    obs = _FakeObservation(
        _FakeState(left, right, joints, gripper),
        "put the black objects into the drawer and close the drawer.",
    )

    # CUDA graph capture makes the first calls slow; warm up before timing.
    log.info("Running warm-up inference...")
    adapter.predict(obs)

    start = time.perf_counter()
    action = np.asarray(adapter.predict(obs))
    latency_ms = (time.perf_counter() - start) * 1000.0

    state = np.concatenate([joints, gripper])
    print("== MolmoAct2 mock inference ==")
    print("model loaded: True")
    print(f"external image shape: {left.shape}")
    print(f"wrist image shape: {right.shape}")
    print(f"state shape: {state.shape}")
    print(f"action shape: {action.shape}")
    print(f"action dtype: {action.dtype}")
    print(f"first action: {action[0]}")
    print(f"inference latency: {latency_ms:.0f} ms")

    assert np.all(np.isfinite(action)), "action contains NaN/Inf values"
    assert action.ndim == 2, f"expected a 2D action chunk, got shape {action.shape}"
    assert action.shape[1] == 8, f"expected action_dim=8, got {action.shape[1]}"
    print("checks passed: finite values, 2D chunk, action_dim=8")

    adapter.close()


if __name__ == "__main__":
    main()
