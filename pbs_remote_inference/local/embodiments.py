"""Concrete :class:`RobotInterface` implementations and the embodiment registry.

The networking / safety layers depend only on
:class:`~pbs_remote_inference.local.robot_interface.RobotInterface`. Adding a new
embodiment (franka, bimanual_yam, g1, ...) is two steps:

1. implement ``RobotInterface`` for it (here or in its own module), and
2. register a factory in :data:`ROBOT_FACTORIES`.

Nothing in ``run_robot.py`` / ``robot_server.py`` changes. Hardware libraries
(lerobot, OpenCV) are imported lazily inside :meth:`SO101Robot.connect` so the
module -- and the simulated ``DummyRobot`` path -- never need them installed.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pbs_remote_inference.local.robot_interface import (
    DummyRobot,
    ObservationData,
    RobotInterface,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SO101 follower
# ---------------------------------------------------------------------------

# The five arm joints reported under robot_state["joints"] and the gripper,
# reported separately, to satisfy the worker adapter contract
# (worker/observation.py requires a "joints" key and treats "gripper"
# specially). A SO101-trained policy returns an action chunk ordered exactly as
# SO101_ACTION_ORDER, in the same arm-frame degree convention the arm reports.
SO101_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
SO101_GRIPPER = "gripper"
SO101_ACTION_ORDER = (*SO101_ARM_JOINTS, SO101_GRIPPER)

# Max change per executed step (degrees for arm joints, percent for gripper).
# Tune down before trusting real execution.
SO101_MAX_STEP = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 15.0,
    "elbow_flex": 15.0,
    "wrist_flex": 10.0,
    "wrist_roll": 10.0,
    "gripper": 20.0,
}


@dataclass
class SO101Config:
    follower_port: str
    follower_id: str = "home_follower"
    instruction: str = "pick up the block"
    # OpenCV camera indices (confirm with a camera-id check on the robot host).
    wrist_camera: int = 0
    external_camera: int = 1
    use_degrees: bool = True
    # Execute only the first N actions of each chunk, then request a fresh obs.
    execute_steps: int = 4
    step_delay: float = 0.1
    enforce_max_step: bool = True
    max_step: dict[str, float] = field(default_factory=lambda: dict(SO101_MAX_STEP))


class SO101Robot(RobotInterface):
    """Real SO101 follower arm plus a wrist and an external USB camera."""

    def __init__(self, config: SO101Config) -> None:
        self.config = config
        self._robot = None
        self._wrist_cap = None
        self._external_cap = None
        self._cv2 = None

    def connect(self) -> None:
        import cv2
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import (
            SOFollowerRobotConfig,
        )

        self._cv2 = cv2
        follower_config = SOFollowerRobotConfig(
            port=self.config.follower_port,
            id=self.config.follower_id,
            use_degrees=self.config.use_degrees,
        )
        self._robot = SOFollower(follower_config)
        self._robot.connect(calibrate=False)
        self._wrist_cap = self._open_camera(self.config.wrist_camera, "wrist")
        self._external_cap = self._open_camera(self.config.external_camera, "external")
        log.info(
            "SO101 connected: port=%s id=%s wrist_cam=%s external_cam=%s",
            self.config.follower_port,
            self.config.follower_id,
            self.config.wrist_camera,
            self.config.external_camera,
        )

    def _open_camera(self, index: int, name: str):
        cap = self._cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {name} camera (index={index})")
        return cap

    def _read_state(self) -> dict[str, float]:
        obs = self._robot.get_observation()
        return {
            k.removesuffix(".pos"): float(v)
            for k, v in obs.items()
            if k.endswith(".pos")
        }

    def _capture_rgb(self, cap, name: str) -> np.ndarray:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame from {name} camera")
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def get_observation(self) -> ObservationData:
        state = self._read_state()
        return ObservationData(
            instruction=self.config.instruction,
            robot_state={
                "joints": [state[name] for name in SO101_ARM_JOINTS],
                "gripper": [state[SO101_GRIPPER]],
            },
            cameras={
                "external": self._capture_rgb(self._external_cap, "external"),
                "wrist": self._capture_rgb(self._wrist_cap, "wrist"),
            },
        )

    def execute_actions(self, actions: np.ndarray) -> None:
        expected = len(SO101_ACTION_ORDER)
        if actions.shape[1] != expected:
            raise ValueError(
                f"SO101 expects action_dim {expected}, got {actions.shape[1]}"
            )
        for i, row in enumerate(actions[: self.config.execute_steps]):
            # Re-read live state: the arm moved since the chunk was requested.
            current = self._read_state()
            target = {name: float(v) for name, v in zip(SO101_ACTION_ORDER, row)}
            if self.config.enforce_max_step and not self._within_step_limits(
                current, target
            ):
                log.error("SO101 step %d exceeds max-step limits; safe stop", i)
                self.safe_stop()
                return
            self._robot.send_action({f"{k}.pos": v for k, v in target.items()})
            if self.config.step_delay > 0:
                time.sleep(self.config.step_delay)

    def _within_step_limits(
        self, current: dict[str, float], target: dict[str, float]
    ) -> bool:
        for name in SO101_ACTION_ORDER:
            limit = self.config.max_step.get(name)
            delta = abs(target[name] - current[name])
            if limit is not None and delta > limit:
                log.warning(
                    "%s delta %.1f exceeds max step %.1f (current=%.1f target=%.1f)",
                    name,
                    delta,
                    limit,
                    current[name],
                    target[name],
                )
                return False
        return True

    def safe_stop(self) -> None:
        if self._robot is None:
            return
        try:
            current = self._read_state()
            self._robot.send_action({f"{k}.pos": v for k, v in current.items()})
            log.warning("SO101 SAFE STOP: holding current position")
        except Exception as exc:  # best-effort hold; never raise from a stop
            log.error("SO101 safe_stop failed: %s", exc)

    def disconnect(self) -> None:
        if self._robot is not None:
            try:
                self._robot.disconnect()
            finally:
                self._robot = None
        for attr in ("_wrist_cap", "_external_cap"):
            cap = getattr(self, attr)
            if cap is not None:
                cap.release()
                setattr(self, attr, None)


# ---------------------------------------------------------------------------
# Embodiment registry
# ---------------------------------------------------------------------------


def _make_dummy(args: argparse.Namespace) -> RobotInterface:
    return DummyRobot(instruction=args.instruction)


def _make_so101(args: argparse.Namespace) -> RobotInterface:
    port = getattr(args, "follower_port", "") or ""
    if not port:
        raise SystemExit("--follower-port is required for embodiment 'so101'.")
    return SO101Robot(
        SO101Config(
            follower_port=port,
            follower_id=args.follower_id,
            instruction=args.instruction,
            wrist_camera=args.wrist_camera,
            external_camera=args.external_camera,
            execute_steps=args.execute_steps,
            step_delay=args.step_delay,
        )
    )


# Register new embodiments here. Each factory maps parsed CLI args to a
# RobotInterface, e.g. "franka", "bimanual_yam", "g1" once implemented.
ROBOT_FACTORIES: dict[str, Callable[[argparse.Namespace], RobotInterface]] = {
    "dummy": _make_dummy,
    "so101": _make_so101,
}


def available_embodiments() -> list[str]:
    return sorted(ROBOT_FACTORIES)


def make_robot(embodiment: str, args: argparse.Namespace) -> RobotInterface:
    try:
        factory = ROBOT_FACTORIES[embodiment]
    except KeyError:
        raise SystemExit(
            f"Unknown embodiment {embodiment!r}. "
            f"Available: {', '.join(available_embodiments())}"
        )
    return factory(args)
