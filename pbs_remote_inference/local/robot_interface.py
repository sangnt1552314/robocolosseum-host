"""Robot-side abstraction and a dummy implementation for networking tests.

Version 1 ships only a :class:`DummyRobot`; wiring a real robot means
implementing :class:`RobotInterface` for that embodiment. The networking layer
depends only on this interface, never on a concrete robot.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ObservationData:
    """A single robot observation in generic, embodiment-agnostic form."""

    instruction: str
    robot_state: dict[str, list[float]]
    # generic camera slot name -> HWC RGB uint8 image.
    cameras: dict[str, np.ndarray] = field(default_factory=dict)


class RobotInterface(ABC):
    @abstractmethod
    def get_observation(self) -> ObservationData:
        """Read the current cameras, robot state and task instruction."""

    @abstractmethod
    def execute_actions(self, actions: np.ndarray) -> None:
        """Execute a validated ``(horizon, action_dim)`` action chunk."""

    @abstractmethod
    def safe_stop(self) -> None:
        """Enter a safe hold/stop state (timeout, disconnect, invalid action)."""

    def connect(self) -> None:
        """Open hardware connections. No-op for simulated robots."""

    def disconnect(self) -> None:
        """Release hardware connections. No-op for simulated robots."""


class DummyRobot(RobotInterface):
    """Deterministic fake robot: fixed state + random camera frames."""

    def __init__(
        self,
        instruction: str = "pick up the cup",
        height: int = 64,
        width: int = 64,
        cameras: tuple[str, ...] = ("external", "wrist"),
        seed: int = 0,
    ) -> None:
        self.instruction = instruction
        self.height = height
        self.width = width
        self.camera_names = cameras
        self._rng = np.random.default_rng(seed)
        self._joints = [-0.127, -0.306, 0.091, -2.414, -0.265, 2.069, 0.124]
        self._gripper = [0.0]

    def get_observation(self) -> ObservationData:
        cameras = {
            name: self._rng.integers(
                0, 256, (self.height, self.width, 3), dtype=np.uint8
            )
            for name in self.camera_names
        }
        return ObservationData(
            instruction=self.instruction,
            robot_state={"joints": list(self._joints), "gripper": list(self._gripper)},
            cameras=cameras,
        )

    def execute_actions(self, actions: np.ndarray) -> None:
        log.info(
            "DummyRobot executing action chunk: horizon=%d dim=%d first=%s",
            actions.shape[0],
            actions.shape[1],
            np.array2string(actions[0], precision=3, max_line_width=120),
        )

    def safe_stop(self) -> None:
        log.warning("DummyRobot entering SAFE STOP (hold position)")
