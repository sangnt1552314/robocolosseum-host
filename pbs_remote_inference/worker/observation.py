"""Bridge an ``observation`` protocol message to the adapter's object interface.

The existing adapters read plain attributes (``observation.instruction``,
``observation.state.joints`` / ``.gripper`` and camera attributes named by
``camera_mapping``). This builds such an object from our wire format, mapping the
generic camera slots (``external`` / ``wrist``) onto the attribute names each
adapter expects. No model code is duplicated or modified.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pbs_remote_inference.protocol import Observation, decode_jpeg
from robocolosseum.config import CameraMapping


def to_adapter_observation(
    observation: Observation, camera_mapping: CameraMapping
) -> SimpleNamespace:
    if "joints" not in observation.robot_state:
        raise ValueError("robot_state must include 'joints' for the current adapters")

    state: dict[str, object] = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in observation.robot_state.items()
    }
    state.setdefault("gripper", None)

    for slot, encoded in observation.cameras.items():
        image = decode_jpeg(encoded)
        state[slot] = image
        if slot == "external":
            state[camera_mapping.external] = image
        elif slot == "wrist":
            state[camera_mapping.wrist] = image

    return SimpleNamespace(
        instruction=observation.instruction,
        state=SimpleNamespace(**state),
    )
