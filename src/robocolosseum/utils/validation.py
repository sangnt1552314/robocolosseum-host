"""Action-chunk validation.

This mirrors the checks the Colosseum SDK performs on ``send_action`` so that
problems are caught (and logged) in dry-run mode too, before anything is ever
sent to a robot.
"""

from __future__ import annotations

import numpy as np


def validate_action_chunk(
    action: np.ndarray,
    *,
    max_horizon: int | None = None,
    expected_dim: int | None = None,
) -> np.ndarray:
    """Return the action as a 2D ``(horizon, action_dim)`` float array or raise."""

    arr = np.asarray(action)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(
            f"action must be (horizon, action_dim) or (action_dim,); got ndim={arr.ndim}"
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"action chunk is empty: shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"action must be numeric; got dtype={arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("action contains NaN or Inf values")
    if max_horizon is not None and arr.shape[0] > max_horizon:
        raise ValueError(
            f"action horizon {arr.shape[0]} exceeds max_horizon {max_horizon}"
        )
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(
            f"action_dim {arr.shape[1]} does not match expected {expected_dim}"
        )
    return arr
