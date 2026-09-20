"""Local-side action validation and safe-state policy.

Robot-specific limits live in :class:`SafetyLimits` so the generic networking
code never hardcodes one embodiment. The GPU worker is never trusted: every
action chunk is validated here before it can reach the robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbs_remote_inference.protocol import Action


class UnsafeActionError(Exception):
    """Raised when a returned action chunk must not be executed."""


@dataclass
class SafetyLimits:
    expected_action_dim: int | None = None
    max_horizon: int | None = None
    min_value: float | None = None
    max_value: float | None = None


def validate_action(
    response: Action,
    expected_request_id: str,
    sent_monotonic: float,
    now_monotonic: float,
    timeout_seconds: float,
    limits: SafetyLimits | None = None,
) -> np.ndarray:
    """Validate a worker response and return a finite ``(horizon, dim)`` chunk.

    Raises :class:`UnsafeActionError` on any failed check so the caller can
    trigger a safe stop instead of executing an unsafe or stale action.
    """
    limits = limits or SafetyLimits()

    if response.request_id != expected_request_id:
        raise UnsafeActionError(
            f"stale response: expected request_id {expected_request_id!r}, "
            f"got {response.request_id!r}"
        )

    latency = now_monotonic - sent_monotonic
    if latency < 0 or latency > timeout_seconds:
        raise UnsafeActionError(
            f"stale response: latency {latency:.3f}s exceeds "
            f"timeout {timeout_seconds:.3f}s"
        )

    actions = np.asarray(response.actions, dtype=np.float64)
    if actions.ndim != 2 or actions.size == 0:
        raise UnsafeActionError(f"expected a 2D action chunk, got shape {actions.shape}")
    if not np.isfinite(actions).all():
        raise UnsafeActionError("action chunk contains NaN or Inf")

    horizon, action_dim = actions.shape
    if limits.expected_action_dim is not None and action_dim != limits.expected_action_dim:
        raise UnsafeActionError(
            f"expected action_dim {limits.expected_action_dim}, got {action_dim}"
        )
    if limits.max_horizon is not None and horizon > limits.max_horizon:
        raise UnsafeActionError(
            f"horizon {horizon} exceeds the maximum {limits.max_horizon}"
        )
    if limits.min_value is not None and float(actions.min()) < limits.min_value:
        raise UnsafeActionError(
            f"action value {float(actions.min())} below limit {limits.min_value}"
        )
    if limits.max_value is not None and float(actions.max()) > limits.max_value:
        raise UnsafeActionError(
            f"action value {float(actions.max())} above limit {limits.max_value}"
        )

    return actions.astype(np.float32)
