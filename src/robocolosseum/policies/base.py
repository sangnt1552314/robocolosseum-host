"""Common adapter interface shared by every policy.

A policy adapter is the only model-specific code the runner touches. The runner
never imports a concrete model; it goes through the registry and this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePolicyAdapter(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load the checkpoint/model onto the GPU."""

    @abstractmethod
    def predict(self, observation: Any) -> np.ndarray:
        """Convert a Colosseum observation into model input, run inference,
        and return an action chunk shaped ``(horizon, action_dim)``.
        """

    def reset(self) -> None:
        """Optional reset between robot rollouts / sessions."""

    def close(self) -> None:
        """Optional cleanup before the process exits."""
