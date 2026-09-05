"""Small timing helpers."""

from __future__ import annotations

import time


class Timer:
    """Context manager that measures wall-clock latency in milliseconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def monotonic() -> float:
    return time.monotonic()
