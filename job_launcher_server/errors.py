"""Shared API error type and machine-readable error codes.

Every failure the client sees is a flat ``{"error": <code>, "detail": <msg>}``
body. Detail messages are always sanitized -- they never contain tokens,
usernames, filesystem paths, or raw Slurm stderr.
"""

from __future__ import annotations


class APIError(Exception):
    def __init__(self, status_code: int, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error = error
        self.detail = detail
