"""Bearer-token authentication.

The token is read from the ``LAUNCHER_API_TOKEN`` environment variable at
request time. It is never logged, never returned in an error, and compared in
constant time.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Request

from .errors import APIError

_ENV_VAR = "LAUNCHER_API_TOKEN"


def _extract_bearer(header: str | None) -> str | None:
    if not header:
        return None
    parts = header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


async def require_token(request: Request) -> None:
    """FastAPI dependency enforcing ``Authorization: Bearer <token>``."""

    expected = os.environ.get(_ENV_VAR)
    if not expected:
        # Fail closed but do not leak that the server is misconfigured beyond a
        # generic message; the launcher.sh script validates this at startup.
        raise APIError(500, "server_not_configured", "Launcher is not configured.")

    presented = _extract_bearer(request.headers.get("Authorization"))
    if presented is None or not secrets.compare_digest(presented, expected):
        raise APIError(401, "unauthorized", "Missing or invalid API token.")
