"""Bearer-token authentication for the launcher API.

The token is read from ``LAUNCHER_API_TOKEN`` at request time, never logged,
never returned in an error, and compared in constant time. This is the launcher
credential; it is distinct from the per-session callback token.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .errors import APIError

_ENV_VAR = "LAUNCHER_API_TOKEN"

bearer_scheme = HTTPBearer(
    auto_error=False, description="Launcher API token (LAUNCHER_API_TOKEN)"
)


async def require_token(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> None:
    expected = os.environ.get(_ENV_VAR)
    if not expected:
        raise APIError(500, "server_not_configured", "Launcher is not configured.")
    presented = credentials.credentials if credentials else None
    if presented is None or not secrets.compare_digest(presented, expected):
        raise APIError(401, "unauthorized", "Missing or invalid API token.")
