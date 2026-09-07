"""Bearer-token authentication.

The token is read from the ``LAUNCHER_API_TOKEN`` environment variable at
request time. It is never logged, never returned in an error, and compared in
constant time.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .errors import APIError

_ENV_VAR = "LAUNCHER_API_TOKEN"

# Declared as an OpenAPI security scheme so the Swagger UI (/docs) shows an
# "Authorize" button. auto_error=False lets us return our own JSON error shape.
bearer_scheme = HTTPBearer(
    auto_error=False, description="Launcher API token (LAUNCHER_API_TOKEN)"
)


async def require_token(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> None:
    """FastAPI dependency enforcing ``Authorization: Bearer <token>``."""

    expected = os.environ.get(_ENV_VAR)
    if not expected:
        # Fail closed but do not leak that the server is misconfigured beyond a
        # generic message; the launcher.sh script validates this at startup.
        raise APIError(500, "server_not_configured", "Launcher is not configured.")

    presented = credentials.credentials if credentials else None
    if presented is None or not secrets.compare_digest(presented, expected):
        raise APIError(401, "unauthorized", "Missing or invalid API token.")
