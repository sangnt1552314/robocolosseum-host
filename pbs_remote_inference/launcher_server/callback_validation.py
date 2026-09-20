"""Callback URL validation (SSRF protection).

External users supply the ``callback_url`` the GPU worker will connect to. We
must never let that turn the PBS worker into a proxy into internal
infrastructure, so we allow only ``wss://`` URLs on an approved public tunnel
domain allowlist and reject IP literals, localhost and private ranges.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

# Approved public tunnel domains. Override via LAUNCHER_ALLOWED_CALLBACK_DOMAINS
# (comma-separated hostname suffixes).
_DEFAULT_ALLOWED_SUFFIXES = (
    "ngrok-free.app",
    "ngrok-free.dev",
    "ngrok.app",
    "ngrok.io",
    "ngrok.dev",
)

_BLOCKED_HOSTNAMES = {"localhost", "0.0.0.0", "127.0.0.1", "::1", "metadata"}


class CallbackUrlError(ValueError):
    """The callback URL is not an allowed public WSS endpoint."""


def allowed_suffixes() -> tuple[str, ...]:
    raw = os.environ.get("LAUNCHER_ALLOWED_CALLBACK_DOMAINS", "")
    if raw.strip():
        return tuple(s.strip().lower() for s in raw.split(",") if s.strip())
    return _DEFAULT_ALLOWED_SUFFIXES


def validate_callback_url(url: str) -> str:
    """Return the URL if it is a safe public WSS endpoint, else raise."""
    if len(url) > 2048:
        raise CallbackUrlError("callback_url is too long")

    parsed = urlparse(url)
    if parsed.scheme != "wss":
        raise CallbackUrlError("callback_url must use the wss:// scheme")

    host = (parsed.hostname or "").lower()
    if not host:
        raise CallbackUrlError("callback_url has no host")
    if host in _BLOCKED_HOSTNAMES:
        raise CallbackUrlError("callback_url host is not allowed")

    # Reject IP literals outright: only approved DNS domains are permitted.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        raise CallbackUrlError("callback_url must be a hostname, not an IP address")

    suffixes = allowed_suffixes()
    if not any(host == s or host.endswith("." + s) for s in suffixes):
        raise CallbackUrlError(
            "callback_url host is not on the approved tunnel domain allowlist"
        )
    return url
