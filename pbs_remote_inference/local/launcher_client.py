"""Thin async HTTP client for the launcher control-plane API.

Sends the launcher API key as ``Authorization: Bearer``. Never logs the key or
the per-session callback token.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)


class LauncherError(Exception):
    """A launcher API call failed."""


class LauncherClient:
    def __init__(self, base_url: str, api_token: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_token}"}
        self._timeout = timeout

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.request(
                    method, url, headers=self._headers, **kwargs
                )
            except httpx.HTTPError as exc:
                raise LauncherError(f"launcher request failed: {exc}") from exc
        if response.status_code >= 400:
            detail = _safe_detail(response)
            raise LauncherError(
                f"launcher returned {response.status_code}: {detail}"
            )
        return response.json() if response.content else {}

    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def models(self) -> list[str]:
        return await self._request("GET", "/models")  # type: ignore[return-value]

    async def submit(self, model: str, callback_url: str, callback_token: str) -> dict:
        return await self._request(
            "POST",
            "/jobs",
            json={
                "model": model,
                "callback_url": callback_url,
                "callback_token": callback_token,
            },
        )

    async def status(self, session_id: str) -> dict:
        return await self._request("GET", f"/jobs/{session_id}")

    async def cancel(self, session_id: str) -> dict:
        return await self._request("DELETE", f"/jobs/{session_id}")


def _safe_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        return str(body.get("detail") or body.get("error") or "")
    except Exception:  # noqa: BLE001
        return ""
