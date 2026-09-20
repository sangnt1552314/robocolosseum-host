"""FastAPI launcher application (control plane only).

Submits, queries and cancels the PBS GPU inference jobs. It never handles robot
observations, actions, camera streams or policy inference -- those flow directly
between the user's local server and the GPU worker.

Run with::

    LAUNCHER_API_TOKEN=... uvicorn pbs_remote_inference.launcher_server.app:app
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .auth import require_token
from .callback_validation import CallbackUrlError, validate_callback_url
from .errors import APIError
from .job_manager import PBSBackend, PBSError, SessionStore
from .model_registry import ModelRegistry, UnknownModelError
from .schemas import (
    CancelResponse,
    HealthResponse,
    JobResponse,
    SessionInfo,
    SubmitJobRequest,
)

logger = logging.getLogger("pbs_launcher")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _redact(token: str) -> str:
    return f"{token[:2]}…({len(token)} chars)" if token else "<empty>"


def create_app(
    backend: PBSBackend | None = None,
    store: SessionStore | None = None,
) -> FastAPI:
    registry = ModelRegistry()
    backend = backend or PBSBackend(registry, project_root=_PROJECT_ROOT)
    store = store or SessionStore()

    app = FastAPI(
        title="PBS Remote Inference Launcher",
        version="1.0",
        description=(
            "Control-plane API to request a model, submit a PBS GPU worker, and "
            "cancel it. Observations/actions never pass through this API. All "
            "endpoints require Bearer LAUNCHER_API_TOKEN."
        ),
    )

    @app.exception_handler(APIError)
    async def _api_error_handler(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error, "detail": exc.detail},
        )

    @app.get("/health", response_model=HealthResponse, dependencies=[Depends(require_token)])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", pbs_available=backend.available())

    @app.get("/models", response_model=list[str], dependencies=[Depends(require_token)])
    async def list_models() -> list[str]:
        return registry.available_models()

    @app.post(
        "/jobs",
        response_model=JobResponse,
        status_code=202,
        dependencies=[Depends(require_token)],
    )
    async def submit_job(req: SubmitJobRequest) -> JobResponse:
        try:
            spec = registry.get(req.model)
        except UnknownModelError:
            raise APIError(
                422,
                "unsupported_model",
                f"Unknown model. Supported: {registry.available_models()}",
            )
        try:
            callback_url = validate_callback_url(req.callback_url)
        except CallbackUrlError as exc:
            raise APIError(400, "invalid_callback_url", str(exc))

        session_id = store.create(req.model, callback_url)
        logger.info(
            "Session created: session_id=%s model=%s callback_host=%s token=%s",
            session_id,
            req.model,
            callback_url.split("/")[2] if "//" in callback_url else "?",
            _redact(req.callback_token),
        )
        logger.info("PBS script selected: %s", spec.script)
        try:
            job_id = backend.submit(
                spec,
                session_id=session_id,
                callback_url=callback_url,
                callback_token=req.callback_token,
            )
        except PBSError:
            store.set(session_id, status="failed")
            raise APIError(502, "pbs_error", "Failed to submit PBS job.")

        store.set(session_id, job_id=job_id, status="queued")
        logger.info("PBS job submitted: session_id=%s job_id=%s", session_id, job_id)
        return JobResponse(
            session_id=session_id, job_id=job_id, model=req.model, status="queued"
        )

    @app.get(
        "/jobs/{session_id}",
        response_model=SessionInfo,
        dependencies=[Depends(require_token)],
    )
    async def job_status(session_id: str) -> SessionInfo:
        session = store.get(session_id)
        if session is None:
            raise APIError(404, "session_not_found", "No such session.")
        # Refresh from PBS unless already in a terminal local state.
        if session.job_id and session.status not in {"finished", "failed", "cancelled"}:
            try:
                live = backend.job_status(session.job_id)
            except PBSError:
                live = None
            if live is not None:
                store.set(session_id, status=live)
                session.status = live
        return SessionInfo(
            session_id=session.session_id,
            job_id=session.job_id,
            model=session.model,
            callback_url=session.callback_url,
            status=session.status,
            created=session.created,
        )

    @app.delete(
        "/jobs/{session_id}",
        response_model=CancelResponse,
        dependencies=[Depends(require_token)],
    )
    async def cancel_job(session_id: str) -> CancelResponse:
        session = store.get(session_id)
        if session is None:
            raise APIError(404, "session_not_found", "No such session.")
        if session.job_id:
            # Only ever cancel the job id we recorded for this session; users can
            # never pass an arbitrary PBS job id to qdel.
            try:
                backend.cancel(session.job_id)
            except PBSError:
                raise APIError(502, "pbs_error", "Failed to cancel PBS job.")
        store.set(session_id, status="cancelled")
        logger.info(
            "Job cancelled: session_id=%s job_id=%s", session_id, session.job_id
        )
        return CancelResponse(
            session_id=session_id, job_id=session.job_id, status="cancelled"
        )

    return app


# Module-level ASGI app for uvicorn.
app = create_app()
