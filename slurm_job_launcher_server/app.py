"""FastAPI launcher application.

A small control-plane API that submits, queries, lists and cancels the
RoboColosseum GPU policy jobs on the NUS SoC Slurm cluster. It never handles
robot observations, actions, camera streams or policy inference -- those stay
inside the GPU policy worker.
"""

from __future__ import annotations

import logging
import threading

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .auth import require_token
from .config import LauncherConfig, load_config
from .errors import APIError
from .registry import PolicyRegistry, UnknownModelError
from .schemas import (
    CancelResponse,
    HealthResponse,
    JobListResponse,
    JobResponse,
    JobSummary,
    SubmitJobRequest,
)
from .slurm_backend import SlurmBackend, SlurmError

logger = logging.getLogger("launcher")


def create_app(
    config: LauncherConfig | None = None,
    backend: SlurmBackend | None = None,
) -> FastAPI:
    config = config or load_config()
    registry = PolicyRegistry(config)
    backend = backend or SlurmBackend(config, registry)

    # Single admission lock. Combined with a single Uvicorn worker (V1), this
    # serialises "check limits -> submit" so concurrent Franka launch requests
    # cannot both slip past the same limit.
    admission_lock = threading.Lock()
    # Fast-path idempotency cache; the durable backstop is the Slurm --comment.
    idempotency: dict[str, str] = {}

    app = FastAPI(
        title="RoboColosseum Job Launcher",
        version="1.0",
        description=(
            "Control-plane API to submit, query, list and cancel GPU policy "
            "jobs on the NUS SoC Slurm cluster. All endpoints require a Bearer "
            "token (LAUNCHER_API_TOKEN). Interactive docs: /docs and /redoc."
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
        return HealthResponse(status="ok", slurm_available=backend.available())

    @app.post(
        "/jobs",
        response_model=JobResponse,
        status_code=202,
        dependencies=[Depends(require_token)],
    )
    async def submit_job(req: SubmitJobRequest) -> JobResponse:
        try:
            entry = registry.get(req.model)
        except UnknownModelError:
            raise APIError(
                422,
                "unsupported_model",
                f"Unknown model. Supported: {registry.available_models()}",
            )

        request_id = req.request_id or None

        with admission_lock:
            # 1. Idempotency: return the existing job for a repeated request_id.
            if request_id:
                existing = _lookup_existing(backend, idempotency, request_id)
                if existing is not None:
                    return JobResponse(
                        job_id=existing.job_id,
                        model=existing.model or req.model,
                        state=existing.state,
                        request_id=request_id,
                    )

            # 2. Global limit.
            try:
                active_total = backend.count_active_jobs()
            except SlurmError:
                raise APIError(502, "slurm_error", "Could not query Slurm.")
            if active_total >= config.max_active_jobs:
                raise APIError(
                    409,
                    "global_limit_reached",
                    f"Maximum active policy jobs reached ({config.max_active_jobs}).",
                )

            # 3. Per-model replica limit.
            active_model = backend.count_active_model_jobs(req.model)
            if active_model >= entry.max_replicas:
                raise APIError(
                    429,
                    "model_replica_limit_reached",
                    f"Maximum replicas for '{req.model}' reached ({entry.max_replicas}).",
                )

            # 4. Submit.
            try:
                job_id = backend.submit(entry, request_id)
            except SlurmError:
                raise APIError(502, "slurm_error", "Failed to submit job to Slurm.")

            if request_id:
                idempotency[request_id] = job_id

        return JobResponse(
            job_id=job_id,
            model=req.model,
            state="PENDING",
            request_id=request_id,
        )

    @app.get("/jobs", response_model=JobListResponse, dependencies=[Depends(require_token)])
    async def list_jobs() -> JobListResponse:
        try:
            jobs = backend.list_jobs()
        except SlurmError:
            raise APIError(502, "slurm_error", "Could not query Slurm.")
        return JobListResponse(
            jobs=[JobSummary(job_id=j.job_id, model=j.model, state=j.state) for j in jobs]
        )

    @app.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_token)])
    async def job_status(job_id: str) -> JobResponse:
        _validate_job_id(job_id)
        try:
            job = backend.get_job(job_id)
        except SlurmError:
            raise APIError(502, "slurm_error", "Could not query Slurm.")
        if job is None:
            raise APIError(404, "job_not_found", "No such job.")
        return JobResponse(job_id=job.job_id, model=job.model, state=job.state)

    @app.post(
        "/jobs/{job_id}/cancel",
        response_model=CancelResponse,
        dependencies=[Depends(require_token)],
    )
    async def cancel_job(job_id: str) -> CancelResponse:
        _validate_job_id(job_id)
        try:
            job = backend.get_job(job_id)
        except SlurmError:
            raise APIError(502, "slurm_error", "Could not query Slurm.")
        if job is None:
            raise APIError(404, "job_not_found", "No such job.")
        # Only cancel jobs this launcher owns and recognises as a policy job.
        if job.model is None or not registry.is_launcher_job_name(job.name):
            raise APIError(
                403,
                "not_a_launcher_job",
                "This job is not a launcher-managed policy job.",
            )
        try:
            backend.cancel(job_id)
        except SlurmError:
            raise APIError(502, "slurm_error", "Failed to cancel job.")
        return CancelResponse(job_id=job_id, status="cancel_requested")

    return app


def _lookup_existing(backend: SlurmBackend, cache: dict[str, str], request_id: str):
    """Resolve a prior submission for request_id from the cache, then Slurm."""

    cached = cache.get(request_id)
    if cached:
        job = backend.get_job(cached)
        if job is not None:
            return job
    found = backend.find_by_request_id(request_id)
    if found is not None:
        cache[request_id] = found.job_id
    return found


def _validate_job_id(job_id: str) -> None:
    if not job_id.isdigit():
        raise APIError(400, "invalid_job_id", "Job id must be numeric.")


# Module-level ASGI app for `uvicorn slurm_job_launcher_server.app:app`.
app = create_app()
