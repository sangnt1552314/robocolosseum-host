"""Request/response models for the launcher API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubmitJobRequest(BaseModel):
    model: str = Field(..., description="Registered public model name, e.g. 'molmoact2'.")
    callback_url: str = Field(
        ..., description="wss:// URL of the user's local worker endpoint (via ngrok)."
    )
    # Per-session token used only between the PBS worker and the user's local
    # server. Restricted charset so it is safe to pass through `qsub -v`.
    callback_token: str = Field(
        ...,
        min_length=8,
        max_length=128,
        pattern=r"^[A-Za-z0-9._\-]+$",
        description="Temporary per-session token (worker <-> local server).",
    )


class JobResponse(BaseModel):
    session_id: str
    job_id: str
    model: str
    status: str


class SessionInfo(BaseModel):
    session_id: str
    job_id: str
    model: str
    callback_url: str
    status: str
    created: float


class CancelResponse(BaseModel):
    session_id: str
    job_id: str
    status: str


class HealthResponse(BaseModel):
    status: str
    pbs_available: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str
