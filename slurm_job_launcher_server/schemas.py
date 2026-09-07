"""Request/response models for the launcher API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubmitJobRequest(BaseModel):
    model: str = Field(..., description="Registered model name, e.g. 'molmoact2-droid'.")
    request_id: str | None = Field(
        default=None,
        description="Optional idempotency key. Re-submitting the same non-empty "
        "value returns the existing job instead of creating a duplicate.",
    )
    enable_action: bool = Field(
        default=False,
        description="Send actions to the robot (physical control). Off by default "
        "(dry-run). Only honored if the launcher config allows it.",
    )


class JobResponse(BaseModel):
    job_id: str
    model: str | None
    state: str
    request_id: str | None = None
    enable_action: bool | None = None


class JobSummary(BaseModel):
    job_id: str
    model: str | None
    state: str


class JobListResponse(BaseModel):
    jobs: list[JobSummary]


class HealthResponse(BaseModel):
    status: str
    slurm_available: bool


class CancelResponse(BaseModel):
    job_id: str
    status: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
