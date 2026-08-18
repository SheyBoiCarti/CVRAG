"""Common response and error schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag.models import ReadinessState


class ErrorDetail(BaseModel):
    """Detailed error object embedded within an ErrorResponse envelope."""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Top-level error response envelope matching Schema Version 1 contract."""

    error: ErrorDetail


class HealthLiveResponse(BaseModel):
    """Liveness probe response indicating HTTP process availability."""

    status: str = "live"


class ReadinessResponse(BaseModel):
    """Readiness probe response indicating pipeline and vector index health."""

    state: ReadinessState
    reason_codes: list[str] = Field(default_factory=list)
    pipeline_version: str | None = None
    index_fingerprint: str | None = None


__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "HealthLiveResponse",
    "ReadinessResponse",
]
