"""Health probe endpoints for process liveness and pipeline readiness."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from backend.app.api.schemas.common import HealthLiveResponse, ReadinessResponse
from rag.pipeline import RAGPipeline

router = APIRouter()


@router.get(
    "/live",
    response_model=HealthLiveResponse,
    summary="Liveness Probe",
    description="Returns 200 if the HTTP process is running. Never probes vector DB or AI providers.",
)
def health_live() -> HealthLiveResponse:
    """Always returns 200 OK while the server process is alive and handling HTTP requests."""
    return HealthLiveResponse(status="live")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "Service not ready"}},
    summary="Readiness Probe",
    description="Evaluates pipeline readiness and index availability. Returns 200 if ready, 503 otherwise.",
)
def health_ready(request: Request, response: Response) -> ReadinessResponse:
    """Evaluates whether the RAG pipeline is fully initialized and ready to serve queries."""
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)

    if pipeline is not None:
        try:
            readiness = pipeline.readiness()
            if not readiness.is_ready:
                response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadinessResponse(
                state=readiness.state,
                reason_codes=list(readiness.reason_codes),
                pipeline_version=readiness.pipeline_version,
                index_fingerprint=readiness.index_fingerprint,
            )
        except Exception:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadinessResponse(
                state="not_ready",
                reason_codes=["pipeline_not_ready"],
                pipeline_version=getattr(request.app.state, "pipeline_version", None),
                index_fingerprint=None,
            )

    state = getattr(request.app.state, "readiness_state", "not_ready")
    reason_codes = getattr(request.app.state, "readiness_reason_codes", ["pipeline_not_ready"])
    pipeline_version = getattr(request.app.state, "pipeline_version", None)
    index_fingerprint = getattr(request.app.state, "index_fingerprint", None)

    if state != "ready":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        state=state,
        reason_codes=list(reason_codes),
        pipeline_version=pipeline_version,
        index_fingerprint=index_fingerprint,
    )


__all__ = ["router"]
