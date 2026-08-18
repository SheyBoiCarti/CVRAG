from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import APIRouter, Depends, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel
import pytest

from backend.app.api.schemas.chat import ChatRequest, ChatResponse
from backend.app.api.schemas.common import (
    ErrorDetail,
    ErrorResponse,
    HealthLiveResponse,
    ReadinessResponse,
)
from backend.app.dependencies import (
    get_generation_semaphore,
    get_pipeline,
    get_settings,
)
from backend.app.main import create_app
from rag.errors import ProviderUnavailableError, RAGError
from rag.models import ReadinessResult
from rag.pipeline import RAGPipeline
from rag.settings import Settings
from tests.fakes import FakeProviderFactories


def test_health_live_endpoint_returns_200_always(tmp_path: Path) -> None:
    """GET /api/v1/health/live returns 200 {"status": "live"} even without API key."""
    settings = Settings(
        NVIDIA_API_KEY=None,
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        payload = response.json()
        assert payload == {"status": "live"}
        HealthLiveResponse.model_validate(payload)


def test_health_ready_returns_200_when_pipeline_is_ready(tmp_path: Path) -> None:
    """GET /api/v1/health/ready returns 200 and ReadinessResponse when pipeline is ready."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(
        state="ready",
        reason_codes=(),
        pipeline_version="dense-v1",
        index_fingerprint=None,
    )

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200
        payload = response.json()
        assert payload["state"] == "ready"
        assert payload["reason_codes"] == []
        assert payload["pipeline_version"] == "dense-v1"
        ReadinessResponse.model_validate(payload)


def test_health_ready_returns_503_when_pipeline_is_not_ready(tmp_path: Path) -> None:
    """GET /api/v1/health/ready returns 503 when pipeline reports not_ready."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(
        state="not_ready",
        reason_codes=("index_empty",),
        pipeline_version="dense-v1",
        index_fingerprint=None,
    )

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 503
        payload = response.json()
        assert payload["state"] == "not_ready"
        assert "index_empty" in payload["reason_codes"]
        ReadinessResponse.model_validate(payload)


def test_lifespan_tolerates_missing_api_key_without_crashing(tmp_path: Path) -> None:
    """Lifespan does not crash when NVIDIA_API_KEY is missing; marks state not_ready."""
    settings = Settings(
        NVIDIA_API_KEY=None,
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        # Liveness is healthy
        live_res = client.get("/api/v1/health/live")
        assert live_res.status_code == 200
        assert live_res.json()["status"] == "live"

        # Readiness reports 503 with safe reason codes
        ready_res = client.get("/api/v1/health/ready")
        assert ready_res.status_code == 503
        ready_payload = ready_res.json()
        assert ready_payload["state"] == "not_ready"
        assert "pipeline_not_ready" in ready_payload["reason_codes"]
        # Ensure no secret or internal detail is leaked
        assert "API_KEY" not in str(ready_payload)
        assert "Traceback" not in str(ready_payload)
        ReadinessResponse.model_validate(ready_payload)


def test_cors_middleware_headers_applied(tmp_path: Path) -> None:
    """CORS headers are correctly returned for configured allowed origins."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_CORS_ORIGINS=("http://localhost:3000", "http://localhost:8501"),
        API_CORS_ALLOW_CREDENTIALS=True,
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
        assert response.headers.get("access-control-allow-credentials") == "true"


def test_exception_handler_returns_safe_error_response(tmp_path: Path) -> None:
    """Unhandled errors return safe ErrorResponse without leaking stack traces or host paths."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    app = create_app(settings=settings)

    test_router = APIRouter()

    @test_router.get("/api/v1/test-error")
    def trigger_error() -> None:
        raise RuntimeError("Secret host path /var/data/users/secret_key.pem failed!")

    app.include_router(test_router)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/test-error")
        assert response.status_code == 500
        payload = response.json()
        assert "error" in payload
        assert payload["error"]["code"] == "internal_error"
        assert payload["error"]["message"] == "An unexpected error occurred."
        assert "secret_key" not in str(payload)
        assert "Traceback" not in str(payload)
        ErrorResponse.model_validate(payload)


def test_provider_unavailable_exception_handler(tmp_path: Path) -> None:
    """ProviderUnavailableError maps to HTTP 503 with retryable=True."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    app = create_app(settings=settings)
    test_router = APIRouter()

    @test_router.get("/api/v1/test-provider-error")
    def trigger_provider_error() -> None:
        raise ProviderUnavailableError("Simulated upstream outage")

    app.include_router(test_router)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/test-provider-error")
        assert response.status_code == 503
        payload = response.json()
        assert payload["error"]["code"] == "provider_unavailable"
        assert payload["error"]["retryable"] is True
        ErrorResponse.model_validate(payload)


def test_rag_error_exception_handler(tmp_path: Path) -> None:
    """Generic RAGError maps to HTTP 500 with retryable=False."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    app = create_app(settings=settings)
    test_router = APIRouter()

    @test_router.get("/api/v1/test-rag-error")
    def trigger_rag_error() -> None:
        raise RAGError("Internal parsing failure")

    app.include_router(test_router)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/test-rag-error")
        assert response.status_code == 500
        payload = response.json()
        assert payload["error"]["code"] == "rag_error"
        assert payload["error"]["retryable"] is False
        ErrorResponse.model_validate(payload)


def test_validation_error_handler(tmp_path: Path) -> None:
    """Request validation errors map to HTTP 422 with structured details."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    app = create_app(settings=settings)
    test_router = APIRouter()

    class SampleBody(BaseModel):
        required_field: str

    @test_router.post("/api/v1/test-validation")
    def trigger_validation(body: SampleBody) -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(test_router)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/test-validation", json={})
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "validation_error"
        assert "details" in payload["error"]
        assert "errors" in payload["error"]["details"]
        ErrorResponse.model_validate(payload)


def test_dependencies_injection(tmp_path: Path) -> None:
    """Dependencies correctly provide Settings, Pipeline, and Semaphore."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_MAX_CONCURRENT_GENERATIONS=3,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    app = create_app(settings=settings, pipeline=mock_pipeline)

    test_router = APIRouter()

    @test_router.get("/api/v1/test-deps")
    def test_deps(
        s: Settings = Depends(get_settings),
        p: RAGPipeline = Depends(get_pipeline),
        sem: asyncio.Semaphore = Depends(get_generation_semaphore),
    ) -> dict[str, str]:
        assert s is settings
        assert p is mock_pipeline
        assert sem._value == 3
        return {"status": "ok"}

    app.include_router(test_router)

    with TestClient(app) as client:
        res = client.get("/api/v1/test-deps")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}


def test_get_pipeline_raises_503_when_uninitialized(tmp_path: Path) -> None:
    """get_pipeline dependency raises 503 when app.state.pipeline is None."""
    settings = Settings(
        NVIDIA_API_KEY=None,
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    app = create_app(settings=settings)

    test_router = APIRouter()

    @test_router.get("/api/v1/test-no-pipeline")
    def test_no_pipeline(p: RAGPipeline = Depends(get_pipeline)) -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(test_router)

    with TestClient(app) as client:
        res = client.get("/api/v1/test-no-pipeline")
        assert res.status_code == 503
        payload = res.json()
        assert payload["error"]["code"] == "pipeline_not_ready"
