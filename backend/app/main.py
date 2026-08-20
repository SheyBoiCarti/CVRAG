"""FastAPI application factory, lifespan state management, CORS, and error handlers."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import chat, health
from backend.app.api.schemas.common import ErrorDetail, ErrorResponse
from rag.errors import ProviderUnavailableError, RAGError
from rag.pipeline import RAGPipeline
from rag.provider_factories import FakeProviderFactories, ProviderFactories
from rag.settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    pipeline: RAGPipeline | None = None,
    provider_factories: ProviderFactories | None = None,
) -> FastAPI:
    """Creates and configures a FastAPI application instance."""
    settings_error: str | None = None
    if settings is None:
        try:
            resolved_settings = get_settings()
        except Exception:
            settings_error = "invalid_settings"
            resolved_settings = Settings()
    else:
        resolved_settings = settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Lifespan context manager maintaining startup/readiness state machine."""
        app.state.settings = resolved_settings
        app.state.generation_semaphore = asyncio.Semaphore(
            resolved_settings.API_MAX_CONCURRENT_GENERATIONS
        )

        if settings_error is not None:
            LOGGER.warning("Application booted with invalid configuration: %s", settings_error)
            app.state.pipeline = None
            app.state.readiness_state = "not_ready"
            app.state.readiness_reason_codes = [settings_error]
            app.state.pipeline_version = resolved_settings.PIPELINE_VERSION
            app.state.index_fingerprint = None
        elif pipeline is not None:
            app.state.pipeline = pipeline
            try:
                readiness = pipeline.readiness()
                app.state.readiness_state = readiness.state
                app.state.readiness_reason_codes = list(readiness.reason_codes)
                app.state.pipeline_version = readiness.pipeline_version
                app.state.index_fingerprint = readiness.index_fingerprint
            except Exception as exc:
                LOGGER.warning("Readiness check failed during startup (type=%s)", type(exc).__name__)
                app.state.readiness_state = "not_ready"
                app.state.readiness_reason_codes = ["pipeline_not_ready"]
                app.state.pipeline_version = resolved_settings.PIPELINE_VERSION
                app.state.index_fingerprint = None
        else:
            try:
                factories = provider_factories
                if factories is None and getattr(resolved_settings, "PROVIDER_MODE", "live") == "fake":
                    factories = FakeProviderFactories()

                if factories is None and not resolved_settings.NVIDIA_API_KEY:
                    LOGGER.info(
                        "NVIDIA_API_KEY is not configured; starting in degraded not_ready state."
                    )
                    app.state.pipeline = None
                    app.state.readiness_state = "not_ready"
                    app.state.readiness_reason_codes = ["pipeline_not_ready"]
                    app.state.pipeline_version = resolved_settings.PIPELINE_VERSION
                    app.state.index_fingerprint = None
                else:
                    instantiated_pipeline = RAGPipeline(
                        settings=resolved_settings,
                        provider_factories=factories,
                    )
                    app.state.pipeline = instantiated_pipeline
                    readiness = instantiated_pipeline.readiness()
                    app.state.readiness_state = readiness.state
                    app.state.readiness_reason_codes = list(readiness.reason_codes)
                    app.state.pipeline_version = readiness.pipeline_version
                    app.state.index_fingerprint = readiness.index_fingerprint
            except Exception as exc:
                LOGGER.warning("Pipeline initialization failed during startup (type=%s)", type(exc).__name__)
                app.state.pipeline = None
                app.state.readiness_state = "not_ready"
                app.state.readiness_reason_codes = ["pipeline_not_ready"]
                app.state.pipeline_version = resolved_settings.PIPELINE_VERSION
                app.state.index_fingerprint = None

        yield

    app = FastAPI(
        title="CV RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Attach initial components to state before lifespan starts
    app.state.settings = resolved_settings
    app.state.pipeline = pipeline
    app.state.provider_factories = provider_factories
    app.state.readiness_state = "starting"
    app.state.readiness_reason_codes = []
    app.state.pipeline_version = resolved_settings.PIPELINE_VERSION
    app.state.index_fingerprint = None

    # Correlation ID middleware
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        correlation_id = (
            request.headers.get("x-correlation-id")
            or request.headers.get("x-request-id")
            or uuid.uuid4().hex
        )
        request.state.correlation_id = correlation_id
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Request-ID"] = correlation_id
        LOGGER.info(
            "HTTP %s %s completed status=%s duration_ms=%s correlation_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            correlation_id,
        )
        return response

    # CORS configuration
    if resolved_settings.API_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.API_CORS_ORIGINS),
            allow_credentials=resolved_settings.API_CORS_ALLOW_CREDENTIALS,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def limit_body_size_middleware(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > resolved_settings.API_MAX_BODY_BYTES:
                    error_resp = ErrorResponse(
                        error=ErrorDetail(
                            code="request_too_large",
                            message=f"Request body size ({cl} bytes) exceeds limit of {resolved_settings.API_MAX_BODY_BYTES} bytes.",
                            retryable=False,
                        )
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content=error_resp.model_dump(),
                    )
            except ValueError:
                pass
            return await call_next(request)

        # Chunked or streaming request without Content-Length
        body_chunks: list[bytes] = []
        total_bytes = 0
        is_too_large = False
        async for chunk in request.stream():
            total_bytes += len(chunk)
            if total_bytes > resolved_settings.API_MAX_BODY_BYTES:
                is_too_large = True
                break
            body_chunks.append(chunk)

        if is_too_large:
            error_resp = ErrorResponse(
                error=ErrorDetail(
                    code="request_too_large",
                    message=f"Request body size exceeds limit of {resolved_settings.API_MAX_BODY_BYTES} bytes.",
                    retryable=False,
                )
            )
            return JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content=error_resp.model_dump(),
            )

        full_body = b"".join(body_chunks)

        async def receive():
            return {"type": "http.request", "body": full_body, "more_body": False}

        reconstructed_request = Request(request.scope, receive=receive)
        return await call_next(reconstructed_request)

    # Safe error exception handlers
    @app.exception_handler(ProviderUnavailableError)
    async def provider_unavailable_handler(
        request: Request, exc: ProviderUnavailableError
    ) -> JSONResponse:
        error_resp = ErrorResponse(
            error=ErrorDetail(
                code="provider_unavailable",
                message="The upstream AI provider is temporarily unavailable.",
                retryable=True,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(RAGError)
    async def rag_error_handler(
        request: Request, exc: RAGError
    ) -> JSONResponse:
        error_resp = ErrorResponse(
            error=ErrorDetail(
                code="rag_error",
                message="An internal RAG pipeline error occurred.",
                retryable=False,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
            detail_dict = exc.detail
            error_resp = ErrorResponse(
                error=ErrorDetail(
                    code=str(detail_dict["code"]),
                    message=str(detail_dict["message"]),
                    retryable=bool(detail_dict.get("retryable", False)),
                    details=detail_dict.get("details"),
                )
            )
        else:
            error_resp = ErrorResponse(
                error=ErrorDetail(
                    code=f"http_{exc.status_code}",
                    message=str(exc.detail),
                    retryable=False,
                )
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_resp.model_dump(),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        sanitized_errors = []
        is_history_error = False
        is_question_error = False
        for err in exc.errors():
            loc = [str(item) for item in err.get("loc", ())]
            if "history" in loc:
                is_history_error = True
            if "question" in loc:
                is_question_error = True
            msg = str(err.get("msg", "Invalid value"))
            err_type = str(err.get("type", "value_error"))
            sanitized_errors.append({"loc": loc, "msg": msg, "type": err_type})

        if is_history_error:
            err_code = "invalid_history"
        elif is_question_error:
            err_code = "invalid_question"
        else:
            err_code = "validation_error"

        error_resp = ErrorResponse(
            error=ErrorDetail(
                code=err_code,
                message="Invalid request payload.",
                retryable=False,
                details={"errors": sanitized_errors},
            )
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        LOGGER.warning("Unhandled server exception (type=%s)", type(exc).__name__)
        error_resp = ErrorResponse(
            error=ErrorDetail(
                code="internal_error",
                message="An unexpected error occurred.",
                retryable=False,
            )
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_resp.model_dump(),
        )

    # Route registration
    app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

    return app


app = create_app()

__all__ = ["app", "create_app"]
