"""FastAPI application factory, lifespan state management, CORS, and error handlers."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import chat, health
from backend.app.api.schemas.common import ErrorDetail, ErrorResponse
from rag.errors import ProviderUnavailableError, RAGError
from rag.pipeline import RAGPipeline
from rag.provider_factories import ProviderFactories
from rag.settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    pipeline: RAGPipeline | None = None,
    provider_factories: ProviderFactories | None = None,
) -> FastAPI:
    """Creates and configures a FastAPI application instance."""
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Lifespan context manager maintaining startup/readiness state machine."""
        app.state.settings = resolved_settings
        app.state.generation_semaphore = asyncio.Semaphore(
            resolved_settings.API_MAX_CONCURRENT_GENERATIONS
        )

        if pipeline is not None:
            app.state.pipeline = pipeline
            try:
                readiness = pipeline.readiness()
                app.state.readiness_state = readiness.state
                app.state.readiness_reason_codes = list(readiness.reason_codes)
                app.state.pipeline_version = readiness.pipeline_version
                app.state.index_fingerprint = readiness.index_fingerprint
            except Exception as exc:
                LOGGER.warning("Readiness check failed during startup: %s", exc)
                app.state.readiness_state = "not_ready"
                app.state.readiness_reason_codes = ["pipeline_not_ready"]
                app.state.pipeline_version = resolved_settings.PIPELINE_VERSION
                app.state.index_fingerprint = None
        else:
            try:
                if not resolved_settings.NVIDIA_API_KEY:
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
                        provider_factories=provider_factories,
                    )
                    app.state.pipeline = instantiated_pipeline
                    readiness = instantiated_pipeline.readiness()
                    app.state.readiness_state = readiness.state
                    app.state.readiness_reason_codes = list(readiness.reason_codes)
                    app.state.pipeline_version = readiness.pipeline_version
                    app.state.index_fingerprint = readiness.index_fingerprint
            except Exception as exc:
                LOGGER.warning("Pipeline initialization failed during startup: %s", exc)
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

    # CORS configuration
    if resolved_settings.API_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.API_CORS_ORIGINS),
            allow_credentials=resolved_settings.API_CORS_ALLOW_CREDENTIALS,
            allow_methods=["*"],
            allow_headers=["*"],
        )

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
        for err in exc.errors():
            loc = [str(item) for item in err.get("loc", ())]
            msg = str(err.get("msg", "Invalid value"))
            err_type = str(err.get("type", "value_error"))
            sanitized_errors.append({"loc": loc, "msg": msg, "type": err_type})

        error_resp = ErrorResponse(
            error=ErrorDetail(
                code="validation_error",
                message="Invalid request payload.",
                retryable=False,
                details={"errors": sanitized_errors},
            )
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_resp.model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        LOGGER.exception("Unhandled server exception: %s", exc)
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
