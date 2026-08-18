"""FastAPI request and process dependencies."""

from __future__ import annotations

import asyncio
from typing import cast

from fastapi import HTTPException, Request, status

from rag.pipeline import RAGPipeline
from rag.settings import Settings, get_settings as load_default_settings


def get_settings(request: Request) -> Settings:
    """Retrieves the Settings instance attached to the FastAPI application state."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return load_default_settings()
    return cast(Settings, settings)


def get_pipeline(request: Request) -> RAGPipeline:
    """Retrieves the active RAGPipeline instance from application state.

    Raises:
        HTTPException: If the pipeline has not been initialized or is unavailable.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "pipeline_not_ready",
                "message": "The RAG pipeline is not ready to process queries.",
                "retryable": True,
            },
        )
    return cast(RAGPipeline, pipeline)


def get_generation_semaphore(request: Request) -> asyncio.Semaphore:
    """Retrieves the asyncio concurrency semaphore limiting parallel generation requests."""
    semaphore = getattr(request.app.state, "generation_semaphore", None)
    if semaphore is None:
        settings = get_settings(request)
        semaphore = asyncio.Semaphore(settings.API_MAX_CONCURRENT_GENERATIONS)
        request.app.state.generation_semaphore = semaphore
    return cast(asyncio.Semaphore, semaphore)


__all__ = [
    "get_generation_semaphore",
    "get_pipeline",
    "get_settings",
]
