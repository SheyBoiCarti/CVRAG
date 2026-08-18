"""Synchronous and streaming chat endpoint routes."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from backend.app.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CitationSchema,
    ClaimVerificationSchema,
)
from backend.app.dependencies import (
    get_generation_semaphore,
    get_pipeline,
    get_settings,
)
from rag.models import ChatTurn
from rag.pipeline import RAGPipeline
from rag.settings import Settings

LOGGER = logging.getLogger(__name__)

router = APIRouter()


async def parse_and_validate_chat_payload(
    request: Request,
    settings: Settings,
) -> tuple[str, list[ChatTurn]]:
    """Parses and validates the incoming chat request body.

    Enforces size limits (API_MAX_BODY_BYTES), question constraints,
    and conversational history length/schema bounds.

    Returns:
        tuple[str, list[ChatTurn]]: Validated sanitized question and list of ChatTurn domain models.

    Raises:
        HTTPException(413): If body size exceeds API_MAX_BODY_BYTES.
        HTTPException(422): If question or history fails validation constraints.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            cl = int(content_length)
            if cl > settings.API_MAX_BODY_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "code": "request_too_large",
                        "message": f"Request body size ({cl} bytes) exceeds limit of {settings.API_MAX_BODY_BYTES} bytes.",
                        "retryable": False,
                    },
                )
        except ValueError:
            pass

    body = await request.body()
    if len(body) > settings.API_MAX_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "request_too_large",
                "message": f"Request body size ({len(body)} bytes) exceeds limit of {settings.API_MAX_BODY_BYTES} bytes.",
                "retryable": False,
            },
        )

    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "validation_error",
                "message": "Malformed JSON payload.",
                "retryable": False,
            },
        )

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "validation_error",
                "message": "Request payload must be a JSON object.",
                "retryable": False,
            },
        )

    if "question" not in data or not isinstance(data["question"], str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "invalid_question",
                "message": "Field 'question' is required and must be a string.",
                "retryable": False,
            },
        )

    raw_question = data["question"]
    if not raw_question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "invalid_question",
                "message": "Question cannot be empty or whitespace-only.",
                "retryable": False,
            },
        )

    if len(raw_question) > settings.MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "invalid_question",
                "message": f"Question exceeds maximum length of {settings.MAX_QUESTION_CHARS} characters.",
                "retryable": False,
            },
        )

    raw_history = data.get("history", [])
    if not isinstance(raw_history, list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "invalid_history",
                "message": "Field 'history' must be a list of chat turn objects.",
                "retryable": False,
            },
        )

    if len(raw_history) > 20:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "invalid_history",
                "message": f"History length cannot exceed 20 turns (received {len(raw_history)}).",
                "retryable": False,
            },
        )

    history_turns: list[ChatTurn] = []
    for idx, turn in enumerate(raw_history):
        if not isinstance(turn, dict) or "user" not in turn or "assistant" not in turn:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "invalid_history",
                    "message": f"History turn at index {idx} must be an object containing 'user' and 'assistant'.",
                    "retryable": False,
                },
            )
        if not isinstance(turn["user"], str) or not isinstance(turn["assistant"], str):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "invalid_history",
                    "message": f"History turn at index {idx} must contain string values for 'user' and 'assistant'.",
                    "retryable": False,
                },
            )
        history_turns.append(ChatTurn(user=turn["user"], assistant=turn["assistant"]))

    return raw_question, history_turns


def check_pipeline_readiness(pipeline: RAGPipeline) -> None:
    """Verifies that the RAG pipeline is healthy and ready to process requests."""
    try:
        readiness = pipeline.readiness()
    except Exception as exc:
        LOGGER.warning("Readiness check error during request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "pipeline_not_ready",
                "message": "The RAG pipeline readiness check failed.",
                "retryable": True,
            },
        ) from exc

    if readiness.state != "ready":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "pipeline_not_ready",
                "message": "The RAG pipeline is not ready to process queries.",
                "retryable": True,
                "details": {"reason_codes": list(readiness.reason_codes)},
            },
        )


@router.post("", response_model=ChatResponse)
async def chat_sync(
    request: Request,
    settings: Settings = Depends(get_settings),
    pipeline: RAGPipeline = Depends(get_pipeline),
    semaphore: asyncio.Semaphore = Depends(get_generation_semaphore),
) -> ChatResponse:
    """Synchronous chat query answering endpoint.

    Validates request payload size and bounds, ensures pipeline readiness,
    and executes answer generation within concurrency limits in a thread pool.
    """
    question, history = await parse_and_validate_chat_payload(request, settings)
    check_pipeline_readiness(pipeline)

    if semaphore.locked():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "capacity_exceeded",
                "message": "Maximum concurrent generation limit reached. Please retry shortly.",
                "retryable": True,
            },
        )

    await semaphore.acquire()
    try:
        answer_result = await run_in_threadpool(
            pipeline.answer,
            question=question,
            history=history,
        )
    finally:
        semaphore.release()

    citations = [
        CitationSchema(
            evidence_id=c.evidence_id,
            chunk_id=c.chunk_id,
            source=Path(c.source).name,
            page=c.page,
            excerpt=c.excerpt,
            support_status=c.support_status,
        )
        for c in answer_result.citations
    ]

    claim_verifications = [
        ClaimVerificationSchema(
            claim_id=cv.claim_id,
            text=cv.text,
            factual=cv.factual,
            importance=cv.importance,
            citation_ids=list(cv.citation_ids),
            status=cv.status,
            reason_code=cv.reason_code,
        )
        for cv in answer_result.claim_verifications
    ]

    return ChatResponse(
        answer=answer_result.answer,
        citations=citations,
        claim_verifications=claim_verifications,
        decision=answer_result.decision,
        reason_codes=list(answer_result.reason_codes),
        retrieval_query=answer_result.retrieval_query,
        pipeline_version=answer_result.pipeline_version,
    )


__all__ = [
    "chat_sync",
    "check_pipeline_readiness",
    "parse_and_validate_chat_payload",
    "router",
]
