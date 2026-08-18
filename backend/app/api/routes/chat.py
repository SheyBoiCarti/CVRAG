"""Synchronous and streaming chat endpoint routes."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import threading
from typing import Any, Sequence
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
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
from rag.models import ChatTurn, PipelineEvent
from rag.pipeline import RAGPipeline
from rag.settings import Settings

LOGGER = logging.getLogger(__name__)

router = APIRouter()


def _serialize_for_json(obj: Any) -> Any:
    """Recursively serializes domain models, dataclasses, dicts, and sequences for SSE JSON payload."""
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _serialize_for_json(obj.to_dict())
    if isinstance(obj, dict):
        d = {k: _serialize_for_json(v) for k, v in obj.items()}
        if "source" in d and isinstance(d["source"], str) and "evidence_id" in d:
            d["source"] = Path(d["source"]).name
        return d
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    return obj


def format_sse_event(event: PipelineEvent) -> str:
    """Formats a PipelineEvent envelope into standard SSE wire format: `data: {...}\\n\\n`."""
    payload = {
        "schema_version": event.schema_version,
        "request_id": event.request_id,
        "sequence": event.sequence,
        "event": event.event,
        "data": _serialize_for_json(event.data),
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


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


@router.post("/stream")
async def chat_stream(
    request: Request,
    settings: Settings = Depends(get_settings),
    pipeline: RAGPipeline = Depends(get_pipeline),
    semaphore: asyncio.Semaphore = Depends(get_generation_semaphore),
) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming chat endpoint.

    Validates request payload size and conversational bounds, ensures pipeline readiness,
    and executes answer generation within concurrency limits via a background worker thread
    and asynchronous queue bridge. Handles early client disconnection cleanly.
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

    req_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    queue: asyncio.Queue[PipelineEvent | Exception | object] = asyncio.Queue(maxsize=1000)
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()
    _SENTINEL = object()

    def run_worker() -> None:
        try:
            stream_iter = pipeline.stream(
                question=question,
                history=history,
                request_id=req_id,
            )
            for event in stream_iter:
                if stop_event.is_set():
                    LOGGER.debug("Worker received stop signal, aborting stream (request_id=%s)", req_id)
                    break
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            LOGGER.warning(
                "Unhandled exception in streaming worker thread (request_id=%s, type=%s): %s",
                req_id,
                type(exc).__name__,
                exc,
            )
            if not stop_event.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    worker_thread = threading.Thread(
        target=run_worker,
        name=f"rag-stream-worker-{req_id}",
        daemon=True,
    )
    worker_thread.start()

    async def stream_generator():
        seq = 1
        accumulated_answer = ""
        terminal_emitted = False
        try:
            while True:
                if await request.is_disconnected():
                    LOGGER.info("Client disconnected during stream (request_id=%s)", req_id)
                    stop_event.set()
                    break

                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        LOGGER.info("Client disconnected during wait timeout (request_id=%s)", req_id)
                        stop_event.set()
                        break
                    continue

                if item is _SENTINEL:
                    break

                if isinstance(item, Exception):
                    if not terminal_emitted and not stop_event.is_set():
                        error_event = PipelineEvent.create(
                            "error",
                            {
                                "code": "stream_interrupted",
                                "message": "The stream was interrupted.",
                                "retryable": False,
                                "partial_answer": accumulated_answer or None,
                            },
                            request_id=req_id,
                            sequence=seq,
                        )
                        seq += 1
                        terminal_emitted = True
                        yield format_sse_event(error_event)
                    break

                if isinstance(item, PipelineEvent):
                    if stop_event.is_set():
                        break

                    if item.event == "answer_delta":
                        delta = item.data.get("delta", "")
                        if delta:
                            accumulated_answer += delta
                    elif item.event in ("answer_completed", "error"):
                        terminal_emitted = True

                    yield format_sse_event(item)
                    seq = max(seq, item.sequence + 1)

                    if item.event in ("answer_completed", "error"):
                        break

        except asyncio.CancelledError:
            LOGGER.info("Streaming response task cancelled (request_id=%s)", req_id)
            stop_event.set()
            raise
        except Exception as exc:
            LOGGER.warning("Unhandled error during stream consumption (request_id=%s): %s", req_id, exc)
            if not terminal_emitted and not stop_event.is_set():
                error_event = PipelineEvent.create(
                    "error",
                    {
                        "code": "stream_interrupted",
                        "message": "The stream was interrupted.",
                        "retryable": False,
                        "partial_answer": accumulated_answer or None,
                    },
                    request_id=req_id,
                    sequence=seq,
                )
                terminal_emitted = True
                yield format_sse_event(error_event)
        finally:
            stop_event.set()
            while not queue.empty():
                try:
                    queue.get_nowait()
                except Exception:
                    break
            semaphore.release()

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
        },
    )


__all__ = [
    "chat_stream",
    "chat_sync",
    "check_pipeline_readiness",
    "format_sse_event",
    "parse_and_validate_chat_payload",
    "router",
]
