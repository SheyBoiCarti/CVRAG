from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock

from fastapi import Request
from fastapi.testclient import TestClient
import pytest

from backend.app.api.schemas.chat import ChatResponse
from backend.app.api.schemas.common import ErrorResponse
from backend.app.main import create_app
from rag.errors import ProviderUnavailableError
from rag.models import AnswerResult, Citation, PipelineEvent, ReadinessResult
from rag.pipeline import RAGPipeline
from rag.settings import Settings
from tests.fakes import FakeProviderFactories


def test_chat_sync_success_response(tmp_path: Path) -> None:
    """POST /api/v1/chat returns 200 with structured ChatResponse."""
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
    mock_pipeline.answer.return_value = AnswerResult(
        answer="John has 5 years of Python experience [S1].",
        citations=(
            Citation(
                evidence_id="S1",
                chunk_id="chunk_1",
                source="john_doe_cv.pdf",
                page=1,
                excerpt="5 years experience in Python",
                support_status="not_checked",
            ),
        ),
        claim_verifications=(),
        decision="answered",
        reason_codes=(),
        retrieval_query="John Python experience",
        pipeline_version="dense-v1",
    )

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "question": "What is John's Python experience?",
                "history": [{"user": "Hi", "assistant": "Hello"}],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        validated = ChatResponse.model_validate(payload)
        assert validated.answer == "John has 5 years of Python experience [S1]."
        assert len(validated.citations) == 1
        assert validated.citations[0].source == "john_doe_cv.pdf"
        assert validated.citations[0].page == 1
        assert validated.decision == "answered"
        assert validated.retrieval_query == "John Python experience"
        assert validated.pipeline_version == "dense-v1"


def test_chat_sync_citations_contain_basenames_only(tmp_path: Path) -> None:
    """Citations returned by /api/v1/chat sanitize source paths to basenames only."""
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
    mock_pipeline.answer.return_value = AnswerResult(
        answer="Cited answer [S1].",
        citations=(
            Citation(
                evidence_id="S1",
                chunk_id="c1",
                source="/absolute/host/server/path/documents/secret_resume.pdf",
                page=1,
                excerpt="Senior Developer",
                support_status="not_checked",
            ),
        ),
        claim_verifications=(),
        decision="answered",
        reason_codes=(),
        retrieval_query="resume",
        pipeline_version="dense-v1",
    )

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "Tell me about the resume."},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["citations"][0]["source"] == "secret_resume.pdf"
        assert "/absolute/host" not in str(payload)


def test_chat_sync_oversized_payload_returns_413(tmp_path: Path) -> None:
    """POST /api/v1/chat returns 413 request_too_large when payload > API_MAX_BODY_BYTES."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_MAX_BODY_BYTES=1024,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        large_padding = "x" * 2000
        response = client.post(
            "/api/v1/chat",
            json={"question": f"Question with padding {large_padding}"},
        )
        assert response.status_code == 413
        payload = response.json()
        assert payload["error"]["code"] == "request_too_large"
        assert payload["error"]["retryable"] is False
        ErrorResponse.model_validate(payload)


def test_chat_sync_content_length_oversized_returns_413(tmp_path: Path) -> None:
    """POST /api/v1/chat checks Content-Length header and returns 413 before body processing."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_MAX_BODY_BYTES=500,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            content=b'{"question": "test"}',
            headers={"Content-Length": "1000", "Content-Type": "application/json"},
        )
        assert response.status_code == 413
        payload = response.json()
        assert payload["error"]["code"] == "request_too_large"


def test_chat_sync_blank_question_returns_422(tmp_path: Path) -> None:
    """Blank or whitespace question returns 422 invalid_question."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"question": "   "})
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "invalid_question"
        assert payload["error"]["retryable"] is False
        ErrorResponse.model_validate(payload)


def test_chat_sync_empty_question_returns_422(tmp_path: Path) -> None:
    """Empty question returns 422 invalid_question."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"question": ""})
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "invalid_question"


def test_chat_sync_oversized_question_returns_422(tmp_path: Path) -> None:
    """Question longer than MAX_QUESTION_CHARS returns 422 invalid_question."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        MAX_QUESTION_CHARS=50,
        API_MAX_BODY_BYTES=32768,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        long_q = "A" * 51
        response = client.post("/api/v1/chat", json={"question": long_q})
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "invalid_question"


def test_chat_sync_malformed_history_type_returns_422(tmp_path: Path) -> None:
    """History not being a list returns 422 invalid_history."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "Valid question?", "history": "not-a-list"},
        )
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "invalid_history"


def test_chat_sync_malformed_history_item_returns_422(tmp_path: Path) -> None:
    """History item missing required user/assistant keys returns 422 invalid_history."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={
                "question": "Valid question?",
                "history": [{"user": "Hello"}],  # missing assistant
            },
        )
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "invalid_history"


def test_chat_sync_history_exceeds_20_turns_returns_422(tmp_path: Path) -> None:
    """History exceeding 20 turns returns 422 invalid_history."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        history = [{"user": f"Q{i}", "assistant": f"A{i}"} for i in range(21)]
        response = client.post(
            "/api/v1/chat",
            json={"question": "Valid question?", "history": history},
        )
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "invalid_history"


def test_chat_sync_pipeline_not_ready_returns_503(tmp_path: Path) -> None:
    """POST /api/v1/chat returns 503 pipeline_not_ready when pipeline reports not_ready."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(
        state="not_ready",
        reason_codes=("index_empty",),
    )

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "What skills are listed?"},
        )
        assert response.status_code == 503
        payload = response.json()
        assert payload["error"]["code"] == "pipeline_not_ready"
        assert payload["error"]["retryable"] is True


def test_chat_sync_pipeline_uninitialized_returns_503(tmp_path: Path) -> None:
    """POST /api/v1/chat returns 503 pipeline_not_ready when app has no pipeline."""
    settings = Settings(
        NVIDIA_API_KEY=None,
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    app = create_app(settings=settings)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "What skills are listed?"},
        )
        assert response.status_code == 503
        payload = response.json()
        assert payload["error"]["code"] == "pipeline_not_ready"


def test_chat_sync_provider_unavailable_returns_503(tmp_path: Path) -> None:
    """ProviderUnavailableError from pipeline.answer() returns 503 provider_unavailable."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")
    mock_pipeline.answer.side_effect = ProviderUnavailableError("NVIDIA API timeout")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "What is John's experience?"},
        )
        assert response.status_code == 503
        payload = response.json()
        assert payload["error"]["code"] == "provider_unavailable"
        assert payload["error"]["retryable"] is True


def test_chat_sync_concurrency_limit_exceeded_returns_503(tmp_path: Path) -> None:
    """When generation semaphore is exhausted, /api/v1/chat returns 503 capacity_exceeded."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_MAX_CONCURRENT_GENERATIONS=1,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")
    mock_pipeline.answer.return_value = AnswerResult(
        answer="Ok answer.",
        decision="answered",
    )

    app = create_app(settings=settings, pipeline=mock_pipeline)

    with TestClient(app) as client:
        # Pre-acquire the generation semaphore to simulate saturated concurrency
        sem: asyncio.Semaphore = app.state.generation_semaphore
        loop = asyncio.new_event_loop()
        loop.run_until_complete(sem.acquire())

        try:
            response = client.post(
                "/api/v1/chat",
                json={"question": "What is John's experience?"},
            )
            assert response.status_code == 503
            payload = response.json()
            assert payload["error"]["code"] == "capacity_exceeded"
            assert payload["error"]["retryable"] is True
        finally:
            sem.release()
            loop.close()

        # Once released, request succeeds
        res_after = client.post(
            "/api/v1/chat",
            json={"question": "What is John's experience?"},
        )
        assert res_after.status_code == 200


def test_chat_sync_generic_error_returns_safe_500(tmp_path: Path) -> None:
    """Unhandled internal pipeline exceptions return safe 500 internal_error."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")
    mock_pipeline.answer.side_effect = RuntimeError(
        "Host disk failed at /etc/secret_keys/path!"
    )

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "What is John's experience?"},
        )
        assert response.status_code == 500
        payload = response.json()
        assert payload["error"]["code"] == "internal_error"
        assert payload["error"]["message"] == "An unexpected error occurred."
        assert "/etc/secret_keys" not in str(payload)


def test_chat_sync_end_to_end_with_fake_providers(tmp_path: Path) -> None:
    """End-to-end sync chat using RAGPipeline with FakeProviderFactories."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "sample_cv.txt").write_text(
        "John Doe is a Senior Python Developer with Kubernetes and FastAPI expertise.",
        encoding="utf-8",
    )

    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=docs_dir,
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    factories = FakeProviderFactories(
        default_response="John Doe specializes in Python and FastAPI [S1]."
    )

    pipeline = RAGPipeline(settings=settings, provider_factories=factories)
    app = create_app(settings=settings, pipeline=pipeline)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            json={"question": "What does John specialize in?"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "John Doe specializes in Python and FastAPI" in payload["answer"]
        assert payload["decision"] == "answered"
        assert len(payload["citations"]) >= 1
        assert payload["citations"][0]["source"] == "sample_cv.txt"
    app.state.pipeline = None
    del pipeline
    import gc
    gc.collect()


def _parse_sse_events(raw_text: str) -> list[dict]:
    events = []
    for line in raw_text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        if line_str.startswith("data:"):
            data_str = line_str[len("data:"):].strip()
            events.append(json.loads(data_str))
    return events


def test_chat_stream_success_events_and_headers(tmp_path: Path) -> None:
    """POST /api/v1/chat/stream returns 200 SSE stream with correct headers and event sequence."""
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
    )

    def mock_stream(question: str, history: any = (), request_id: str | None = None):
        req_id = request_id or "req-test-1"
        yield PipelineEvent.create("retrieval_started", {}, request_id=req_id, sequence=1)
        yield PipelineEvent.create(
            "retrieval_completed",
            {"evidence_count": 1, "degraded_reason_codes": []},
            request_id=req_id,
            sequence=2,
        )
        yield PipelineEvent.create("generation_started", {}, request_id=req_id, sequence=3)
        yield PipelineEvent.create(
            "answer_delta",
            {"delta": "John Doe has 5 years "},
            request_id=req_id,
            sequence=4,
        )
        yield PipelineEvent.create(
            "answer_delta",
            {"delta": "of Python experience [S1]."},
            request_id=req_id,
            sequence=5,
        )
        result = AnswerResult(
            answer="John Doe has 5 years of Python experience [S1].",
            citations=(
                Citation(
                    evidence_id="S1",
                    chunk_id="c1",
                    source="/path/to/john_doe_cv.pdf",
                    page=1,
                    excerpt="5 years of Python",
                    support_status="not_checked",
                ),
            ),
            claim_verifications=(),
            decision="answered",
            reason_codes=(),
            retrieval_query="Python experience",
            pipeline_version="dense-v1",
        )
        yield PipelineEvent.create(
            "answer_completed",
            {"result": result, "replace_streamed_answer": False},
            request_id=req_id,
            sequence=6,
        )

    mock_pipeline.stream.side_effect = mock_stream
    app = create_app(settings=settings, pipeline=mock_pipeline)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={"question": "What is John's experience?"},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            assert response.headers.get("cache-control") == "no-cache"
            assert response.headers.get("x-accel-buffering") == "no"

            raw_body = response.read().decode("utf-8")
            events = _parse_sse_events(raw_body)

            assert len(events) == 6
            event_names = [e["event"] for e in events]
            assert event_names == [
                "retrieval_started",
                "retrieval_completed",
                "generation_started",
                "answer_delta",
                "answer_delta",
                "answer_completed",
            ]

            for i, ev in enumerate(events, 1):
                assert ev["sequence"] == i
                assert ev["schema_version"] == "1"
                assert ev["request_id"] != ""

            # Check deltas
            deltas = [e["data"]["delta"] for e in events if e["event"] == "answer_delta"]
            assert "".join(deltas) == "John Doe has 5 years of Python experience [S1]."

            # Check terminal event
            term = events[-1]
            assert term["event"] == "answer_completed"
            assert term["data"]["replace_streamed_answer"] is False
            res = term["data"]["result"]
            assert res["answer"] == "John Doe has 5 years of Python experience [S1]."
            assert res["decision"] == "answered"
            assert len(res["citations"]) == 1
            assert res["citations"][0]["source"] == "john_doe_cv.pdf"


def test_chat_stream_empty_retrieval_abstention(tmp_path: Path) -> None:
    """Empty retrieval emits retrieval_started, retrieval_completed(0), and terminal answer_completed."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    def mock_stream(question: str, history: any = (), request_id: str | None = None):
        req_id = request_id or "req-empty-1"
        yield PipelineEvent.create("retrieval_started", {}, request_id=req_id, sequence=1)
        yield PipelineEvent.create(
            "retrieval_completed",
            {"evidence_count": 0, "degraded_reason_codes": []},
            request_id=req_id,
            sequence=2,
        )
        result = AnswerResult(
            answer="I do not have enough information.",
            citations=(),
            claim_verifications=(),
            decision="abstained",
            reason_codes=("no_evidence",),
            retrieval_query="something obscure",
            pipeline_version="dense-v1",
        )
        yield PipelineEvent.create(
            "answer_completed",
            {"result": result, "replace_streamed_answer": False},
            request_id=req_id,
            sequence=3,
        )

    mock_pipeline.stream.side_effect = mock_stream
    app = create_app(settings=settings, pipeline=mock_pipeline)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={"question": "What is obscure?"},
        ) as response:
            assert response.status_code == 200
            events = _parse_sse_events(response.read().decode("utf-8"))
            assert len(events) == 3
            assert [e["event"] for e in events] == [
                "retrieval_started",
                "retrieval_completed",
                "answer_completed",
            ]
            assert events[-1]["data"]["result"]["decision"] == "abstained"


def test_chat_stream_preflight_oversized_payload_returns_413(tmp_path: Path) -> None:
    """POST /api/v1/chat/stream returns 413 request_too_large before stream headers."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_MAX_BODY_BYTES=1024,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        large_padding = "x" * 2000
        response = client.post(
            "/api/v1/chat/stream",
            json={"question": f"Question with padding {large_padding}"},
        )
        assert response.status_code == 413
        payload = response.json()
        assert payload["error"]["code"] == "request_too_large"
        assert payload["error"]["retryable"] is False


def test_chat_stream_preflight_content_length_oversized_returns_413(tmp_path: Path) -> None:
    """POST /api/v1/chat/stream returns 413 based on Content-Length header before stream headers."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_MAX_BODY_BYTES=500,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/stream",
            content=b'{"question": "test"}',
            headers={"Content-Length": "1000", "Content-Type": "application/json"},
        )
        assert response.status_code == 413
        payload = response.json()
        assert payload["error"]["code"] == "request_too_large"


def test_chat_stream_preflight_blank_question_returns_422(tmp_path: Path) -> None:
    """POST /api/v1/chat/stream returns 422 invalid_question for blank question."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post("/api/v1/chat/stream", json={"question": "   "})
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "invalid_question"


def test_chat_stream_preflight_oversized_question_returns_422(tmp_path: Path) -> None:
    """POST /api/v1/chat/stream returns 422 invalid_question when question > MAX_QUESTION_CHARS."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        MAX_QUESTION_CHARS=50,
        API_MAX_BODY_BYTES=32768,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        long_q = "B" * 51
        response = client.post("/api/v1/chat/stream", json={"question": long_q})
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "invalid_question"


def test_chat_stream_preflight_invalid_history_returns_422(tmp_path: Path) -> None:
    """POST /api/v1/chat/stream returns 422 invalid_history for malformed history."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/stream",
            json={"question": "Valid?", "history": [{"missing_assistant": "foo"}]},
        )
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "invalid_history"


def test_chat_stream_preflight_history_exceeds_20_turns_returns_422(tmp_path: Path) -> None:
    """POST /api/v1/chat/stream returns 422 invalid_history when history > 20 turns."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        history = [{"user": f"Q{i}", "assistant": f"A{i}"} for i in range(21)]
        response = client.post(
            "/api/v1/chat/stream",
            json={"question": "Valid?", "history": history},
        )
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "invalid_history"


def test_chat_stream_preflight_pipeline_not_ready_returns_503(tmp_path: Path) -> None:
    """POST /api/v1/chat/stream returns 503 pipeline_not_ready before stream headers."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(
        state="not_ready",
        reason_codes=("index_empty",),
    )

    app = create_app(settings=settings, pipeline=mock_pipeline)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/stream",
            json={"question": "What is Python?"},
        )
        assert response.status_code == 503
        payload = response.json()
        assert payload["error"]["code"] == "pipeline_not_ready"


def test_chat_stream_preflight_concurrency_limit_exceeded_returns_503(tmp_path: Path) -> None:
    """When generation semaphore is saturated, /api/v1/chat/stream returns 503 capacity_exceeded."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_MAX_CONCURRENT_GENERATIONS=1,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    app = create_app(settings=settings, pipeline=mock_pipeline)

    with TestClient(app) as client:
        sem: asyncio.Semaphore = app.state.generation_semaphore
        loop = asyncio.new_event_loop()
        loop.run_until_complete(sem.acquire())

        try:
            response = client.post(
                "/api/v1/chat/stream",
                json={"question": "What is John's experience?"},
            )
            assert response.status_code == 503
            payload = response.json()
            assert payload["error"]["code"] == "capacity_exceeded"
            assert payload["error"]["retryable"] is True
        finally:
            sem.release()
            loop.close()


def test_chat_stream_midstream_exception_emits_error_event(tmp_path: Path) -> None:
    """Mid-stream exception emits terminal error event with code stream_interrupted and closes."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    def mock_stream_failing(question: str, history: any = (), request_id: str | None = None):
        req_id = request_id or "req-fail-1"
        yield PipelineEvent.create("retrieval_started", {}, request_id=req_id, sequence=1)
        yield PipelineEvent.create(
            "retrieval_completed",
            {"evidence_count": 1, "degraded_reason_codes": []},
            request_id=req_id,
            sequence=2,
        )
        yield PipelineEvent.create("generation_started", {}, request_id=req_id, sequence=3)
        yield PipelineEvent.create(
            "answer_delta",
            {"delta": "Partial text"},
            request_id=req_id,
            sequence=4,
        )
        raise RuntimeError("Provider connection died during generation")

    mock_pipeline.stream.side_effect = mock_stream_failing
    app = create_app(settings=settings, pipeline=mock_pipeline)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={"question": "What is John's experience?"},
        ) as response:
            assert response.status_code == 200
            events = _parse_sse_events(response.read().decode("utf-8"))
            assert len(events) >= 2
            terminal_event = events[-1]
            assert terminal_event["event"] == "error"
            assert terminal_event["data"]["code"] == "stream_interrupted"
            assert terminal_event["data"]["retryable"] is False


def test_chat_stream_client_disconnect_releases_semaphore(tmp_path: Path) -> None:
    """Early client disconnect signals worker thread and releases semaphore."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_MAX_CONCURRENT_GENERATIONS=1,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    worker_stopped = threading.Event()

    def slow_stream(question: str, history: any = (), request_id: str | None = None):
        req_id = request_id or "req-disc-1"
        yield PipelineEvent.create("retrieval_started", {}, request_id=req_id, sequence=1)
        for i in range(2, 50):
            time.sleep(0.05)
            yield PipelineEvent.create(
                "answer_delta",
                {"delta": f"token_{i} "},
                request_id=req_id,
                sequence=i,
            )
        worker_stopped.set()

    mock_pipeline.stream.side_effect = slow_stream
    app = create_app(settings=settings, pipeline=mock_pipeline)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={"question": "What is John's experience?"},
        ) as response:
            assert response.status_code == 200
            # Read only first line then exit context manager immediately (disconnect)
            for line in response.iter_lines():
                if line.startswith("data:"):
                    break

        # Verify semaphore is released and a subsequent request can succeed
        assert not app.state.generation_semaphore.locked()

        # Follow-up fast request succeeds immediately
        mock_pipeline.stream.side_effect = lambda question, history=(), request_id=None: [
            PipelineEvent.create("retrieval_started", {}, request_id=request_id or "req2", sequence=1),
            PipelineEvent.create(
                "answer_completed",
                {"result": AnswerResult(answer="Done", decision="answered"), "replace_streamed_answer": False},
                request_id=request_id or "req2",
                sequence=2,
            ),
        ]
        with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={"question": "Quick question?"},
        ) as response2:
            assert response2.status_code == 200
            events2 = _parse_sse_events(response2.read().decode("utf-8"))
            assert len(events2) == 2
            assert events2[-1]["event"] == "answer_completed"


def test_chat_stream_end_to_end_with_fake_providers(tmp_path: Path) -> None:
    """End-to-end SSE chat stream with FakeProviderFactories and RAGPipeline."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "sample_cv.txt").write_text(
        "Alice Smith is a Principal Cloud Architect specializing in AWS and Terraform.",
        encoding="utf-8",
    )

    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=docs_dir,
        VECTOR_DB_DIR=tmp_path / "vector_db",
    )
    factories = FakeProviderFactories(
        default_response="Alice Smith is a Principal Cloud Architect [S1]."
    )

    pipeline = RAGPipeline(settings=settings, provider_factories=factories)
    app = create_app(settings=settings, pipeline=pipeline)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={"question": "Who is Alice Smith?"},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            events = _parse_sse_events(response.read().decode("utf-8"))
            assert len(events) >= 4
            assert events[0]["event"] == "retrieval_started"
            assert events[1]["event"] == "retrieval_completed"
            assert events[2]["event"] == "generation_started"
            assert events[-1]["event"] == "answer_completed"
            assert "Alice Smith" in events[-1]["data"]["result"]["answer"]
            assert events[-1]["data"]["result"]["citations"][0]["source"] == "sample_cv.txt"
    app.state.pipeline = None
    del pipeline
    import gc
    gc.collect()


def test_chat_sync_chunked_body_exceeding_limit_returns_413(tmp_path: Path) -> None:
    """Streaming chunked request exceeding API_MAX_BODY_BYTES aborts with 413."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_MAX_BODY_BYTES=100,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")
    app = create_app(settings=settings, pipeline=mock_pipeline)

    def oversized_chunks():
        for _ in range(5):
            yield b"x" * 30

    with TestClient(app) as client:
        res = client.post(
            "/api/v1/chat",
            content=oversized_chunks(),
            headers={"Content-Type": "application/json"},
        )
        assert res.status_code == 413
        assert res.json()["error"]["code"] == "request_too_large"


def test_chat_question_whitespace_normalization_allows_valid_1000_char_core(tmp_path: Path) -> None:
    """A question with leading/trailing whitespace whose trimmed length is exactly MAX_QUESTION_CHARS is accepted."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        MAX_QUESTION_CHARS=1000,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")
    mock_pipeline.answer.return_value = AnswerResult(answer="Answered", decision="answered")
    app = create_app(settings=settings, pipeline=mock_pipeline)

    padded_question = "   " + ("q" * 1000) + "   "
    with TestClient(app) as client:
        res = client.post(
            "/api/v1/chat",
            json={"question": padded_question},
        )
        assert res.status_code == 200
        assert res.json()["answer"] == "Answered"


def test_multiple_chat_requests_reuse_single_pipeline_instance(tmp_path: Path) -> None:
    """Sequential chat requests reuse the existing pipeline instance without re-instantiation."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "cv.txt").write_text("Experienced Software Engineer.", encoding="utf-8")
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=docs_dir,
        VECTOR_DB_DIR=tmp_path / "vdb",
    )
    factories = FakeProviderFactories(default_response="Engineered response.")
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)
    app = create_app(settings=settings, pipeline=pipeline)

    with TestClient(app) as client:
        res1 = client.post("/api/v1/chat", json={"question": "Question 1?"})
        assert res1.status_code == 200

        res2 = client.post("/api/v1/chat", json={"question": "Question 2?"})
        assert res2.status_code == 200

        assert app.state.pipeline is pipeline
    app.state.pipeline = None
    del pipeline
    import gc
    gc.collect()


def test_chat_stream_client_disconnect_stops_producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disconnecting from SSE stream terminates the producer loop and releases concurrency capacity."""
    settings = Settings(
        NVIDIA_API_KEY="mock-key",
        DOCUMENTS_DIR=tmp_path / "docs",
        VECTOR_DB_DIR=tmp_path / "vector_db",
        API_MAX_CONCURRENT_GENERATIONS=1,
    )
    mock_pipeline = MagicMock(spec=RAGPipeline)
    mock_pipeline.readiness.return_value = ReadinessResult(state="ready")

    producer_closed = threading.Event()

    class CancellableGenerator:
        def __init__(self):
            self.stopped = False
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.stopped or self.count >= 20:
                raise StopIteration
            self.count += 1
            return PipelineEvent.create("answer_delta", {"delta": "token "}, request_id="req-d", sequence=self.count)

        def close(self):
            self.stopped = True
            producer_closed.set()

    mock_pipeline.stream.return_value = CancellableGenerator()
    app = create_app(settings=settings, pipeline=mock_pipeline)

    disconnect_called = [0]

    async def fake_is_disconnected(self):
        disconnect_called[0] += 1
        return disconnect_called[0] >= 3

    monkeypatch.setattr(Request, "is_disconnected", fake_is_disconnected)

    with TestClient(app) as client:
        with client.stream("POST", "/api/v1/chat/stream", json={"question": "Test disconnect"}) as res:
            assert res.status_code == 200
            for line in res.iter_lines():
                if line.startswith("data:"):
                    break

        assert producer_closed.wait(timeout=2.0)
        assert not app.state.generation_semaphore.locked()

