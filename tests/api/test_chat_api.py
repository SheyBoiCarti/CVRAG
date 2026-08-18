from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from backend.app.api.schemas.chat import ChatResponse
from backend.app.api.schemas.common import ErrorResponse
from backend.app.main import create_app
from rag.errors import ProviderUnavailableError
from rag.models import AnswerResult, Citation, ReadinessResult
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
