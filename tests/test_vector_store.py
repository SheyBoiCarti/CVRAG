from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pytest

from rag.vector_store import (
    VectorStoreManager,
    provider_error_details,
    retry_transient,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


def test_ensure_documents_is_idempotent(tmp_path):
    chunks = [
        Document(
            page_content="Python",
            metadata={"chunk_id": "chunk-1", "source": "cv.pdf", "page": 0},
        )
    ]
    manager = VectorStoreManager(persist_dir=tmp_path, embeddings=FakeEmbeddings())

    assert manager.ensure_documents(chunks) == 1
    assert manager.ensure_documents(chunks) == 1
    assert manager.get_document_count() == 1


def test_permanent_embedding_failure_is_not_retried():
    attempts = 0

    def fail():
        nonlocal attempts
        attempts += 1
        raise ValueError("invalid request")

    with pytest.raises(ValueError, match="invalid request"):
        retry_transient(fail, max_attempts=3)

    assert attempts == 1


def test_transient_embedding_failure_is_bounded(monkeypatch):
    attempts = 0
    monkeypatch.setattr("rag.vector_store.time.sleep", lambda _delay: None)
    monkeypatch.setattr("rag.vector_store.random.uniform", lambda _a, _b: 0)

    def fail():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("temporary")

    with pytest.raises(TimeoutError, match="temporary"):
        retry_transient(fail, max_attempts=3)

    assert attempts == 3


@pytest.mark.parametrize(
    "exc,expected",
    [
        (TimeoutError("request timed out"), True),
        (ConnectionError("connection reset"), True),
        (Exception("[500] Something went wrong with the request."), True),
        (Exception("[599] Upstream failure"), True),
        (Exception("[600] Non-HTTP status"), False),
        (Exception("[503] ResourceExhausted: Worker limit reached"), True),
        (Exception("[429] Rate limit exceeded"), True),
        (Exception("[404] Not Found"), False),
        (ValueError("Invalid argument"), False),
    ],
)
def test_is_transient_error_classification(exc, expected):
    from rag.vector_store import is_transient_error

    assert is_transient_error(exc) is expected


def test_retry_transient_succeeds_after_transient_500(monkeypatch):
    attempts = 0
    monkeypatch.setattr("rag.vector_store.time.sleep", lambda _delay: None)
    monkeypatch.setattr("rag.vector_store.random.uniform", lambda _a, _b: 0)

    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise Exception("[500] Something went wrong with the request.")
        return "success"

    result = retry_transient(flaky, max_attempts=3)
    assert result == "success"
    assert attempts == 2


def test_provider_error_details_extracts_status_and_safe_summary():
    exc = Exception("[500] Something went wrong with the request.\n{'status': 500}")

    assert provider_error_details(exc) == (
        500,
        "[500] Something went wrong with the request.",
    )


def test_provider_error_details_omits_multiline_response_body():
    exc = Exception("[422] Invalid request\nCV text echoed by provider")

    status, detail = provider_error_details(exc)

    assert status == 422
    assert detail == "[422] Invalid request"
    assert "CV text" not in detail


def test_provider_error_details_redacts_credentials():
    exc = Exception(
        "Authorization: Bearer secret-token NVIDIA_API_KEY=nvapi-secret-value"
    )

    status, detail = provider_error_details(exc)

    assert status is None
    assert "secret-token" not in detail
    assert "nvapi-secret-value" not in detail
    assert detail == "Authorization: Bearer [REDACTED] NVIDIA_API_KEY=[REDACTED]"


def test_retry_log_has_safe_provider_context(monkeypatch, caplog):
    monkeypatch.setattr("rag.vector_store.time.sleep", lambda _delay: None)
    monkeypatch.setattr("rag.vector_store.random.uniform", lambda _a, _b: 0)

    def fail():
        raise Exception("[503] unavailable NVIDIA_API_KEY=nvapi-secret")

    with caplog.at_level("WARNING"), pytest.raises(Exception):
        retry_transient(
            fail,
            max_attempts=2,
            operation_name="embed_documents",
            model="nvidia/nv-embedqa-e5-v5",
        )

    assert "operation=embed_documents" in caplog.text
    assert "model=nvidia/nv-embedqa-e5-v5" in caplog.text
    assert "status=503" in caplog.text
    assert "nvapi-secret" not in caplog.text

