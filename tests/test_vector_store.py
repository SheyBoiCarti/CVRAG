from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pytest

from rag.vector_store import VectorStoreManager, retry_transient


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
