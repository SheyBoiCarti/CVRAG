from __future__ import annotations

from rag.pipeline import RAGPipeline
from rag.settings import Settings
from tests.fakes import FakeProviderFactories


def test_hybrid_pipeline_indexes_once_and_answers_with_offline_providers(tmp_path) -> None:
    """A hybrid request uses one canonical index without any provider network calls."""
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "cv.md").write_text("Kubernetes and C# platform engineer", encoding="utf-8")
    pipeline = RAGPipeline(
        settings=Settings(NVIDIA_API_KEY="test", DOCUMENTS_DIR=documents, VECTOR_DB_DIR=tmp_path / "db", RETRIEVAL_MODE="hybrid"),
        provider_factories=FakeProviderFactories(default_response="Platform engineer [S1]."),
    )

    result = pipeline.answer("What platform experience is listed?")

    assert result.decision == "answered"
    assert result.citations[0].source == "cv.md"
    assert pipeline.readiness().index_fingerprint is not None
