from __future__ import annotations

from rag.ingestion.indexer import IndexSynchronizer
from rag.settings import Settings
from tests.fakes import FakeEmbeddings


def test_indexer_reuses_matching_active_snapshot_and_rebuilds_changed_corpus(tmp_path) -> None:
    """An unchanged corpus is reused, while changed bytes activate a new immutable build."""
    documents = tmp_path / "documents"
    documents.mkdir()
    source = documents / "Custom_CV.md"
    source.write_text("Python and C# engineer", encoding="utf-8")
    settings = Settings(
        NVIDIA_API_KEY="test-key",
        DOCUMENTS_DIR=documents,
        VECTOR_DB_DIR=tmp_path / "vector_db",
        INDEX_SCHEMA_VERSION="v4",
    )
    indexer = IndexSynchronizer(settings=settings, embeddings=FakeEmbeddings(dimension=8))

    first = indexer.ensure_active_index()
    reused = indexer.ensure_active_index()
    source.write_text("Python and Kubernetes engineer", encoding="utf-8")
    rebuilt = indexer.ensure_active_index()

    assert first.index_id == reused.index_id
    assert first.chunks == reused.chunks
    assert rebuilt.index_id != first.index_id
    assert (settings.VECTOR_DB_DIR / "active-index.json").is_file()
    assert all("Kubernetes" in chunk.content for chunk in rebuilt.chunks)
