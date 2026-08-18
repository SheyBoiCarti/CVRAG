from __future__ import annotations

from rag.persistence.chroma_store import ChromaStoreAdapter
from rag.retrieval.base import ChunkRecord
from rag.retrieval.dense import DenseRetriever
from tests.fakes import FakeEmbeddings


def test_dense_retriever_preserves_chroma_score_as_diagnostic(tmp_path) -> None:
    """Dense candidates expose their original relevance score without conversion."""
    store = ChromaStoreAdapter(persist_dir=tmp_path, collection_name="dense_test", embeddings=FakeEmbeddings(dimension=8))
    store.add_chunks([ChunkRecord("c1", "Python engineering", "cv.md", 0, 0)])

    result = DenseRetriever(store).search("Python", k=1)[0]

    assert result.chunk.chunk_id == "c1"
    assert result.rank == 1
    assert result.methods == ("dense",)
    assert result.score == result.stage_scores["dense"]
