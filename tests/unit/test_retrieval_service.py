from __future__ import annotations

from rag.retrieval.base import ChunkRecord, RankedChunk
from rag.retrieval.reranker import NoOpReranker
from rag.retrieval.service import RetrievalService
from rag.settings import Settings


class StaticRetriever:
    def __init__(self, candidates):
        self.candidates = candidates

    def search(self, query, k):
        return self.candidates[:k]


def test_hybrid_service_fuses_candidates_and_returns_final_context() -> None:
    """Hybrid mode exposes fused diagnostics while handing only final context onward."""
    c1 = ChunkRecord("c1", "Python", "cv.md", 0, 0)
    c2 = ChunkRecord("c2", "Kubernetes", "cv.md", 0, 1)
    settings = Settings(
        RETRIEVAL_MODE="hybrid", CONTEXT_K=2, DENSE_CANDIDATE_K=2,
        SPARSE_CANDIDATE_K=2, FUSION_CANDIDATE_K=2, RERANK_CANDIDATE_K=2,
    )
    service = RetrievalService(
        settings=settings,
        dense_retriever=StaticRetriever([RankedChunk(c1, 1, 0.8, ("dense",), {"dense": 0.8})]),
        sparse_retriever=StaticRetriever([RankedChunk(c2, 1, 3.0, ("sparse",), {"sparse": 3.0})]),
        reranker=NoOpReranker(),
        index_fingerprint="index-fp",
    )

    result = service.retrieve("skills")

    assert [item.chunk.chunk_id for item in result.context] == ["c1", "c2"]
    assert len(result.fused_candidates) == 2
    assert result.degraded_reason_codes == ()
    assert result.index_fingerprint == "index-fp"
