from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.retrieval.base import ChunkRecord, RankedChunk
from rag.retrieval.reranker import NoOpReranker, NvidiaReranker


def test_noop_reranker_preserves_fused_order_and_truncates() -> None:
    """Disabled reranking cannot silently reorder the fused candidates."""
    candidates = [
        RankedChunk(ChunkRecord("c1", "First", "cv.md", 0, 0), 1, 0.5, ("dense",), {"rrf": 0.5}),
        RankedChunk(ChunkRecord("c2", "Second", "cv.md", 0, 1), 2, 0.4, ("sparse",), {"rrf": 0.4}),
    ]
    assert [candidate.chunk.chunk_id for candidate in NoOpReranker().rerank("query", candidates, 1)] == ["c1"]


def test_nvidia_reranker_rejects_unknown_provider_chunk_ids() -> None:
    """Provider results cannot inject evidence that was not a retrieval candidate."""
    class Client:
        def compress_documents(self, documents, query):
            return [Document(page_content="unknown", metadata={"chunk_id": "not-a-candidate", "relevance_score": 0.9})]

    candidates = [RankedChunk(ChunkRecord("c1", "First", "cv.md", 0, 0), 1, 0.5, ("dense",), {})]
    with pytest.raises(ValueError, match="unknown chunk_id"):
        NvidiaReranker(Client()).rerank("query", candidates, 1)
