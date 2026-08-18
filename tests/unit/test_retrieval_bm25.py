from __future__ import annotations

from rag.retrieval.base import ChunkRecord
from rag.retrieval.bm25 import BM25Retriever, tokenize_technical_text


def test_bm25_preserves_technical_tokens_and_ranks_exact_match() -> None:
    """Sparse retrieval keeps technical punctuation and favors a rare exact term."""
    tokens = tokenize_technical_text("C# .NET, Node.js CI/CD, C++")
    assert tokens == ["c#", ".net", "node.js", "ci/cd", "c++"]

    retriever = BM25Retriever([
        ChunkRecord("c1", "Built C# .NET services", "cv.md", 0, 0),
        ChunkRecord("c2", "Led product discovery workshops", "cv.md", 0, 1),
        ChunkRecord("c3", "Maintained Python tools", "cv.md", 0, 2),
    ])

    results = retriever.search("C# .NET", k=3)
    assert [item.chunk.chunk_id for item in results] == ["c1"]
    assert results[0].methods == ("sparse",)
    assert results[0].stage_scores["sparse"] > 0
    assert retriever.search("   ", k=3) == []
