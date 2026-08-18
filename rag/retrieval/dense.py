"""Dense retrieval over the public Chroma persistence adapter."""

from __future__ import annotations

from rag.persistence.chroma_store import ChromaStoreAdapter
from rag.retrieval.base import RankedChunk


class DenseRetriever:
    """Converts Chroma relevance results into immutable dense-ranked chunks."""

    def __init__(self, store: ChromaStoreAdapter) -> None:
        self.store = store

    def search(self, query: str, k: int) -> list[RankedChunk]:
        if not query.strip() or k <= 0:
            return []
        return [
            RankedChunk(chunk=chunk, rank=rank, score=score, methods=("dense",), stage_scores={"dense": score})
            for rank, (chunk, score) in enumerate(self.store.similarity_search_with_scores(query, k=k), start=1)
        ]
