"""Pluggable rerankers for fused retrieval candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

from rag.retrieval.base import RankedChunk


class NoOpReranker:
    """Preserves fused order when reranking is disabled or unavailable."""

    def rerank(self, query: str, candidates: Sequence[RankedChunk], top_n: int) -> list[RankedChunk]:
        return [
            RankedChunk(candidate.chunk, rank, candidate.score, candidate.methods, candidate.stage_scores)
            for rank, candidate in enumerate(candidates[:max(top_n, 0)], start=1)
        ]


class NvidiaReranker:
    """Uses the public NVIDIA compression API and maps output back to candidates."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def rerank(self, query: str, candidates: Sequence[RankedChunk], top_n: int) -> list[RankedChunk]:
        if not candidates or top_n <= 0:
            return []
        by_id = {candidate.chunk.chunk_id: candidate for candidate in candidates}
        if len(by_id) != len(candidates):
            raise ValueError("reranker candidates must have unique chunk_ids")
        documents = [
            Document(page_content=candidate.chunk.content, metadata={"chunk_id": candidate.chunk.chunk_id, "prior_rank": candidate.rank, **dict(candidate.chunk.metadata)})
            for candidate in candidates
        ]
        result = self.client.compress_documents(documents=documents, query=query)
        reranked: list[RankedChunk] = []
        seen: set[str] = set()
        for rank, document in enumerate(result, start=1):
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if chunk_id not in by_id:
                raise ValueError(f"reranker returned unknown chunk_id: {chunk_id}")
            if chunk_id in seen:
                raise ValueError(f"reranker returned duplicate chunk_id: {chunk_id}")
            seen.add(chunk_id)
            original = by_id[chunk_id]
            raw_score = document.metadata.get("relevance_score")
            score = float(raw_score) if raw_score is not None else original.score
            stage_scores = dict(original.stage_scores)
            if raw_score is not None:
                stage_scores["reranker"] = score
            reranked.append(RankedChunk(original.chunk, rank, score, original.methods, stage_scores))
            if len(reranked) == top_n:
                break
        return reranked
