"""Deterministic technical-token BM25 sparse retrieval."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from rag.retrieval.base import ChunkRecord, RankedChunk

_TECHNICAL_TOKEN = re.compile(r"(?:\.[\w]+(?:[./+#_-][\w]+)*(?:[#+]+)?|[\w]+(?:[./+#_-][\w]+)*(?:[#+]+)?)", re.UNICODE)


def tokenize_technical_text(text: str) -> list[str]:
    """Normalizes text while retaining meaningful technical token punctuation."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [match.group(0) for match in _TECHNICAL_TOKEN.finditer(normalized)]


class BM25Retriever:
    """In-memory sparse retriever built from the canonical chunk snapshot."""

    def __init__(self, chunks: Sequence[ChunkRecord], *, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> None:
        self.chunks = tuple(chunks)
        self._index = (
            BM25Okapi([tokenize_technical_text(chunk.content) for chunk in self.chunks], k1=k1, b=b, epsilon=epsilon)
            if self.chunks else None
        )

    def search(self, query: str, k: int) -> list[RankedChunk]:
        """Ranks positive BM25 matches with chunk-ID deterministic ties."""
        tokens = tokenize_technical_text(query)
        if not tokens or k <= 0 or self._index is None:
            return []
        scored = [
            (self.chunks[index], float(score))
            for index, score in enumerate(self._index.get_scores(tokens))
            if score > 0
        ]
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return [
            RankedChunk(chunk=chunk, rank=rank, score=score, methods=("sparse",), stage_scores={"sparse": score})
            for rank, (chunk, score) in enumerate(scored[:k], start=1)
        ]


__all__ = ["BM25Retriever", "tokenize_technical_text"]
