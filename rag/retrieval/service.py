"""Composition of dense, sparse, fusion, and optional reranking stages.

Architecture & Role:
-------------------
`RetrievalService` orchestrates the multi-stage hybrid retrieval pipeline:
1. Dense Retrieval: Semantic vector similarity using ChromaDB + embedding models (e.g. `nv-embedqa-e5-v5`).
2. Sparse Retrieval: Lexical BM25 keyword matching over tokenized corpus chunks to catch exact acronyms/terms.
3. Reciprocal Rank Fusion (RRF): Blends rank scores from both retrievers with configurable weights without
   requiring score calibration or score normalization.
4. Cross-Encoder Reranking (Optional): If enabled (`RERANK_ENABLED=True`), sends top-K fused candidates to NVIDIA
   cross-encoder for fine-grained semantic relevance scoring.
5. Degradation Handling: If a single retrieval mode fails transiently, logs diagnostic codes and degrades gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag.retrieval.base import RankedChunk, Reranker, Retriever
from rag.retrieval.rrf import reciprocal_rank_fusion
from rag.settings import Settings
from rag.vector_store import is_transient_error

LOGGER = logging.getLogger(__name__)


class RetrievalUnavailableError(RuntimeError):
    """Raised when no retrieval stage can produce context."""


@dataclass(frozen=True)
class RetrievalResult:
    """Immutable envelope carrying retrieved chunks, intermediate stage candidates, and diagnostics."""

    query: str
    context: tuple[RankedChunk, ...]
    fused_candidates: tuple[RankedChunk, ...]
    degraded_reason_codes: tuple[str, ...]
    index_fingerprint: str
    dense_candidates: tuple[RankedChunk, ...] = ()
    sparse_candidates: tuple[RankedChunk, ...] = ()
    reranked_candidates: tuple[RankedChunk, ...] = ()


class RetrievalService:
    """Runs configured retrieval stages in their prescribed deterministic order."""

    def __init__(self, *, settings: Settings, dense_retriever: Retriever, sparse_retriever: Retriever | None, reranker: Reranker, index_fingerprint: str) -> None:
        self.settings = settings
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.index_fingerprint = index_fingerprint

    def retrieve(self, query: str) -> RetrievalResult:
        degraded: list[str] = []
        dense = self._search("dense", self.dense_retriever, query, self.settings.DENSE_CANDIDATE_K, degraded)
        sparse: list[RankedChunk] = []
        if self.settings.RETRIEVAL_MODE == "dense":
            candidates = dense
        else:
            if self.sparse_retriever is None:
                raise RetrievalUnavailableError("sparse retriever is not configured")
            sparse = self._search("sparse", self.sparse_retriever, query, self.settings.SPARSE_CANDIDATE_K, degraded)
            if not dense and not sparse and len(degraded) == 2:
                raise RetrievalUnavailableError("all hybrid retrieval stages are unavailable")
            candidates = reciprocal_rank_fusion(
                {"dense": dense, "sparse": sparse}, rank_constant=self.settings.RRF_CONSTANT,
                weights={"dense": self.settings.DENSE_RRF_WEIGHT, "sparse": self.settings.SPARSE_RRF_WEIGHT}, limit=self.settings.FUSION_CANDIDATE_K,
            )
        fused = tuple(candidates)
        final = list(candidates)
        reranked: list[RankedChunk] = []
        if self.settings.RERANK_ENABLED and final:
            try:
                final = self.reranker.rerank(query, final[:self.settings.RERANK_CANDIDATE_K], self.settings.CONTEXT_K)
                reranked = list(final)
            except Exception as exc:
                if not is_transient_error(exc):
                    raise
                LOGGER.warning("Reranker unavailable (type=%s)", type(exc).__name__)
                degraded.append("reranker_unavailable")
                final = final[:self.settings.CONTEXT_K]
        else:
            final = final[:self.settings.CONTEXT_K]
        return RetrievalResult(
            query=query,
            context=tuple(final),
            fused_candidates=fused,
            degraded_reason_codes=tuple(degraded),
            index_fingerprint=self.index_fingerprint,
            dense_candidates=tuple(dense),
            sparse_candidates=tuple(sparse),
            reranked_candidates=tuple(reranked),
        )

    @staticmethod
    def _search(name: str, retriever: Retriever, query: str, k: int, degraded: list[str]) -> list[RankedChunk]:
        try:
            return retriever.search(query, k)
        except Exception as exc:
            if not is_transient_error(exc):
                raise
            LOGGER.warning("%s retrieval unavailable (type=%s)", name, type(exc).__name__)
            degraded.append(f"{name}_unavailable")
            return []
