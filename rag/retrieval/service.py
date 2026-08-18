"""Composition of dense, sparse, fusion, and optional reranking stages."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

from rag.retrieval.base import RankedChunk, Reranker, Retriever
from rag.retrieval.rrf import reciprocal_rank_fusion
from rag.settings import Settings
from rag.vector_store import is_transient_error


LOGGER = logging.getLogger(__name__)


class RetrievalUnavailableError(RuntimeError):
    """Raised when no retrieval stage can produce context."""


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    context: tuple[RankedChunk, ...]
    fused_candidates: tuple[RankedChunk, ...]
    degraded_reason_codes: tuple[str, ...]
    index_fingerprint: str


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
        if self.settings.RERANK_ENABLED and final:
            try:
                final = self.reranker.rerank(query, final[:self.settings.RERANK_CANDIDATE_K], self.settings.CONTEXT_K)
            except Exception as exc:
                if not is_transient_error(exc):
                    raise
                LOGGER.warning("Reranker unavailable (type=%s)", type(exc).__name__)
                degraded.append("reranker_unavailable")
                final = final[:self.settings.CONTEXT_K]
        else:
            final = final[:self.settings.CONTEXT_K]
        return RetrievalResult(query, tuple(final), fused, tuple(degraded), self.index_fingerprint)

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
