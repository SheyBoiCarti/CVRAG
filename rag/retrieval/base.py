"""Immutable retrieval data contracts shared by all retrieval stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol


def _immutable_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True)
class ChunkRecord:
    """A canonical, source-addressable chunk from an indexed corpus snapshot."""

    chunk_id: str
    content: str
    source: str
    page: int
    chunk_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Returns the canonical JSON-serializable chunk representation."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RankedChunk:
    """A chunk together with deterministic stage-local ranking diagnostics."""

    chunk: ChunkRecord
    rank: int
    score: float | None
    methods: tuple[str, ...]
    stage_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "stage_scores", _immutable_mapping(self.stage_scores))

    def to_dict(self) -> dict[str, Any]:
        """Returns canonical JSON-serializable ranked chunk representation."""
        return {
            "chunk": self.chunk.to_dict(),
            "rank": self.rank,
            "score": self.score,
            "methods": list(self.methods),
            "stage_scores": dict(self.stage_scores),
        }


class Retriever(Protocol):
    """Searches a corpus and returns stage-ranked canonical chunks."""

    def search(self, query: str, k: int) -> list[RankedChunk]: ...


class Reranker(Protocol):
    """Reorders an already-ranked candidate set for final context selection."""

    def rerank(
        self, query: str, candidates: Sequence[RankedChunk], top_n: int
    ) -> list[RankedChunk]: ...


__all__ = ["ChunkRecord", "RankedChunk", "Reranker", "Retriever"]
