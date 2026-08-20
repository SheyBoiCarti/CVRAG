"""Base contracts and deterministic identity functions for document chunking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from langchain_core.documents import Document

from rag.retrieval.base import ChunkRecord


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_chunk_id(
    *,
    source: str,
    page: int,
    strategy: str,
    parameters: Mapping[str, Any],
    chunk_index: int,
    content: str,
) -> str:
    """Computes deterministic SHA-256 chunk identity from all identity fields."""
    canonical_params = _canonical_json(dict(parameters)).decode("utf-8")
    identity = f"{source}:{page}:{strategy}:{canonical_params}:{chunk_index}:{content}".encode()
    return hashlib.sha256(identity).hexdigest()


class Chunker(Protocol):
    """Protocol for interchangeable document chunking strategies."""

    @property
    def strategy_name(self) -> str:
        """The canonical name of the chunking strategy."""
        ...

    def parameters(self) -> Mapping[str, Any]:
        """Effective parameters that configure this chunker instance."""
        ...

    def split(self, documents: Sequence[Document]) -> list[ChunkRecord]:
        """Splits incoming documents into an ordered list of ChunkRecords."""
        ...
