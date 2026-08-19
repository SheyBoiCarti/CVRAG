from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import pytest
from langchain_core.documents import Document

from rag.ingestion.chunking.base import Chunker, compute_chunk_id
from rag.retrieval.base import ChunkRecord


def test_compute_chunk_id_deterministic() -> None:
    source = "resume.md"
    page = 0
    strategy = "recursive_bullet_v1"
    parameters = {"chunk_size": 700, "chunk_overlap": 140}
    chunk_index = 0
    content = "Sample content"

    id1 = compute_chunk_id(
        source=source,
        page=page,
        strategy=strategy,
        parameters=parameters,
        chunk_index=chunk_index,
        content=content,
    )
    id2 = compute_chunk_id(
        source=source,
        page=page,
        strategy=strategy,
        parameters={"chunk_overlap": 140, "chunk_size": 700},  # different dict order
        chunk_index=chunk_index,
        content=content,
    )
    assert id1 == id2
    assert len(id1) == 64  # sha256 hex string


def test_compute_chunk_id_sensitivity() -> None:
    base_args = {
        "source": "resume.md",
        "page": 0,
        "strategy": "recursive_bullet_v1",
        "parameters": {"chunk_size": 700, "chunk_overlap": 140},
        "chunk_index": 0,
        "content": "Sample content",
    }
    base_id = compute_chunk_id(**base_args)

    for field, new_val in [
        ("source", "other.md"),
        ("page", 1),
        ("strategy", "fixed_character_v1"),
        ("parameters", {"chunk_size": 500, "chunk_overlap": 100}),
        ("chunk_index", 1),
        ("content", "Other content"),
    ]:
        modified_args = dict(base_args)
        modified_args[field] = new_val
        new_id = compute_chunk_id(**modified_args)
        assert new_id != base_id, f"Changing {field} must change chunk_id"


class DummyChunker:
    @property
    def strategy_name(self) -> str:
        return "dummy_v1"

    def parameters(self) -> Mapping[str, Any]:
        return {"param": 1}

    def split(self, documents: Sequence[Document]) -> list[ChunkRecord]:
        return []


def test_chunker_protocol_conformance() -> None:
    chunker: Chunker = DummyChunker()
    assert chunker.strategy_name == "dummy_v1"
    assert chunker.parameters() == {"param": 1}
    assert chunker.split([]) == []
