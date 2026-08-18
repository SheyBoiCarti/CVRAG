from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from rag.retrieval.base import ChunkRecord, RankedChunk


def test_ranked_chunk_exposes_immutable_retrieval_evidence() -> None:
    """A retrieval result cannot be changed after its rank has been assigned."""
    record = ChunkRecord(
        chunk_id="chunk-python",
        content="Built Python services.",
        source="Custom_CV.md",
        page=0,
        chunk_index=2,
        metadata={"section": "experience"},
    )
    ranked = RankedChunk(
        chunk=record,
        rank=1,
        score=0.83,
        methods=("dense",),
        stage_scores={"dense": 0.83},
    )

    assert ranked.chunk.chunk_id == "chunk-python"
    assert ranked.rank == 1
    assert ranked.stage_scores == {"dense": 0.83}
    with pytest.raises(FrozenInstanceError):
        ranked.rank = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        ranked.chunk.metadata["section"] = "skills"  # type: ignore[index]
