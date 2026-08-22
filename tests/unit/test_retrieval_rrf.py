from __future__ import annotations

import pytest

from rag.retrieval.base import ChunkRecord, RankedChunk
from rag.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_accumulates_both_retrievers_and_breaks_ties_deterministically() -> None:
    """A shared candidate gets both hand-calculated reciprocal-rank contributions."""
    c1 = ChunkRecord("c1", "First", "cv.md", 0, 0)
    c2 = ChunkRecord("c2", "Second", "cv.md", 0, 1)
    fused = reciprocal_rank_fusion(
        {
            "dense": [RankedChunk(c1, 1, 0.8, ("dense",), {"dense": 0.8}), RankedChunk(c2, 2, 0.7, ("dense",), {"dense": 0.7})],
            "sparse": [RankedChunk(c2, 1, 4.0, ("sparse",), {"sparse": 4.0}), RankedChunk(c1, 2, 3.0, ("sparse",), {"sparse": 3.0})],
        },
        rank_constant=60,
        weights={"dense": 1.0, "sparse": 1.0},
        limit=2,
    )

    assert [item.chunk.chunk_id for item in fused] == ["c1", "c2"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert fused[0].methods == ("dense", "sparse")
    assert fused[0].stage_scores == {"dense": 0.8, "sparse": 3.0, "rrf": fused[0].score}
