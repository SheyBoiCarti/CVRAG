"""Weighted reciprocal-rank fusion with deterministic diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from rag.retrieval.base import RankedChunk


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RankedChunk]], *, rank_constant: int, weights: Mapping[str, float], limit: int
) -> list[RankedChunk]:
    """Fuses stage rankings using weighted reciprocal rank contributions."""
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("weights must be finite and nonnegative")
    if not any(weight > 0 for weight in weights.values()):
        raise ValueError("at least one weight must be positive")
    unknown = set(rankings).difference(weights)
    if unknown:
        raise ValueError(f"missing weights for ranking methods: {', '.join(sorted(unknown))}")

    accumulated: dict[str, dict[str, Any]] = {}
    for method, ranking in rankings.items():
        seen: set[str] = set()
        for candidate in ranking:
            chunk_id = candidate.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            entry = accumulated.setdefault(chunk_id, {
                "chunk": candidate.chunk, "score": 0.0, "methods": [], "stage_scores": {}, "ranks": {},
            })
            entry["score"] += weights[method] / (rank_constant + candidate.rank)
            entry["methods"].append(method)
            entry["stage_scores"].update(candidate.stage_scores)
            entry["ranks"][method] = candidate.rank

    ordered = sorted(
        accumulated.values(),
        key=lambda entry: (
            -entry["score"], min(entry["ranks"].values()), entry["ranks"].get("dense", math.inf),
            entry["ranks"].get("sparse", math.inf), entry["chunk"].chunk_id,
        ),
    )
    return [
        RankedChunk(
            chunk=entry["chunk"], rank=rank, score=entry["score"], methods=tuple(entry["methods"]),
            stage_scores={**entry["stage_scores"], "rrf": entry["score"]},
        )
        for rank, entry in enumerate(ordered[:limit], start=1)
    ]
