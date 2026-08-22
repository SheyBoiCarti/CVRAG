"""Deterministic retrieval metrics: Hit@K, Recall@K, AllEvidence@K, MRR, and nDCG@K."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from evaluation.schema import EvidenceLocator, GoldenCase
from rag.retrieval.base import RankedChunk


def normalize_text(text: str) -> str:
    """Normalizes text using Unicode NFKC, casefolding, and collapsing whitespace."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def matches_locator(chunk: RankedChunk, locator: EvidenceLocator) -> bool:
    """Checks if a retrieved RankedChunk matches an EvidenceLocator.

    A chunk matches an evidence locator when:
    1. Source matches (case-insensitive filename match).
    2. Page matches.
    3. Normalized locator anchor text is contained within normalized chunk content.
    """
    if chunk.chunk.source.strip().casefold() != locator.source.strip().casefold():
        return False
    if chunk.chunk.page != locator.page:
        return False
    norm_anchor = normalize_text(locator.anchor)
    norm_content = normalize_text(chunk.chunk.content)
    return norm_anchor in norm_content


@dataclass(frozen=True)
class RetrievalCaseMetrics:
    """Retrieval evaluation metrics for an individual case."""

    case_id: str
    is_applicable: bool
    hit_at_k: Mapping[int, float] = field(default_factory=dict)
    recall_at_k: Mapping[int, float] = field(default_factory=dict)
    all_evidence_at_k: Mapping[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Mapping[int, float] = field(default_factory=dict)
    first_match_rank: int | None = None
    matched_groups_by_k: Mapping[int, int] = field(default_factory=dict)
    total_required_groups: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hit_at_k", MappingProxyType(dict(self.hit_at_k)))
        object.__setattr__(self, "recall_at_k", MappingProxyType(dict(self.recall_at_k)))
        object.__setattr__(self, "all_evidence_at_k", MappingProxyType(dict(self.all_evidence_at_k)))
        object.__setattr__(self, "ndcg_at_k", MappingProxyType(dict(self.ndcg_at_k)))
        object.__setattr__(self, "matched_groups_by_k", MappingProxyType(dict(self.matched_groups_by_k)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "is_applicable": self.is_applicable,
            "hit_at_k": {str(k): v for k, v in self.hit_at_k.items()},
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "all_evidence_at_k": {str(k): v for k, v in self.all_evidence_at_k.items()},
            "mrr": self.mrr,
            "ndcg_at_k": {str(k): v for k, v in self.ndcg_at_k.items()},
            "first_match_rank": self.first_match_rank,
            "total_required_groups": self.total_required_groups,
        }


def evaluate_retrieval(
    case: GoldenCase,
    retrieved_chunks: Sequence[RankedChunk],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalCaseMetrics:
    """Evaluates retrieved chunks against golden evidence groups for a single case."""
    if not case.answerable or not case.evidence_groups:
        return RetrievalCaseMetrics(
            case_id=case.id,
            is_applicable=False,
            total_required_groups=0,
        )

    total_groups = len(case.evidence_groups)
    sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.rank)

    # Find the rank of the first chunk matching ANY required evidence group
    first_match_rank: int | None = None
    for chunk in sorted_chunks:
        is_match = False
        for group in case.evidence_groups:
            for locator in group:
                if matches_locator(chunk, locator):
                    is_match = True
                    break
            if is_match:
                break
        if is_match:
            first_match_rank = chunk.rank
            break

    mrr = 1.0 / first_match_rank if first_match_rank is not None else 0.0

    hit_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}
    all_evidence_at_k: dict[int, float] = {}
    matched_groups_by_k: dict[int, int] = {}
    ndcg_at_k: dict[int, float] = {}

    for k in ks:
        top_k_chunks = sorted_chunks[:k]
        matched_groups_count = 0
        for group in case.evidence_groups:
            group_matched = False
            for locator in group:
                for chunk in top_k_chunks:
                    if matches_locator(chunk, locator):
                        group_matched = True
                        break
                if group_matched:
                    break
            if group_matched:
                matched_groups_count += 1

        hit = 1.0 if matched_groups_count > 0 else 0.0
        recall = matched_groups_count / total_groups if total_groups > 0 else 0.0
        all_evidence = 1.0 if matched_groups_count == total_groups else 0.0

        hit_at_k[k] = hit
        recall_at_k[k] = recall
        all_evidence_at_k[k] = all_evidence
        matched_groups_by_k[k] = matched_groups_count

        # Compute binary nDCG@K
        dcg = 0.0
        for i, chunk in enumerate(top_k_chunks, 1):
            rel = 1.0 if any(any(matches_locator(chunk, loc) for loc in g) for g in case.evidence_groups) else 0.0
            if rel > 0:
                dcg += rel / math.log2(i + 1)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(total_groups, k) + 1))
        ndcg_at_k[k] = (dcg / idcg) if idcg > 0 else 0.0

    return RetrievalCaseMetrics(
        case_id=case.id,
        is_applicable=True,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        all_evidence_at_k=all_evidence_at_k,
        mrr=mrr,
        ndcg_at_k=ndcg_at_k,
        first_match_rank=first_match_rank,
        matched_groups_by_k=matched_groups_by_k,
        total_required_groups=total_groups,
    )


def aggregate_retrieval_metrics(
    metrics_list: Sequence[RetrievalCaseMetrics],
) -> dict[str, float]:
    """Computes macro averages for applicable retrieval cases."""
    applicable = [m for m in metrics_list if m.is_applicable]
    if not applicable:
        return {"applicable_cases": 0}

    n = len(applicable)
    ks: set[int] = set()
    for m in applicable:
        ks.update(m.hit_at_k.keys())

    agg: dict[str, float] = {"applicable_cases": n}
    for k in sorted(ks):
        agg[f"hit_at_{k}"] = sum(m.hit_at_k.get(k, 0.0) for m in applicable) / n
        agg[f"recall_at_{k}"] = sum(m.recall_at_k.get(k, 0.0) for m in applicable) / n
        agg[f"all_evidence_at_{k}"] = sum(m.all_evidence_at_k.get(k, 0.0) for m in applicable) / n
        agg[f"ndcg_at_{k}"] = sum(m.ndcg_at_k.get(k, 0.0) for m in applicable) / n

    agg["mrr"] = sum(m.mrr for m in applicable) / n
    return agg


__all__ = [
    "RetrievalCaseMetrics",
    "aggregate_retrieval_metrics",
    "evaluate_retrieval",
    "matches_locator",
    "normalize_text",
]
