"""Deterministic citation metrics: ID validity, coverage, locator accuracy, claim support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evaluation.schema import GoldenCase
from rag.models import AnswerResult
from rag.retrieval.base import RankedChunk


@dataclass(frozen=True)
class CitationCaseMetrics:
    """Citation metrics evaluated for an individual case."""

    case_id: str
    is_applicable: bool
    total_citations: int
    valid_citations: int
    total_factual_claims: int
    cited_factual_claims: int
    accurate_locators: int
    supported_claims: int
    citation_id_validity: float
    claim_citation_coverage: float
    locator_accuracy: float
    supported_claim_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "is_applicable": self.is_applicable,
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "total_factual_claims": self.total_factual_claims,
            "cited_factual_claims": self.cited_factual_claims,
            "accurate_locators": self.accurate_locators,
            "supported_claims": self.supported_claims,
            "citation_id_validity": self.citation_id_validity,
            "claim_citation_coverage": self.claim_citation_coverage,
            "locator_accuracy": self.locator_accuracy,
            "supported_claim_ratio": self.supported_claim_ratio,
        }


def evaluate_citations(
    case: GoldenCase,
    answer_result: AnswerResult,
    retrieved_evidence: Sequence[RankedChunk],
) -> CitationCaseMetrics:
    """Evaluates citation metrics for an AnswerResult and candidate retrieved chunks."""
    citations = answer_result.citations
    claims = answer_result.claim_verifications

    if not case.answerable or answer_result.decision == "abstained":
        return CitationCaseMetrics(
            case_id=case.id,
            is_applicable=False,
            total_citations=len(citations),
            valid_citations=0,
            total_factual_claims=len(claims),
            cited_factual_claims=0,
            accurate_locators=0,
            supported_claims=0,
            citation_id_validity=1.0 if not citations else 0.0,
            claim_citation_coverage=1.0 if not claims else 0.0,
            locator_accuracy=1.0 if not citations else 0.0,
            supported_claim_ratio=1.0 if not claims else 0.0,
        )

    retrieved_chunk_ids = {c.chunk.chunk_id for c in retrieved_evidence}
    chunk_map = {c.chunk.chunk_id: c.chunk for c in retrieved_evidence}

    # 1. Citation ID validity: citation evidence_id / chunk_id is in retrieved candidates
    total_citations = len(citations)
    valid_citations = 0
    accurate_locators = 0
    for cit in citations:
        c_id = cit.chunk_id or cit.evidence_id
        if c_id in retrieved_chunk_ids:
            valid_citations += 1
            retrieved_chunk = chunk_map[c_id]
            if (
                cit.source.strip().casefold() == retrieved_chunk.source.strip().casefold()
                and cit.page == retrieved_chunk.page
            ):
                accurate_locators += 1

    citation_id_validity = (valid_citations / total_citations) if total_citations > 0 else 1.0
    locator_accuracy = (accurate_locators / total_citations) if total_citations > 0 else 1.0

    # 2. Claim citation coverage and support status
    factual_claims = [cl for cl in claims if getattr(cl, "factual", True)]
    total_factual_claims = len(factual_claims)
    cited_factual_claims = sum(1 for cl in factual_claims if cl.citation_ids)
    supported_claims = sum(1 for cl in factual_claims if cl.status == "supported")

    claim_citation_coverage = (
        (cited_factual_claims / total_factual_claims) if total_factual_claims > 0 else 1.0
    )
    supported_claim_ratio = (
        (supported_claims / total_factual_claims) if total_factual_claims > 0 else 1.0
    )

    return CitationCaseMetrics(
        case_id=case.id,
        is_applicable=True,
        total_citations=total_citations,
        valid_citations=valid_citations,
        total_factual_claims=total_factual_claims,
        cited_factual_claims=cited_factual_claims,
        accurate_locators=accurate_locators,
        supported_claims=supported_claims,
        citation_id_validity=citation_id_validity,
        claim_citation_coverage=claim_citation_coverage,
        locator_accuracy=locator_accuracy,
        supported_claim_ratio=supported_claim_ratio,
    )


def aggregate_citation_metrics(
    metrics_list: Sequence[CitationCaseMetrics],
) -> dict[str, float]:
    """Computes macro averages for applicable citation metrics."""
    applicable = [m for m in metrics_list if m.is_applicable]
    if not applicable:
        return {"applicable_cases": 0}

    n = len(applicable)
    return {
        "applicable_cases": n,
        "citation_id_validity": sum(m.citation_id_validity for m in applicable) / n,
        "claim_citation_coverage": sum(m.claim_citation_coverage for m in applicable) / n,
        "locator_accuracy": sum(m.locator_accuracy for m in applicable) / n,
        "supported_claim_ratio": sum(m.supported_claim_ratio for m in applicable) / n,
    }


__all__ = [
    "CitationCaseMetrics",
    "evaluate_citations",
    "aggregate_citation_metrics",
]
