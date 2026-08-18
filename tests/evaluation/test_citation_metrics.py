import pytest
from evaluation.schema import GoldenCase
from evaluation.metrics.citations import (
    evaluate_citations,
    aggregate_citation_metrics,
    CitationCaseMetrics,
)
from rag.models import AnswerResult, Citation, ClaimVerification
from rag.retrieval.base import ChunkRecord, RankedChunk


def test_citation_metrics_validity_and_coverage():
    case = GoldenCase.from_dict({
        "id": "c1",
        "category": "straightforward",
        "question": "q",
        "history": [],
        "answerable": True,
        "reference_answer": "ans",
        "required_facts": [],
        "forbidden_facts": [],
        "evidence_groups": [],
    })
    retrieved = [
        RankedChunk(ChunkRecord("c1", "Content 1", "Custom_CV.md", 1, 0), 1, 0.9, ("dense",)),
        RankedChunk(ChunkRecord("c2", "Content 2", "Custom_CV.md", 1, 1), 2, 0.8, ("dense",)),
    ]

    # Valid citations
    citations = (
        Citation(evidence_id="c1", chunk_id="c1", source="Custom_CV.md", page=1, excerpt="Content 1"),
    )
    claims = (
        ClaimVerification(claim_id="cl1", text="Claim 1", factual=True, citation_ids=("c1",), status="supported"),
    )
    ans = AnswerResult("Answer [S1]", citations=citations, claim_verifications=claims)

    res = evaluate_citations(case, ans, retrieved)
    assert res.citation_id_validity == 1.0
    assert res.claim_citation_coverage == 1.0
    assert res.locator_accuracy == 1.0
    assert res.supported_claim_ratio == 1.0


def test_citation_metrics_with_invalid_citation_id():
    case = GoldenCase.from_dict({
        "id": "c1",
        "category": "straightforward",
        "question": "q",
        "history": [],
        "answerable": True,
        "reference_answer": "ans",
        "required_facts": [],
        "forbidden_facts": [],
        "evidence_groups": [],
    })
    retrieved = [
        RankedChunk(ChunkRecord("c1", "Content 1", "Custom_CV.md", 1, 0), 1, 0.9, ("dense",)),
    ]

    # Invalid citation referencing non-existent chunk c999
    citations = (
        Citation(evidence_id="c999", chunk_id="c999", source="Custom_CV.md", page=1, excerpt="Fake"),
    )
    claims = (
        ClaimVerification(claim_id="cl1", text="Claim 1", factual=True, citation_ids=("c999",), status="invalid_citation"),
    )
    ans = AnswerResult("Answer [S999]", citations=citations, claim_verifications=claims)

    res = evaluate_citations(case, ans, retrieved)
    assert res.citation_id_validity == 0.0
    assert res.claim_citation_coverage == 1.0
    assert res.locator_accuracy == 0.0
    assert res.supported_claim_ratio == 0.0


def test_aggregate_citation_metrics():
    case = GoldenCase.from_dict({
        "id": "c1", "category": "straightforward", "question": "q", "history": [], "answerable": True,
        "reference_answer": "ans", "required_facts": [], "forbidden_facts": [], "evidence_groups": [],
    })
    m1 = CitationCaseMetrics(
        case_id="c1",
        is_applicable=True,
        total_citations=2,
        valid_citations=2,
        total_factual_claims=2,
        cited_factual_claims=2,
        accurate_locators=2,
        supported_claims=2,
        citation_id_validity=1.0,
        claim_citation_coverage=1.0,
        locator_accuracy=1.0,
        supported_claim_ratio=1.0,
    )
    m2 = CitationCaseMetrics(
        case_id="c2",
        is_applicable=True,
        total_citations=2,
        valid_citations=1,
        total_factual_claims=2,
        cited_factual_claims=1,
        accurate_locators=1,
        supported_claims=1,
        citation_id_validity=0.5,
        claim_citation_coverage=0.5,
        locator_accuracy=0.5,
        supported_claim_ratio=0.5,
    )

    agg = aggregate_citation_metrics([m1, m2])
    assert agg["citation_id_validity"] == pytest.approx(0.75)
    assert agg["claim_citation_coverage"] == pytest.approx(0.75)
    assert agg["locator_accuracy"] == pytest.approx(0.75)
    assert agg["supported_claim_ratio"] == pytest.approx(0.75)
