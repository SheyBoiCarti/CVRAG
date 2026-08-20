from __future__ import annotations

from rag.models import Claim, Evidence
from rag.verification.structural import (
    build_initial_citations,
    filter_cited_evidence,
    verify_structural,
)


def _make_sample_evidence() -> list[Evidence]:
    return [
        Evidence(
            evidence_id="S1",
            chunk_id="chunk-1",
            content="Sheharyar worked at Wirk.",
            source="Custom_CV.md",
            page=1,
        ),
        Evidence(
            evidence_id="S2",
            chunk_id="chunk-2",
            content="He graduated Cum Laude from NMU.",
            source="Custom_CV.md",
            page=1,
        ),
        Evidence(
            evidence_id="S3",
            chunk_id="chunk-3",
            content="Unused third chunk about tools.",
            source="Custom_CV.md",
            page=1,
        ),
    ]


def test_structural_verification_valid_citations():
    evidence = _make_sample_evidence()
    claims = [
        Claim(claim_id="C1", text="Worked at Wirk.", citation_ids=("S1",), factual=True),
        Claim(claim_id="C2", text="Graduated Cum Laude.", citation_ids=("S2",), factual=True),
    ]

    verifications = verify_structural(claims, evidence)
    assert len(verifications) == 2
    assert verifications[0].status == "not_checked"
    assert verifications[0].reason_code == ""
    assert verifications[1].status == "not_checked"
    assert verifications[1].reason_code == ""


def test_structural_verification_uncited_claim():
    evidence = _make_sample_evidence()
    claims = [
        Claim(claim_id="C1", text="Worked at Wirk.", citation_ids=(), factual=True),
    ]

    verifications = verify_structural(claims, evidence)
    assert len(verifications) == 1
    assert verifications[0].status == "uncited"
    assert verifications[0].reason_code == "uncited_claim"


def test_structural_verification_invalid_citation_tag():
    evidence = _make_sample_evidence()
    claims = [
        Claim(claim_id="C1", text="Worked at Wirk.", citation_ids=("S99",), factual=True),
    ]

    verifications = verify_structural(claims, evidence)
    assert len(verifications) == 1
    assert verifications[0].status == "invalid_citation"
    assert verifications[0].reason_code == "invalid_citation"


def test_structural_verification_nonfactual_claim():
    evidence = _make_sample_evidence()
    claims = [
        Claim(claim_id="C1", text="What did he do?", citation_ids=(), factual=False),
    ]

    verifications = verify_structural(claims, evidence)
    assert len(verifications) == 1
    assert verifications[0].status == "not_checked"
    assert verifications[0].reason_code == ""


def test_filter_cited_evidence_and_build_citations():
    evidence = _make_sample_evidence()
    claims = [
        Claim(claim_id="C1", text="Worked at Wirk.", citation_ids=("S1",), factual=True),
        # S2 is not cited, S3 is not cited
    ]

    cited_ev = filter_cited_evidence(claims, evidence)
    assert len(cited_ev) == 1
    assert cited_ev[0].evidence_id == "S1"

    citations = build_initial_citations(cited_ev)
    assert len(citations) == 1
    assert citations[0].evidence_id == "S1"
    assert citations[0].chunk_id == "chunk-1"
    assert citations[0].source == "Custom_CV.md"
    assert citations[0].page == 1
    assert citations[0].support_status == "not_checked"
