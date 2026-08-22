from __future__ import annotations

from rag.models import Citation, Claim, ClaimVerification, Evidence
from rag.verification.support import (
    FakeSupportVerifier,
    verify_claims_support,
)


def _sample_evidence() -> list[Evidence]:
    return [
        Evidence(
            evidence_id="S1",
            chunk_id="c1",
            content="Sheharyar worked as a Software Engineer at Wirk.",
            source="Custom_CV.md",
            page=1,
        ),
        Evidence(
            evidence_id="S2",
            chunk_id="c2",
            content="He graduated Cum Laude from NMU in 2022.",
            source="Custom_CV.md",
            page=1,
        ),
    ]


def test_fake_support_verifier_supported():
    evidence = _sample_evidence()
    claim = Claim(
        claim_id="C1",
        text="Sheharyar worked at Wirk.",
        citation_ids=("S1",),
        factual=True,
    )
    verifier = FakeSupportVerifier(default_supported=True)
    verdict = verifier.verify(claim, evidence[:1])
    assert verdict.combined_supported is True
    assert verdict.supporting_evidence_ids == ("S1",)


def test_fake_support_verifier_unsupported_by_rule():
    evidence = _sample_evidence()
    claim = Claim(
        claim_id="C1",
        text="Sheharyar worked at Google.",
        citation_ids=("S1",),
        factual=True,
    )
    # Reject claims containing "Google"
    verifier = FakeSupportVerifier(
        rule_fn=lambda c, ev: "Google" not in c.text,
        rejection_reason="unsupported_claim",
    )
    verdict = verifier.verify(claim, evidence[:1])
    assert verdict.combined_supported is False
    assert verdict.reason_code == "unsupported_claim"


def test_verify_claims_support_lifecycle():
    evidence = _sample_evidence()
    claims = [
        Claim(claim_id="C1", text="Worked at Wirk.", citation_ids=("S1",), factual=True),
        Claim(claim_id="C2", text="Worked at Google.", citation_ids=("S2",), factual=True),
        Claim(claim_id="C3", text="Uncited fact.", citation_ids=(), factual=True),
    ]
    verifications = [
        ClaimVerification(claim_id="C1", text="Worked at Wirk.", factual=True, citation_ids=("S1",), status="not_checked"),
        ClaimVerification(claim_id="C2", text="Worked at Google.", factual=True, citation_ids=("S2",), status="not_checked"),
        ClaimVerification(claim_id="C3", text="Uncited fact.", factual=True, citation_ids=(), status="uncited", reason_code="uncited_claim"),
    ]
    citations = [
        Citation(evidence_id="S1", chunk_id="c1", source="Custom_CV.md", page=1, excerpt="Worked at Wirk", support_status="not_checked"),
        Citation(evidence_id="S2", chunk_id="c2", source="Custom_CV.md", page=1, excerpt="NMU grad", support_status="not_checked"),
    ]

    verifier = FakeSupportVerifier(
        rule_fn=lambda c, ev: "Google" not in c.text,
        rejection_reason="unsupported_claim",
    )

    verified_claims, verified_cits = verify_claims_support(
        claims=claims,
        verifications=verifications,
        evidence=evidence,
        citations=citations,
        verifier=verifier,
    )

    assert len(verified_claims) == 3
    assert verified_claims[0].status == "supported"
    assert verified_claims[1].status == "unsupported"
    assert verified_claims[1].reason_code == "unsupported_claim"
    assert verified_claims[2].status == "uncited"

    assert len(verified_cits) == 2
    assert verified_cits[0].support_status == "supported"
    assert verified_cits[1].support_status == "unsupported"


def test_citation_support_aggregation_precedence():
    # If S1 is cited by C1 (supported) and C2 (unsupported), S1 becomes 'unsupported'
    evidence = _sample_evidence()
    claims = [
        Claim(claim_id="C1", text="Worked at Wirk.", citation_ids=("S1",), factual=True),
        Claim(claim_id="C2", text="Was CEO at Wirk.", citation_ids=("S1",), factual=True),
    ]
    verifications = [
        ClaimVerification(claim_id="C1", text="Worked at Wirk.", factual=True, citation_ids=("S1",), status="not_checked"),
        ClaimVerification(claim_id="C2", text="Was CEO at Wirk.", factual=True, citation_ids=("S1",), status="not_checked"),
    ]
    citations = [
        Citation(evidence_id="S1", chunk_id="c1", source="Custom_CV.md", page=1, excerpt="Worked at Wirk", support_status="not_checked"),
    ]

    verifier = FakeSupportVerifier(
        rule_fn=lambda c, ev: "CEO" not in c.text,
        rejection_reason="unsupported_claim",
    )

    verified_claims, verified_cits = verify_claims_support(
        claims=claims,
        verifications=verifications,
        evidence=evidence,
        citations=citations,
        verifier=verifier,
    )

    assert verified_claims[0].status == "supported"
    assert verified_claims[1].status == "unsupported"
    # S1 aggregated status must be 'unsupported' because C2 rejected it
    assert verified_cits[0].support_status == "unsupported"
