from __future__ import annotations

import dataclasses
from rag.models import (
    AnswerabilityDecision,
    AnswerabilitySignals,
    Claim,
    ClaimVerification,
)
from rag.verification.models import SupportVerdict


def test_claim_model_immutability_and_defaults():
    claim = Claim(
        claim_id="C1",
        text="Sheharyar worked at Wirk.",
        citation_ids=("S1",),
        factual=True,
        importance="core",
    )
    assert claim.claim_id == "C1"
    assert claim.text == "Sheharyar worked at Wirk."
    assert claim.citation_ids == ("S1",)
    assert claim.factual is True
    assert claim.importance == "core"

    try:
        claim.text = "modified"  # type: ignore
        assert False, "Should be frozen"
    except dataclasses.FrozenInstanceError:
        pass

    data = claim.to_dict()
    assert data["claim_id"] == "C1"
    assert data["citation_ids"] == ["S1"]
    assert data["factual"] is True
    assert data["importance"] == "core"


def test_claim_verification_model():
    cv = ClaimVerification(
        claim_id="C1",
        text="Sheharyar worked at Wirk.",
        factual=True,
        importance="core",
        status="supported",
        citation_ids=("S1",),
        reason_code="supported",
    )
    assert cv.claim_id == "C1"
    assert cv.status == "supported"
    assert cv.citation_ids == ("S1",)
    assert cv.reason_code == "supported"

    data = cv.to_dict()
    assert data["status"] == "supported"
    assert data["reason_code"] == "supported"


def test_answerability_signals_model():
    signals = AnswerabilitySignals(
        retrieved_count=3,
        dense_sparse_overlap=0.66,
        top_rrf_score=0.032,
        top_reranker_score=0.85,
        reranker_margin=0.25,
        retrieval_degraded=False,
    )
    assert signals.retrieved_count == 3
    assert signals.dense_sparse_overlap == 0.66
    assert signals.top_reranker_score == 0.85
    assert signals.retrieval_degraded is False

    data = signals.to_dict()
    assert data["retrieved_count"] == 3
    assert data["dense_sparse_overlap"] == 0.66


def test_answerability_signals_none_preservation():
    # Missing signals must remain None and not convert to 0
    signals = AnswerabilitySignals(
        retrieved_count=0,
        dense_sparse_overlap=None,
        top_rrf_score=None,
        top_reranker_score=None,
        reranker_margin=None,
        retrieval_degraded=False,
    )
    assert signals.dense_sparse_overlap is None
    assert signals.top_rrf_score is None
    assert signals.top_reranker_score is None
    assert signals.reranker_margin is None


def test_answerability_decision_model():
    decision = AnswerabilityDecision(
        decision="answer",
        reason_codes=(),
        policy_version="1.0",
    )
    assert decision.decision == "answer"
    assert decision.policy_version == "1.0"
    assert decision.to_dict()["decision"] == "answer"

    abstain_dec = AnswerabilityDecision(
        decision="abstain",
        reason_codes=("no_evidence",),
        policy_version="1.0",
    )
    assert abstain_dec.decision == "abstain"
    assert abstain_dec.reason_codes == ("no_evidence",)


def test_support_verdict_model():
    verdict = SupportVerdict(
        claim_id="C1",
        combined_supported=True,
        supporting_evidence_ids=("S1",),
        reason_code="",
        verifier_model="meta/llama-3.3-70b-instruct",
        verifier_version="1.0",
    )
    assert verdict.claim_id == "C1"
    assert verdict.combined_supported is True
    assert verdict.supporting_evidence_ids == ("S1",)
    assert verdict.verifier_model == "meta/llama-3.3-70b-instruct"
    assert verdict.to_dict()["combined_supported"] is True
