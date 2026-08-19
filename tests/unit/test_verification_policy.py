from __future__ import annotations

from typing import Any
from rag.models import AnswerResult, Citation, Claim, Evidence
from rag.settings import Settings
from rag.verification.policy import (
    STANDARD_ABSTENTION_TEXT,
    execute_post_generation_policy,
)
from rag.verification.support import FakeSupportVerifier


def _sample_evidence() -> list[Evidence]:
    return [
        Evidence(
            evidence_id="S1",
            chunk_id="chunk-1",
            content="Sheharyar worked as a Software Engineer at Wirk.",
            source="Custom_CV.md",
            page=1,
        ),
        Evidence(
            evidence_id="S2",
            chunk_id="chunk-2",
            content="He graduated with Honours in Computer Science Cum Laude.",
            source="Custom_CV.md",
            page=1,
        ),
    ]


def test_post_generation_all_supported_returns_answered():
    evidence = _sample_evidence()
    raw_answer = "Sheharyar worked at Wirk [S1]. He graduated Cum Laude [S2]."
    verifier = FakeSupportVerifier(default_supported=True)
    settings = Settings(NVIDIA_API_KEY="test-key")

    result, replaced = execute_post_generation_policy(
        raw_answer=raw_answer,
        evidence=evidence,
        verifier=verifier,
        settings=settings,
        generator_fn=lambda prompt_payload: "should not be called",
    )

    assert result.decision == "answered"
    assert result.answer == raw_answer
    assert replaced is False
    assert len(result.citations) == 2
    assert len(result.claim_verifications) == 2
    for cv in result.claim_verifications:
        assert cv.status == "supported"


def test_post_generation_corrective_regeneration_success():
    evidence = _sample_evidence()
    raw_answer = "Sheharyar worked at Google [S1]."  # "Google" will be rejected by verifier
    corrected_answer = "Sheharyar worked at Wirk [S1]."

    # Reject claims containing "Google"
    verifier = FakeSupportVerifier(
        rule_fn=lambda c, ev: "Google" not in c.text,
        rejection_reason="unsupported_claim",
    )
    settings = Settings(NVIDIA_API_KEY="test-key")

    generation_calls = []

    def mock_generator(payload: dict[str, Any]) -> str:
        generation_calls.append(payload)
        return corrected_answer

    result, replaced = execute_post_generation_policy(
        raw_answer=raw_answer,
        evidence=evidence,
        verifier=verifier,
        settings=settings,
        generator_fn=mock_generator,
    )

    assert len(generation_calls) == 1
    assert result.decision == "answered"
    assert result.answer == corrected_answer
    assert replaced is True
    assert len(result.claim_verifications) == 1
    assert result.claim_verifications[0].status == "supported"


def test_post_generation_correction_fails_again_abstains():
    evidence = _sample_evidence()
    raw_answer = "Sheharyar worked at Google [S1]."
    still_bad_answer = "He was CEO at Apple [S1]."

    # Reject both Google and Apple
    verifier = FakeSupportVerifier(
        rule_fn=lambda c, ev: "Google" not in c.text and "Apple" not in c.text,
        rejection_reason="unsupported_claim",
    )
    settings = Settings(NVIDIA_API_KEY="test-key")

    generation_calls = []

    def mock_generator(payload: dict[str, Any]) -> str:
        generation_calls.append(payload)
        return still_bad_answer

    result, replaced = execute_post_generation_policy(
        raw_answer=raw_answer,
        evidence=evidence,
        verifier=verifier,
        settings=settings,
        generator_fn=mock_generator,
    )

    assert len(generation_calls) == 1  # At most ONE corrective attempt
    assert result.decision == "abstained"
    assert result.answer == STANDARD_ABSTENTION_TEXT
    assert replaced is True
    assert "unsupported_claim" in result.reason_codes


def test_post_generation_reranker_degraded_reason():
    evidence = _sample_evidence()
    raw_answer = "Sheharyar worked at Wirk [S1]."
    verifier = FakeSupportVerifier(default_supported=True)
    settings = Settings(NVIDIA_API_KEY="test-key")

    result, replaced = execute_post_generation_policy(
        raw_answer=raw_answer,
        evidence=evidence,
        verifier=verifier,
        settings=settings,
        generator_fn=lambda p: "",
        degraded_reason_codes=("reranker_unavailable",),
    )

    assert result.decision == "degraded"
    assert "reranker_unavailable" in result.reason_codes
    assert result.answer == raw_answer


def test_post_generation_verifier_unavailable_fail_closed():
    evidence = _sample_evidence()
    raw_answer = "Sheharyar worked at Wirk [S1]."
    verifier = FakeSupportVerifier(error=RuntimeError("Endpoint 503 unavailable"))
    settings = Settings(NVIDIA_API_KEY="test-key", VERIFIER_FAIL_CLOSED=True)

    result, replaced = execute_post_generation_policy(
        raw_answer=raw_answer,
        evidence=evidence,
        verifier=verifier,
        settings=settings,
        generator_fn=lambda p: "",
    )

    assert result.decision == "abstained"
    assert result.answer == STANDARD_ABSTENTION_TEXT
    assert "verification_unavailable" in result.reason_codes
    assert replaced is True


def test_post_generation_verifier_unavailable_fail_open():
    evidence = _sample_evidence()
    raw_answer = "Sheharyar worked at Wirk [S1]."
    verifier = FakeSupportVerifier(error=RuntimeError("Endpoint 503 unavailable"))
    settings = Settings(NVIDIA_API_KEY="test-key", VERIFIER_FAIL_CLOSED=False)

    result, replaced = execute_post_generation_policy(
        raw_answer=raw_answer,
        evidence=evidence,
        verifier=verifier,
        settings=settings,
        generator_fn=lambda p: "",
    )

    assert result.decision == "degraded"
    assert "verification_unavailable" in result.reason_codes
    assert result.answer == raw_answer
    assert replaced is False
