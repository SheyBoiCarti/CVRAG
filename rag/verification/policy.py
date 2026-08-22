"""Post-generation verification, single-shot corrective regeneration, and abstention policy.

Architecture & Verification State Machine:
-----------------------------------------
1. Structural Verification:
   - Scans generated answer for citation brackets `[S1]`, `[S2]`.
   - Ensures cited evidence IDs exist in the retrieved context pool.
2. Semantic Support Verification (NLI):
   - Pairs each segmented claim sentence with its cited evidence texts.
   - Evaluates natural language inference entailment scores against `SUPPORT_THRESHOLD`.
3. Single-Shot Corrective Rewrite:
   - If unsupported claims or structural invalidities are found, formats the failures into `CORRECTION_PROMPT`.
   - Re-invokes LLM once to prune or correct ungrounded statements.
4. Deterministic Abstention:
   - If the revised response still contains unsupported claims, the system abstains cleanly with
     `STANDARD_ABSTENTION_TEXT` rather than risking hallucination.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from rag.models import AnswerResult, Evidence
from rag.settings import Settings, get_settings
from rag.verification.citation_parser import parse_claims
from rag.verification.structural import (
    build_initial_citations,
    filter_cited_evidence,
    verify_structural,
)
from rag.verification.support import (
    SupportVerifier,
    verify_claims_support,
)

LOGGER = logging.getLogger(__name__)

STANDARD_ABSTENTION_TEXT = (
    "I don't have enough supported information in the CV to answer that confidently."
)


def _format_evidence_context(evidence: Sequence[Evidence]) -> str:
    """Formats evidence items into structured XML blocks for corrective prompt synthesis."""
    blocks = []
    for ev in evidence:
        blocks.append(
            f'<evidence id="{ev.evidence_id}" source="{ev.source}" page="{ev.page}">\n'
            f"{ev.content}\n"
            f"</evidence>"
        )
    return "\n\n".join(blocks)


def execute_post_generation_policy(
    raw_answer: str,
    evidence: Sequence[Evidence],
    verifier: SupportVerifier,
    settings: Settings | None = None,
    generator_fn: Callable[[dict[str, Any]], str] | None = None,
    degraded_reason_codes: Sequence[str] = (),
    retrieval_query: str = "",
    question: str = "",
    history_messages: Sequence[tuple[str, str]] = (),
) -> tuple[AnswerResult, bool]:
    """Executes post-generation claim verification, single corrective regeneration, and deterministic decisions.

    Returns:
        tuple[AnswerResult, bool]: (final_answer_result, replace_streamed_answer).
        `replace_streamed_answer` is True if a corrective pass altered the text, signaling the frontend to replace
        the live-streamed draft.
    """
    effective_settings = settings or get_settings()

    # Pass 1: Parse and structurally verify initial raw answer
    claims = parse_claims(raw_answer)
    verifications = verify_structural(claims, evidence)
    cited_ev = filter_cited_evidence(claims, evidence)
    initial_citations = build_initial_citations(cited_ev)

    # Semantic support verification
    try:
        verified_claims, verified_cits = verify_claims_support(
            claims=claims,
            verifications=verifications,
            evidence=cited_ev,
            citations=initial_citations,
            verifier=verifier,
        )
    except Exception as exc:
        LOGGER.warning(
            "Semantic support verifier failed during post-generation policy (type=%s)",
            type(exc).__name__,
        )
        if effective_settings.VERIFIER_FAIL_CLOSED:
            return (
                AnswerResult(
                    answer=STANDARD_ABSTENTION_TEXT,
                    citations=(),
                    claim_verifications=verifications,
                    decision="abstained",
                    reason_codes=("verification_unavailable",),
                    retrieval_query=retrieval_query,
                    pipeline_version=effective_settings.PIPELINE_VERSION,
                ),
                True,
            )
        else:
            return (
                AnswerResult(
                    answer=raw_answer,
                    citations=initial_citations,
                    claim_verifications=verifications,
                    decision="degraded",
                    reason_codes=("verification_unavailable",),
                    retrieval_query=retrieval_query,
                    pipeline_version=effective_settings.PIPELINE_VERSION,
                ),
                False,
            )

    failed_core_claims = [
        cv
        for cv in verified_claims
        if cv.factual and cv.importance == "core" and cv.status != "supported"
    ]

    # Outcome 1: All core factual claims supported
    if not failed_core_claims and verified_claims:
        decision = "degraded" if degraded_reason_codes else "answered"
        return (
            AnswerResult(
                answer=raw_answer,
                citations=verified_cits,
                claim_verifications=verified_claims,
                decision=decision,
                reason_codes=tuple(degraded_reason_codes),
                retrieval_query=retrieval_query,
                pipeline_version=effective_settings.PIPELINE_VERSION,
            ),
            False,
        )

    # If no claims extracted or nonfactual only (e.g. initial answer was empty or nonfactual)
    if not verified_claims:
        return (
            AnswerResult(
                answer=raw_answer,
                citations=(),
                claim_verifications=(),
                decision="degraded" if degraded_reason_codes else "answered",
                reason_codes=tuple(degraded_reason_codes),
                retrieval_query=retrieval_query,
                pipeline_version=effective_settings.PIPELINE_VERSION,
            ),
            False,
        )

    # Outcome 2: Core claims failed. Attempt at most ONE corrective regeneration
    if generator_fn is not None and effective_settings.MAX_CORRECTION_ATTEMPTS > 0:
        failed_summary_lines = []
        for cv in failed_core_claims:
            reason = cv.reason_code or cv.status
            failed_summary_lines.append(f"- Claim {cv.claim_id}: '{cv.text}' (issue: {reason})")
        failed_claims_summary = "\n".join(failed_summary_lines)

        correction_payload = {
            "failed_claims_summary": failed_claims_summary,
            "context": _format_evidence_context(evidence),
            "question": question or "Answer the user question accurately grounded in the CV.",
            "original_answer": raw_answer,
            "chat_history": list(history_messages),
        }

        try:
            revised_answer = generator_fn(correction_payload).strip()
            # Pass 2: Parse and verify revised answer from scratch
            rev_claims = parse_claims(revised_answer)
            rev_verifications = verify_structural(rev_claims, evidence)
            rev_cited_ev = filter_cited_evidence(rev_claims, evidence)
            rev_citations = build_initial_citations(rev_cited_ev)

            rev_verified_claims, rev_verified_cits = verify_claims_support(
                claims=rev_claims,
                verifications=rev_verifications,
                evidence=rev_cited_ev,
                citations=rev_citations,
                verifier=verifier,
            )

            rev_failed_core = [
                cv
                for cv in rev_verified_claims
                if cv.factual and cv.importance == "core" and cv.status != "supported"
            ]

            if not rev_failed_core and rev_verified_claims:
                # Corrected answer passes verification!
                decision = "degraded" if degraded_reason_codes else "answered"
                return (
                    AnswerResult(
                        answer=revised_answer,
                        citations=rev_verified_cits,
                        claim_verifications=rev_verified_claims,
                        decision=decision,
                        reason_codes=tuple(degraded_reason_codes),
                        retrieval_query=retrieval_query,
                        pipeline_version=effective_settings.PIPELINE_VERSION,
                    ),
                    True,
                )
        except Exception as exc:
            LOGGER.warning(
                "Corrective regeneration attempt failed (type=%s)",
                type(exc).__name__,
            )

    # Outcome 3: Corrective regeneration failed or second pass still contained failed core claims -> STRICT ABSTENTION
    failure_reasons = []
    for cv in (rev_failed_core if "rev_failed_core" in locals() and rev_failed_core else failed_core_claims):
        reason = cv.reason_code or cv.status
        if reason not in failure_reasons:
            failure_reasons.append(reason)
    if not failure_reasons:
        failure_reasons.append("unsupported_claim")

    all_reasons = list(degraded_reason_codes) + [r for r in failure_reasons if r not in degraded_reason_codes]

    return (
        AnswerResult(
            answer=STANDARD_ABSTENTION_TEXT,
            citations=(),
            claim_verifications=tuple(rev_verified_claims if "rev_verified_claims" in locals() else verified_claims),
            decision="abstained",
            reason_codes=tuple(all_reasons),
            retrieval_query=retrieval_query,
            pipeline_version=effective_settings.PIPELINE_VERSION,
        ),
        True,
    )


__all__ = [
    "STANDARD_ABSTENTION_TEXT",
    "execute_post_generation_policy",
]
