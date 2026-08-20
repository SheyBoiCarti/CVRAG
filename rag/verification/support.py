"""Semantic claim support verification protocol, fakes, and aggregation logic."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from rag.models import (
    Citation,
    Claim,
    ClaimVerification,
    Evidence,
    EvidenceSupportStatus,
)
from rag.settings import Settings, get_settings
from rag.vector_store import retry_transient
from rag.verification.models import SupportVerdict

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class SupportVerifier(Protocol):
    """Protocol for semantic claim entailment verification against retrieved evidence."""

    def verify(
        self, claim: Claim, evidence: Sequence[Evidence]
    ) -> SupportVerdict:
        """Verifies if the cited evidence collectively entails the factual claim."""
        ...


class FakeSupportVerifier:
    """Deterministic in-memory support verifier for tests and offline evaluation."""

    def __init__(
        self,
        default_supported: bool = True,
        rule_fn: Callable[[Claim, Sequence[Evidence]], bool] | None = None,
        verdicts_by_claim_id: Mapping[str, SupportVerdict | bool] | None = None,
        rejection_reason: str = "unsupported_claim",
        verifier_model: str = "fake-verifier",
        verifier_version: str = "1.0",
        error: Exception | None = None,
    ) -> None:
        self.default_supported = default_supported
        self.rule_fn = rule_fn
        self.verdicts_by_claim_id = dict(verdicts_by_claim_id or {})
        self.rejection_reason = rejection_reason
        self.verifier_model = verifier_model
        self.verifier_version = verifier_version
        self.error = error
        self.calls: list[tuple[Claim, tuple[Evidence, ...]]] = []

    def verify(self, claim: Claim, evidence: Sequence[Evidence]) -> SupportVerdict:
        self.calls.append((claim, tuple(evidence)))
        if self.error is not None:
            raise self.error

        if claim.claim_id in self.verdicts_by_claim_id:
            custom = self.verdicts_by_claim_id[claim.claim_id]
            if isinstance(custom, SupportVerdict):
                return custom
            supported = bool(custom)
            return SupportVerdict(
                claim_id=claim.claim_id,
                combined_supported=supported,
                supporting_evidence_ids=tuple(ev.evidence_id for ev in evidence) if supported else (),
                reason_code="" if supported else self.rejection_reason,
                verifier_model=self.verifier_model,
                verifier_version=self.verifier_version,
            )

        if self.rule_fn is not None:
            supported = bool(self.rule_fn(claim, evidence))
            return SupportVerdict(
                claim_id=claim.claim_id,
                combined_supported=supported,
                supporting_evidence_ids=tuple(ev.evidence_id for ev in evidence) if supported else (),
                reason_code="" if supported else self.rejection_reason,
                verifier_model=self.verifier_model,
                verifier_version=self.verifier_version,
            )

        supported = self.default_supported
        return SupportVerdict(
            claim_id=claim.claim_id,
            combined_supported=supported,
            supporting_evidence_ids=tuple(ev.evidence_id for ev in evidence) if supported else (),
            reason_code="" if supported else self.rejection_reason,
            verifier_model=self.verifier_model,
            verifier_version=self.verifier_version,
        )


class NvidiaSupportVerifier:
    """Production support verifier executing structured verification via NVIDIA LLM endpoint."""

    _SYSTEM_PROMPT = (
        "You are a strict, objective factual entailment verifier.\n"
        "Input contains a factual claim and untrusted CV evidence chunks.\n"
        "Determine if the supplied evidence collectively and directly entails the factual claim.\n"
        "Do not assume or extrapolate beyond the provided text.\n"
        "Return ONLY a valid JSON object with exact keys:\n"
        '{"supported": true|false, "supporting_evidence_ids": ["S1", ...], "reason_code": "supported"|"unsupported_claim"}\n'
        "No chain-of-thought, no Markdown formatting, no explanation."
    )

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or ChatNVIDIA(
            model=self.settings.LLM_MODEL,
            api_key=self.settings.NVIDIA_API_KEY,
            temperature=0.0,
            max_completion_tokens=200,
            timeout=self.settings.REQUEST_TIMEOUT,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._SYSTEM_PROMPT),
                (
                    "human",
                    "Claim:\n{claim_text}\n\n"
                    "Cited Evidence:\n{evidence_context}\n\n"
                    "JSON Verdict:",
                ),
            ]
        )

    def _format_evidence_context(self, evidence: Sequence[Evidence]) -> str:
        blocks = []
        for ev in evidence:
            blocks.append(
                f'<evidence id="{ev.evidence_id}" source="{ev.source}">\n'
                f"{ev.content}\n"
                f"</evidence>"
            )
        return "\n\n".join(blocks)

    def verify(self, claim: Claim, evidence: Sequence[Evidence]) -> SupportVerdict:
        if not evidence:
            return SupportVerdict(
                claim_id=claim.claim_id,
                combined_supported=False,
                supporting_evidence_ids=(),
                reason_code="unsupported_claim",
                verifier_model=str(getattr(self.settings, "LLM_MODEL", "nvidia-llm")),
                verifier_version="1.0",
            )

        valid_evidence_ids = {ev.evidence_id for ev in evidence}
        evidence_context = self._format_evidence_context(evidence)
        chain = self.prompt | self.llm | StrOutputParser()

        def _invoke() -> str:
            return chain.invoke(
                {
                    "claim_text": claim.text,
                    "evidence_context": evidence_context,
                }
            )

        try:
            raw_output = retry_transient(
                _invoke,
                max_retries=self.settings.MAX_RETRIES,
                operation_name="semantic_claim_verification",
                model=str(getattr(self.settings, "LLM_MODEL", "nvidia-llm")),
            )
        except Exception as exc:
            LOGGER.warning("Semantic support verifier endpoint failed (type=%s)", type(exc).__name__)
            raise

        try:
            # Extract JSON block even if model wraps in markdown fence
            json_text = raw_output.strip()
            if "```" in json_text:
                match = re.search(r"\{.*\}", json_text, re.DOTALL)
                if match:
                    json_text = match.group(0)
            data = json.loads(json_text)
            supported = bool(data.get("supported", False))
            raw_supporting_ids = data.get("supporting_evidence_ids", [])
            valid_supporting_ids = tuple(
                str(sid) for sid in raw_supporting_ids if str(sid) in valid_evidence_ids
            )
            reason_code = str(data.get("reason_code", "supported" if supported else "unsupported_claim"))

            return SupportVerdict(
                claim_id=claim.claim_id,
                combined_supported=supported,
                supporting_evidence_ids=valid_supporting_ids if supported else (),
                reason_code="" if supported else reason_code,
                verifier_model=str(getattr(self.settings, "LLM_MODEL", "nvidia-llm")),
                verifier_version="1.0",
            )
        except Exception as exc:
            LOGGER.warning(
                "Semantic verifier returned malformed output (type=%s, output=%r)",
                type(exc).__name__,
                raw_output[:200] if "raw_output" in locals() else "",
            )
            return SupportVerdict(
                claim_id=claim.claim_id,
                combined_supported=False,
                supporting_evidence_ids=(),
                reason_code="verifier_malformed_output",
                verifier_model=str(getattr(self.settings, "LLM_MODEL", "nvidia-llm")),
                verifier_version="1.0",
            )


def aggregate_citation_support_status(
    evidence_id: str,
    citing_verdicts: Sequence[tuple[ClaimVerification, SupportVerdict | None]],
) -> EvidenceSupportStatus:
    """Aggregates public citation support status with precedence: unsupported > not_checked > supported."""
    relevant_verdicts = [
        (cv, verdict)
        for cv, verdict in citing_verdicts
        if evidence_id in cv.citation_ids
    ]
    if not relevant_verdicts:
        return "not_checked"

    has_unsupported = False
    has_not_checked = False
    all_supported = True

    for cv, verdict in relevant_verdicts:
        if cv.status == "unsupported":
            has_unsupported = True
            all_supported = False
        elif cv.status in ("not_checked", "uncited", "invalid_citation"):
            has_not_checked = True
            all_supported = False
        elif cv.status == "supported":
            if verdict is not None and evidence_id not in verdict.supporting_evidence_ids:
                has_unsupported = True
                all_supported = False
        else:
            has_not_checked = True
            all_supported = False

    if has_unsupported:
        return "unsupported"
    if has_not_checked or not all_supported:
        return "not_checked"
    return "supported"


def verify_claims_support(
    claims: Sequence[Claim],
    verifications: Sequence[ClaimVerification],
    evidence: Sequence[Evidence],
    citations: Sequence[Citation],
    verifier: SupportVerifier,
) -> tuple[tuple[ClaimVerification, ...], tuple[Citation, ...]]:
    """Executes semantic support verification on structurally valid claims and aggregates citation statuses."""
    evidence_map = {ev.evidence_id: ev for ev in evidence}
    updated_verifications: list[ClaimVerification] = []
    verdicts_by_claim_id: dict[str, SupportVerdict | None] = {}

    for claim, cv in zip(claims, verifications):
        # Only verify claims that are factual and passed structural checks (status="not_checked" with citations)
        if cv.status == "not_checked" and claim.factual and claim.citation_ids:
            cited_ev = [evidence_map[cid] for cid in claim.citation_ids if cid in evidence_map]
            verdict = verifier.verify(claim, cited_ev)
            verdicts_by_claim_id[claim.claim_id] = verdict

            if verdict.combined_supported:
                updated_verifications.append(
                    ClaimVerification(
                        claim_id=claim.claim_id,
                        text=claim.text,
                        factual=True,
                        importance=claim.importance,
                        citation_ids=claim.citation_ids,
                        status="supported",
                        reason_code="",
                    )
                )
            else:
                updated_verifications.append(
                    ClaimVerification(
                        claim_id=claim.claim_id,
                        text=claim.text,
                        factual=True,
                        importance=claim.importance,
                        citation_ids=claim.citation_ids,
                        status="unsupported",
                        reason_code=verdict.reason_code or "unsupported_claim",
                    )
                )
        else:
            verdicts_by_claim_id[claim.claim_id] = None
            updated_verifications.append(cv)

    # Aggregate citation support statuses
    citing_pairs = list(zip(updated_verifications, [verdicts_by_claim_id.get(cv.claim_id) for cv in updated_verifications]))
    updated_citations: list[Citation] = []
    for cit in citations:
        agg_status = aggregate_citation_support_status(cit.evidence_id, citing_pairs)
        updated_citations.append(
            Citation(
                evidence_id=cit.evidence_id,
                chunk_id=cit.chunk_id,
                source=cit.source,
                page=cit.page,
                excerpt=cit.excerpt,
                support_status=agg_status,
            )
        )

    return tuple(updated_verifications), tuple(updated_citations)


__all__ = [
    "FakeSupportVerifier",
    "NvidiaSupportVerifier",
    "SupportVerifier",
    "aggregate_citation_support_status",
    "verify_claims_support",
]
