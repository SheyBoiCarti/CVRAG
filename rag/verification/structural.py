"""Deterministic structural citation verification and evidence filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from rag.models import Citation, Claim, ClaimVerification, Evidence


def verify_structural(
    claims: Sequence[Claim],
    evidence: Sequence[Evidence],
) -> tuple[ClaimVerification, ...]:
    """Deterministically verifies citation tag presence and valid referencing for all claims.

    - Factual claim without citations -> uncited (reason: uncited_claim).
    - Factual claim with tag not in retrieved evidence -> invalid_citation (reason: invalid_citation).
    - Nonfactual claim -> not_checked (reason: "").
    - Factual claim with valid citations -> not_checked (ready for semantic support verification).
    """
    valid_evidence_ids = {ev.evidence_id for ev in evidence}
    verifications: list[ClaimVerification] = []

    for claim in claims:
        if not claim.factual:
            verifications.append(
                ClaimVerification(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    factual=False,
                    importance=claim.importance,
                    citation_ids=claim.citation_ids,
                    status="not_checked",
                    reason_code="",
                )
            )
            continue

        if not claim.citation_ids:
            verifications.append(
                ClaimVerification(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    factual=True,
                    importance=claim.importance,
                    citation_ids=(),
                    status="uncited",
                    reason_code="uncited_claim",
                )
            )
            continue

        # Check if all referenced citation IDs exist in the retrieved evidence
        invalid_ids = [cid for cid in claim.citation_ids if cid not in valid_evidence_ids]
        if invalid_ids:
            verifications.append(
                ClaimVerification(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    factual=True,
                    importance=claim.importance,
                    citation_ids=claim.citation_ids,
                    status="invalid_citation",
                    reason_code="invalid_citation",
                )
            )
            continue

        verifications.append(
            ClaimVerification(
                claim_id=claim.claim_id,
                text=claim.text,
                factual=True,
                importance=claim.importance,
                citation_ids=claim.citation_ids,
                status="not_checked",
                reason_code="",
            )
        )

    return tuple(verifications)


def filter_cited_evidence(
    claims: Sequence[Claim],
    evidence: Sequence[Evidence],
) -> tuple[Evidence, ...]:
    """Filters retrieved evidence to include ONLY items referenced by at least one claim."""
    cited_ids = {cid for claim in claims for cid in claim.citation_ids}
    return tuple(ev for ev in evidence if ev.evidence_id in cited_ids)


def build_initial_citations(
    cited_evidence: Sequence[Evidence],
) -> tuple[Citation, ...]:
    """Builds initial Citation models for cited evidence chunks with sanitized basenames."""
    citations: list[Citation] = []
    for ev in cited_evidence:
        source_basename = Path(str(ev.source or "CV")).name
        citations.append(
            Citation(
                evidence_id=ev.evidence_id,
                chunk_id=ev.chunk_id,
                source=source_basename,
                page=ev.page,
                excerpt=ev.content.strip(),
                support_status="not_checked",
            )
        )
    return tuple(citations)


__all__ = [
    "verify_structural",
    "filter_cited_evidence",
    "build_initial_citations",
]
