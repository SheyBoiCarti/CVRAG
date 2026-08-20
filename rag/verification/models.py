"""Verifier-internal provider verdict, calibration, and support models.

These models are internal to the verification package and must not redefine public domain models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from rag.models import Claim, ClaimImportance, ClaimVerificationStatus, Evidence


@dataclass(frozen=True)
class SupportVerdict:
    """Outcome of claim-level semantic support verification against cited evidence."""

    claim_id: str
    combined_supported: bool
    supporting_evidence_ids: tuple[str, ...] = ()
    reason_code: str = ""
    verifier_model: str = ""
    verifier_version: str = "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.supporting_evidence_ids, tuple):
            object.__setattr__(
                self, "supporting_evidence_ids", tuple(self.supporting_evidence_ids)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "combined_supported": self.combined_supported,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "reason_code": self.reason_code,
            "verifier_model": self.verifier_model,
            "verifier_version": self.verifier_version,
        }


@dataclass(frozen=True)
class CalibrationPolicyArtifact:
    """Serialized calibration policy containing optimized thresholds and validation metadata."""

    policy_version: str
    dataset_hash: str
    corpus_fingerprint: str
    index_fingerprint: str
    model_name: str
    signal_name: str
    threshold: float
    cost_objective: float
    false_answer_rate: float
    false_abstention_rate: float
    ci_lower: float
    ci_upper: float
    locked_set_evaluated: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "dataset_hash": self.dataset_hash,
            "corpus_fingerprint": self.corpus_fingerprint,
            "index_fingerprint": self.index_fingerprint,
            "model_name": self.model_name,
            "signal_name": self.signal_name,
            "threshold": self.threshold,
            "cost_objective": self.cost_objective,
            "false_answer_rate": self.false_answer_rate,
            "false_abstention_rate": self.false_abstention_rate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "locked_set_evaluated": self.locked_set_evaluated,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "SupportVerdict",
    "CalibrationPolicyArtifact",
]
