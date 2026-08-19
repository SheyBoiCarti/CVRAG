"""Immutable domain models and event envelope contracts for the RAG pipeline.

These models are standard-library-only, frontend-independent, and immutable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import types
from typing import Any, Literal, Mapping, Sequence

PipelineDecision = Literal["answered", "abstained", "degraded"]
EvidenceSupportStatus = Literal["supported", "unsupported", "not_checked"]
ClaimVerificationStatus = Literal["supported", "unsupported", "uncited", "invalid_citation", "not_checked"]
ClaimImportance = Literal["core", "optional"]
ReadinessState = Literal["starting", "ready", "not_ready"]


def _serialize_value(val: Any) -> Any:
    """Recursively serializes domain models, mappings, and sequences for JSON."""
    if hasattr(val, "to_dict") and callable(val.to_dict):
        return _serialize_value(val.to_dict())
    if isinstance(val, Mapping):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_serialize_value(v) for v in val]
    return val


@dataclass(frozen=True)
class ChatTurn:
    """Represents a single conversational turn between user and assistant."""

    user: str
    assistant: str

    def to_dict(self) -> dict[str, str]:
        return {
            "user": self.user,
            "assistant": self.assistant,
        }


@dataclass(frozen=True)
class Evidence:
    """Retrieved evidence chunk from the vector store."""

    evidence_id: str
    chunk_id: str
    content: str
    source: str
    page: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", types.MappingProxyType({}))
        elif not isinstance(self.metadata, types.MappingProxyType):
            object.__setattr__(self, "metadata", types.MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Citation:
    """Attribution reference for generated content."""

    evidence_id: str
    chunk_id: str
    source: str
    page: int
    excerpt: str
    support_status: EvidenceSupportStatus = "not_checked"

    def __post_init__(self) -> None:
        if self.source:
            norm_source = self.source.replace("\\", "/").rstrip("/").split("/")[-1]
            object.__setattr__(self, "source", norm_source or self.source)

    def to_dict(self) -> dict[str, Any]:
        norm_source = self.source.replace("\\", "/").rstrip("/").split("/")[-1] if self.source else ""
        return {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "source": norm_source,
            "page": self.page,
            "excerpt": self.excerpt,
            "support_status": self.support_status,
        }


@dataclass(frozen=True)
class ClaimVerification:
    """Verification outcome for an individual factual claim."""

    claim_id: str
    text: str
    factual: bool = True
    importance: ClaimImportance = "core"
    citation_ids: tuple[str, ...] = ()
    status: ClaimVerificationStatus = "not_checked"
    reason_code: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.citation_ids, tuple):
            object.__setattr__(self, "citation_ids", tuple(self.citation_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "factual": self.factual,
            "importance": self.importance,
            "citation_ids": list(self.citation_ids),
            "status": self.status,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class AnswerResult:
    """Complete, immutable result of a RAG query answering operation."""

    answer: str
    citations: tuple[Citation, ...] = ()
    claim_verifications: tuple[ClaimVerification, ...] = ()
    decision: PipelineDecision = "answered"
    reason_codes: tuple[str, ...] = ()
    retrieval_query: str = ""
    pipeline_version: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.citations, tuple):
            object.__setattr__(self, "citations", tuple(self.citations))
        if not isinstance(self.claim_verifications, tuple):
            object.__setattr__(self, "claim_verifications", tuple(self.claim_verifications))
        if not isinstance(self.reason_codes, tuple):
            object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() if hasattr(c, "to_dict") else c for c in self.citations],
            "claim_verifications": [
                cv.to_dict() if hasattr(cv, "to_dict") else cv for cv in self.claim_verifications
            ],
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "retrieval_query": self.retrieval_query,
            "pipeline_version": self.pipeline_version,
        }


@dataclass(frozen=True)
class ReadinessResult:
    """Pipeline and index health readiness result."""

    state: ReadinessState
    reason_codes: tuple[str, ...] = ()
    pipeline_version: str | None = None
    index_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reason_codes, tuple):
            object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    @property
    def is_ready(self) -> bool:
        return self.state == "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason_codes": list(self.reason_codes),
            "pipeline_version": self.pipeline_version,
            "index_fingerprint": self.index_fingerprint,
        }


@dataclass(frozen=True)
class PipelineEvent:
    """Envelope for streaming pipeline events following Schema Version 1."""

    schema_version: str = "1"
    request_id: str = ""
    sequence: int = 0
    event: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data is None:
            object.__setattr__(self, "data", types.MappingProxyType({}))
        elif not isinstance(self.data, types.MappingProxyType):
            object.__setattr__(self, "data", types.MappingProxyType(dict(self.data)))

    @classmethod
    def create(
        cls,
        event: str,
        data: Mapping[str, Any] | dict[str, Any],
        *,
        request_id: str,
        sequence: int,
        schema_version: str = "1",
    ) -> PipelineEvent:
        """Helper factory for creating envelope-compliant events."""
        return cls(
            schema_version=schema_version,
            request_id=request_id,
            sequence=sequence,
            event=event,
            data=dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "event": self.event,
            "data": _serialize_value(self.data),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_sse(self) -> str:
        return f"data: {self.to_json()}\n\n"


__all__ = [
    "PipelineDecision",
    "EvidenceSupportStatus",
    "ClaimVerificationStatus",
    "ClaimImportance",
    "ReadinessState",
    "ChatTurn",
    "Evidence",
    "Citation",
    "ClaimVerification",
    "AnswerResult",
    "ReadinessResult",
    "PipelineEvent",
]
