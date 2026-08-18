"""Pydantic request and response schemas for chat endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatTurnSchema(BaseModel):
    """Conversational turn between user and assistant."""

    user: str
    assistant: str


class ChatRequest(BaseModel):
    """Payload for conversational question answering queries."""

    question: str
    history: list[ChatTurnSchema] = Field(default_factory=list)


class CitationSchema(BaseModel):
    """Attribution reference for generated answer assertions."""

    evidence_id: str
    chunk_id: str
    source: str
    page: int
    excerpt: str
    support_status: str = "not_checked"


class ClaimVerificationSchema(BaseModel):
    """Verification outcome for an individual factual claim."""

    claim_id: str
    text: str
    factual: bool = True
    importance: str = "core"
    citation_ids: list[str] = Field(default_factory=list)
    status: str = "not_checked"
    reason_code: str = ""


class ChatResponse(BaseModel):
    """Structured response payload for synchronous chat queries."""

    answer: str
    citations: list[CitationSchema] = Field(default_factory=list)
    claim_verifications: list[ClaimVerificationSchema] = Field(default_factory=list)
    decision: str = "answered"
    reason_codes: list[str] = Field(default_factory=list)
    retrieval_query: str = ""
    pipeline_version: str = ""


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ChatTurnSchema",
    "CitationSchema",
    "ClaimVerificationSchema",
]
