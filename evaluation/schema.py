"""Immutable evaluation data contracts, schemas, and artifact models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from rag.models import AnswerResult
from rag.retrieval.base import RankedChunk

CaseCategory = Literal[
    "straightforward",
    "exact_term",
    "multi_hop",
    "conversational_follow_up",
    "ambiguous",
    "unanswerable",
]

ALLOWED_CATEGORIES: set[str] = {
    "straightforward",
    "exact_term",
    "multi_hop",
    "conversational_follow_up",
    "ambiguous",
    "unanswerable",
}

FactMatcher = Literal["exact_phrase", "token_phrase", "alias_set"]

ALLOWED_FACT_MATCHERS: set[str] = {
    "exact_phrase",
    "token_phrase",
    "alias_set",
}


def _immutable_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


def _serialize_for_json(val: Any) -> Any:
    """Recursively serializes models, mappings, and sequences for standard JSON."""
    if hasattr(val, "to_dict") and callable(val.to_dict):
        return _serialize_for_json(val.to_dict())
    if isinstance(val, Mapping):
        return {str(k): _serialize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_serialize_for_json(v) for v in val]
    return val


def canonical_json_dumps(obj: Any) -> str:
    """Serializes a Python object to canonical JSON with sorted keys, no whitespace, and UTF-8."""
    serializable = _serialize_for_json(obj)
    return json.dumps(serializable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_content_hash(content: Mapping[str, Any] | dict[str, Any]) -> str:
    """Computes SHA-256 hash over canonical JSON representation of content."""
    canonical = canonical_json_dumps(content)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FactSpec:
    """Specification of a required or forbidden factual assertion in an answer."""

    matcher: FactMatcher
    value: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.matcher not in ALLOWED_FACT_MATCHERS:
            raise ValueError(f"Invalid matcher '{self.matcher}'. Must be one of {ALLOWED_FACT_MATCHERS}")
        if not isinstance(self.aliases, tuple):
            object.__setattr__(self, "aliases", tuple(self.aliases))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FactSpec:
        return cls(
            matcher=data["matcher"],
            value=data["value"],
            aliases=tuple(data.get("aliases", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "matcher": self.matcher,
            "value": self.value,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True)
class EvidenceLocator:
    """Canonical ground truth location for retrieved evidence."""

    source: str
    page: int
    anchor: str

    def __post_init__(self) -> None:
        if not self.source or not self.source.strip():
            raise ValueError("Source cannot be empty.")
        if self.page < 1:
            raise ValueError(f"Page must be >= 1, got {self.page}")
        if not self.anchor or not self.anchor.strip():
            raise ValueError("Anchor cannot be empty.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceLocator:
        return cls(
            source=data["source"],
            page=int(data["page"]),
            anchor=data["anchor"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "anchor": self.anchor,
        }


@dataclass(frozen=True)
class GoldenCase:
    """A versioned evaluation record with ground truth questions, facts, and evidence groups."""

    id: str
    category: CaseCategory
    question: str
    history: tuple[tuple[str, str], ...] = ()
    answerable: bool = True
    reference_answer: str = ""
    required_facts: tuple[FactSpec, ...] = ()
    forbidden_facts: tuple[FactSpec, ...] = ()
    evidence_groups: tuple[tuple[EvidenceLocator, ...], ...] = ()
    ambiguity_note: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Case id cannot be empty.")
        if self.category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Invalid category '{self.category}'. Allowed: {ALLOWED_CATEGORIES}")
        if not self.question or not self.question.strip():
            raise ValueError("Question cannot be empty.")
        if not isinstance(self.history, tuple):
            object.__setattr__(self, "history", tuple(tuple(turn) for turn in self.history))
        if not isinstance(self.required_facts, tuple):
            object.__setattr__(self, "required_facts", tuple(self.required_facts))
        if not isinstance(self.forbidden_facts, tuple):
            object.__setattr__(self, "forbidden_facts", tuple(self.forbidden_facts))
        if not isinstance(self.evidence_groups, tuple):
            object.__setattr__(
                self,
                "evidence_groups",
                tuple(tuple(group) for group in self.evidence_groups),
            )
        if not isinstance(self.tags, tuple):
            object.__setattr__(self, "tags", tuple(self.tags))

        # Schema rules
        if not self.answerable:
            if self.evidence_groups:
                raise ValueError(
                    f"Unanswerable case '{self.id}' cannot have required evidence groups."
                )
        else:
            if self.category != "unanswerable" and not self.evidence_groups and not self.tags:
                pass  # answerable cases generally specify evidence groups

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GoldenCase:
        history = tuple(
            (turn[0], turn[1]) if isinstance(turn, (list, tuple)) else (turn.get("user", ""), turn.get("assistant", ""))
            for turn in data.get("history", ())
        )
        req_facts = tuple(
            f if isinstance(f, FactSpec) else FactSpec.from_dict(f)
            for f in data.get("required_facts", ())
        )
        forb_facts = tuple(
            f if isinstance(f, FactSpec) else FactSpec.from_dict(f)
            for f in data.get("forbidden_facts", ())
        )
        evidence_groups = tuple(
            tuple(
                loc if isinstance(loc, EvidenceLocator) else EvidenceLocator.from_dict(loc)
                for loc in group
            )
            for group in data.get("evidence_groups", ())
        )
        return cls(
            id=data["id"],
            category=data["category"],
            question=data["question"],
            history=history,
            answerable=bool(data.get("answerable", True)),
            reference_answer=data.get("reference_answer", ""),
            required_facts=req_facts,
            forbidden_facts=forb_facts,
            evidence_groups=evidence_groups,
            ambiguity_note=data.get("ambiguity_note"),
            tags=tuple(data.get("tags", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "history": [list(h) for h in self.history],
            "answerable": self.answerable,
            "reference_answer": self.reference_answer,
            "required_facts": [f.to_dict() for f in self.required_facts],
            "forbidden_facts": [f.to_dict() for f in self.forbidden_facts],
            "evidence_groups": [
                [loc.to_dict() for loc in group] for group in self.evidence_groups
            ],
            "ambiguity_note": self.ambiguity_note,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class EvaluationTrace:
    """Trace diagnostics captured across retrieval stages and answer synthesis."""

    case_id: str
    retrieval_query: str
    dense_results: tuple[RankedChunk, ...]
    sparse_results: tuple[RankedChunk, ...]
    fused_results: tuple[RankedChunk, ...]
    reranked_results: tuple[RankedChunk, ...]
    answer_result: AnswerResult
    timings_ms: Mapping[str, float]
    token_usage: Mapping[str, int]
    configuration: Mapping[str, Any]
    corpus_fingerprint: str
    index_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.dense_results, tuple):
            object.__setattr__(self, "dense_results", tuple(self.dense_results))
        if not isinstance(self.sparse_results, tuple):
            object.__setattr__(self, "sparse_results", tuple(self.sparse_results))
        if not isinstance(self.fused_results, tuple):
            object.__setattr__(self, "fused_results", tuple(self.fused_results))
        if not isinstance(self.reranked_results, tuple):
            object.__setattr__(self, "reranked_results", tuple(self.reranked_results))
        object.__setattr__(self, "timings_ms", _immutable_mapping(self.timings_ms))
        object.__setattr__(self, "token_usage", _immutable_mapping(self.token_usage))
        object.__setattr__(self, "configuration", _immutable_mapping(self.configuration))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "retrieval_query": self.retrieval_query,
            "dense_results": [r.to_dict() for r in self.dense_results],
            "sparse_results": [r.to_dict() for r in self.sparse_results],
            "fused_results": [r.to_dict() for r in self.fused_results],
            "reranked_results": [r.to_dict() for r in self.reranked_results],
            "answer_result": self.answer_result.to_dict(),
            "timings_ms": dict(self.timings_ms),
            "token_usage": dict(self.token_usage),
            "configuration": dict(self.configuration),
            "corpus_fingerprint": self.corpus_fingerprint,
            "index_fingerprint": self.index_fingerprint,
        }


@dataclass(frozen=True)
class RunArtifact:
    """Serializable evaluation run artifact separating reproducible content from volatile metadata."""

    content_fingerprint: str
    content: Mapping[str, Any]
    run_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", _immutable_mapping(self.content))
        object.__setattr__(self, "run_metadata", _immutable_mapping(self.run_metadata))

    @classmethod
    def create(
        cls,
        *,
        content: Mapping[str, Any] | dict[str, Any],
        run_id: str,
        timestamp: str,
        git_revision: str = "",
        git_dirty: bool = False,
        duration_ms: float = 0.0,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> RunArtifact:
        fingerprint = compute_content_hash(content)
        metadata = {
            "run_id": run_id,
            "timestamp": timestamp,
            "git_revision": git_revision,
            "git_dirty": git_dirty,
            "duration_ms": duration_ms,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return cls(
            content_fingerprint=fingerprint,
            content=content,
            run_metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_fingerprint": self.content_fingerprint,
            "content": _serialize_for_json(self.content),
            "run_metadata": _serialize_for_json(self.run_metadata),
        }

    def save(self, filepath: str | Any) -> None:
        from pathlib import Path

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


__all__ = [
    "ALLOWED_CATEGORIES",
    "ALLOWED_FACT_MATCHERS",
    "CaseCategory",
    "EvaluationTrace",
    "EvidenceLocator",
    "FactMatcher",
    "FactSpec",
    "GoldenCase",
    "RunArtifact",
    "canonical_json_dumps",
    "compute_content_hash",
]
