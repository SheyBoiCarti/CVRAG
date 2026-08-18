"""Base protocol, structured outcome contracts, and disk caching for LLM judges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from evaluation.schema import GoldenCase


@dataclass(frozen=True)
class JudgeResult:
    """Structured evaluation judgment from an LLM or heuristic judge."""

    correctness: float | None
    faithfulness: float | None
    citation_support: float | None
    rationale: str
    model: str
    prompt_version: str
    provider_revision: str = ""
    timestamp: str = ""
    raw_response: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "faithfulness": self.faithfulness,
            "citation_support": self.citation_support,
            "rationale": self.rationale,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "provider_revision": self.provider_revision,
            "timestamp": self.timestamp,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JudgeResult:
        return cls(
            correctness=data.get("correctness"),
            faithfulness=data.get("faithfulness"),
            citation_support=data.get("citation_support"),
            rationale=str(data.get("rationale", "")),
            model=str(data.get("model", "")),
            prompt_version=str(data.get("prompt_version", "")),
            provider_revision=str(data.get("provider_revision", "")),
            timestamp=str(data.get("timestamp", "")),
            error=data.get("error"),
        )


class AnswerJudge(Protocol):
    """Protocol for LLM-based answer and faithfulness judges."""

    def judge(
        self,
        case: GoldenCase,
        answer: str,
        evidence_texts: Sequence[str],
    ) -> JudgeResult: ...


class JudgeCacheManager:
    """Hash-keyed disk cache for LLM judge outcomes to guarantee replayability and auditability."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            base_dir = Path(__file__).resolve().parents[2]
            self.cache_dir = base_dir / ".judge_cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_cache_key(
        case_id: str,
        answer: str,
        evidence_texts: Sequence[str],
        model: str,
        prompt_version: str,
    ) -> str:
        evidence_joined = "\n---\n".join(sorted(evidence_texts))
        raw = f"{case_id}::{answer.strip()}::{evidence_joined}::{model}::{prompt_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        case_id: str,
        answer: str,
        evidence_texts: Sequence[str],
        model: str,
        prompt_version: str,
    ) -> JudgeResult | None:
        key = self.compute_cache_key(case_id, answer, evidence_texts, model, prompt_version)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return JudgeResult.from_dict(data)
            except Exception:
                return None
        return None

    def put(
        self,
        case_id: str,
        answer: str,
        evidence_texts: Sequence[str],
        model: str,
        prompt_version: str,
        result: JudgeResult,
    ) -> None:
        key = self.compute_cache_key(case_id, answer, evidence_texts, model, prompt_version)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception:
            pass


__all__ = [
    "JudgeResult",
    "AnswerJudge",
    "JudgeCacheManager",
]
