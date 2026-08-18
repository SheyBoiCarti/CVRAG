"""Deterministic in-memory FakeJudge for offline testing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from evaluation.judges.base import AnswerJudge, JudgeResult
from evaluation.schema import GoldenCase


class FakeJudge:
    """Offline test judge that records calls and produces configured or default deterministic results."""

    def __init__(
        self,
        correctness: float = 1.0,
        faithfulness: float = 1.0,
        citation_support: float = 1.0,
        rationale: str = "Fake judge evaluation passed.",
        model: str = "fake-judge-model",
        prompt_version: str = "fake-v1",
    ) -> None:
        self.correctness = correctness
        self.faithfulness = faithfulness
        self.citation_support = citation_support
        self.rationale = rationale
        self.model = model
        self.prompt_version = prompt_version
        self.calls: list[dict[str, Any]] = []

    def judge(
        self,
        case: GoldenCase,
        answer: str,
        evidence_texts: Sequence[str],
    ) -> JudgeResult:
        self.calls.append({
            "case_id": case.id,
            "answer": answer,
            "evidence_texts": list(evidence_texts),
        })
        return JudgeResult(
            correctness=self.correctness,
            faithfulness=self.faithfulness,
            citation_support=self.citation_support,
            rationale=self.rationale,
            model=self.model,
            prompt_version=self.prompt_version,
            provider_revision="fake-rev-0",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


__all__ = ["FakeJudge"]
