"""Deterministic answer metrics: fact coverage, forbidden facts, exact match, abstentions."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence
import unicodedata

from evaluation.schema import FactSpec, GoldenCase
from rag.models import AnswerResult


def normalize_answer_text(text: str) -> str:
    """Normalizes answer text using Unicode NFKC, casefolding, and collapsing whitespace."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def match_fact(text: str, fact: FactSpec) -> bool:
    """Matches a fact specification against text based on its configured matcher."""
    norm_text = normalize_answer_text(text)
    if not norm_text:
        return False

    matcher = fact.matcher
    norm_val = normalize_answer_text(fact.value)

    if matcher == "exact_phrase":
        return norm_val in norm_text

    if matcher == "token_phrase":
        pattern = r"(?<!\w)" + re.escape(norm_val) + r"(?!\w)"
        return bool(re.search(pattern, norm_text))

    if matcher == "alias_set":
        candidates = [fact.value] + list(fact.aliases)
        for cand in candidates:
            cand_norm = normalize_answer_text(cand)
            pattern = r"(?<!\w)" + re.escape(cand_norm) + r"(?!\w)"
            if re.search(pattern, norm_text):
                return True
        return False

    return False


@dataclass(frozen=True)
class AnswerCaseMetrics:
    """Answer metrics evaluated for an individual case."""

    case_id: str
    is_answerable: bool
    decision: str
    required_fact_coverage: float
    forbidden_fact_violation: float
    exact_match: float
    correct_abstention: float
    incorrect_abstention: float
    total_required_facts: int
    matched_required_facts: int
    total_forbidden_facts: int
    violated_forbidden_facts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "is_answerable": self.is_answerable,
            "decision": self.decision,
            "required_fact_coverage": self.required_fact_coverage,
            "forbidden_fact_violation": self.forbidden_fact_violation,
            "exact_match": self.exact_match,
            "correct_abstention": self.correct_abstention,
            "incorrect_abstention": self.incorrect_abstention,
            "total_required_facts": self.total_required_facts,
            "matched_required_facts": self.matched_required_facts,
            "total_forbidden_facts": self.total_forbidden_facts,
            "violated_forbidden_facts": self.violated_forbidden_facts,
        }


def evaluate_answer(case: GoldenCase, answer_result: AnswerResult) -> AnswerCaseMetrics:
    """Evaluates an AnswerResult against a GoldenCase."""
    answer_text = answer_result.answer
    decision = answer_result.decision

    is_abstained = (decision == "abstained") or (not answer_text.strip())

    if not case.answerable:
        # Ground truth is unanswerable
        correct_abstention = 1.0 if is_abstained else 0.0
        incorrect_abstention = 0.0
        req_cov = 0.0
        exact_m = 0.0
        total_req = 0
        matched_req = 0
    else:
        # Ground truth is answerable
        correct_abstention = 0.0
        incorrect_abstention = 1.0 if is_abstained else 0.0
        total_req = len(case.required_facts)
        matched_req = 0
        if not is_abstained and total_req > 0:
            matched_req = sum(1 for f in case.required_facts if match_fact(answer_text, f))
            req_cov = matched_req / total_req
        elif not is_abstained and total_req == 0:
            req_cov = 1.0
        else:
            req_cov = 0.0

        # Exact match (casefold / whitespace stripped)
        norm_actual = normalize_answer_text(answer_text)
        norm_ref = normalize_answer_text(case.reference_answer)
        exact_m = 1.0 if (norm_actual == norm_ref and norm_actual != "") else 0.0

    # Forbidden facts check
    total_forb = len(case.forbidden_facts)
    violated_forb = 0
    if total_forb > 0 and not is_abstained:
        violated_forb = sum(1 for f in case.forbidden_facts if match_fact(answer_text, f))
        forb_viol = violated_forb / total_forb
    else:
        forb_viol = 0.0

    return AnswerCaseMetrics(
        case_id=case.id,
        is_answerable=case.answerable,
        decision=decision,
        required_fact_coverage=req_cov,
        forbidden_fact_violation=forb_viol,
        exact_match=exact_m,
        correct_abstention=correct_abstention,
        incorrect_abstention=incorrect_abstention,
        total_required_facts=total_req,
        matched_required_facts=matched_req,
        total_forbidden_facts=total_forb,
        violated_forbidden_facts=violated_forb,
    )


def aggregate_answer_metrics(
    metrics_list: Sequence[AnswerCaseMetrics],
) -> dict[str, float]:
    """Computes macro aggregates for answer metrics."""
    if not metrics_list:
        return {}

    ans_cases = [m for m in metrics_list if m.is_answerable]
    unans_cases = [m for m in metrics_list if not m.is_answerable]

    agg: dict[str, float] = {
        "total_cases": len(metrics_list),
        "answerable_cases": len(ans_cases),
        "unanswerable_cases": len(unans_cases),
    }

    if ans_cases:
        n_ans = len(ans_cases)
        agg["required_fact_coverage"] = sum(m.required_fact_coverage for m in ans_cases) / n_ans
        agg["forbidden_fact_violation"] = sum(m.forbidden_fact_violation for m in ans_cases) / n_ans
        agg["exact_match"] = sum(m.exact_match for m in ans_cases) / n_ans
        agg["incorrect_abstention_rate"] = sum(m.incorrect_abstention for m in ans_cases) / n_ans

    if unans_cases:
        n_unans = len(unans_cases)
        agg["correct_abstention_rate"] = sum(m.correct_abstention for m in unans_cases) / n_unans
        agg["false_answer_rate"] = 1.0 - agg["correct_abstention_rate"]

    return agg


__all__ = [
    "normalize_answer_text",
    "match_fact",
    "AnswerCaseMetrics",
    "evaluate_answer",
    "aggregate_answer_metrics",
]
