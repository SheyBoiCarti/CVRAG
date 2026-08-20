"""Evaluation metrics package."""

from evaluation.metrics.answers import (
    AnswerCaseMetrics,
    aggregate_answer_metrics,
    evaluate_answer,
    match_fact,
    normalize_answer_text,
)
from evaluation.metrics.citations import (
    CitationCaseMetrics,
    aggregate_citation_metrics,
    evaluate_citations,
)
from evaluation.metrics.retrieval import (
    RetrievalCaseMetrics,
    aggregate_retrieval_metrics,
    evaluate_retrieval,
    matches_locator,
    normalize_text,
)

__all__ = [
    "AnswerCaseMetrics",
    "CitationCaseMetrics",
    "RetrievalCaseMetrics",
    "aggregate_answer_metrics",
    "aggregate_citation_metrics",
    "aggregate_retrieval_metrics",
    "evaluate_answer",
    "evaluate_citations",
    "evaluate_retrieval",
    "match_fact",
    "matches_locator",
    "normalize_answer_text",
    "normalize_text",
]
