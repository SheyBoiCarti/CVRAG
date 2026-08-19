"""Pre-generation answerability signal extraction, calibrated threshold evaluation, and policy models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rag.models import AnswerabilityDecision, AnswerabilitySignals
from rag.verification.models import CalibrationPolicyArtifact

LOGGER = logging.getLogger(__name__)


def extract_answerability_signals(
    retrieval_result: Any,
    documents_count: int | None = None,
) -> AnswerabilitySignals:
    """Extracts diagnostic candidate signals from retrieval result stages.

    Missing signals remain None and are never silently converted to zero.
    """
    if retrieval_result is None:
        return AnswerabilitySignals(
            retrieved_count=documents_count if documents_count is not None else 0,
            dense_sparse_overlap=None,
            top_rrf_score=None,
            top_reranker_score=None,
            reranker_margin=None,
            retrieval_degraded=False,
        )

    context = getattr(retrieval_result, "context", ()) or ()
    retrieved_count = len(context) if context else (documents_count if documents_count is not None else 0)

    dense_candidates = getattr(retrieval_result, "dense_candidates", ()) or ()
    sparse_candidates = getattr(retrieval_result, "sparse_candidates", ()) or ()
    fused_candidates = getattr(retrieval_result, "fused_candidates", ()) or ()
    reranked_candidates = getattr(retrieval_result, "reranked_candidates", ()) or ()
    degraded_codes = getattr(retrieval_result, "degraded_reason_codes", ()) or ()

    # Dense / Sparse overlap among final context chunks
    dense_sparse_overlap: float | None = None
    if dense_candidates and sparse_candidates and context:
        dense_ids = {c.chunk.chunk_id for c in dense_candidates if hasattr(c, "chunk")}
        sparse_ids = {c.chunk.chunk_id for c in sparse_candidates if hasattr(c, "chunk")}
        context_ids = [c.chunk.chunk_id for c in context if hasattr(c, "chunk")]
        if context_ids:
            both_count = sum(1 for cid in context_ids if cid in dense_ids and cid in sparse_ids)
            dense_sparse_overlap = float(both_count) / float(len(context_ids))

    # Top RRF score
    top_rrf_score: float | None = None
    if fused_candidates and getattr(fused_candidates[0], "score", None) is not None:
        top_rrf_score = float(fused_candidates[0].score)

    # Top reranker score and margin
    top_reranker_score: float | None = None
    reranker_margin: float | None = None
    if reranked_candidates and getattr(reranked_candidates[0], "score", None) is not None:
        top_reranker_score = float(reranked_candidates[0].score)
        if len(reranked_candidates) >= 2 and getattr(reranked_candidates[1], "score", None) is not None:
            reranker_margin = float(top_reranker_score - float(reranked_candidates[1].score))

    retrieval_degraded = bool(degraded_codes)

    return AnswerabilitySignals(
        retrieved_count=retrieved_count,
        dense_sparse_overlap=dense_sparse_overlap,
        top_rrf_score=top_rrf_score,
        top_reranker_score=top_reranker_score,
        reranker_margin=reranker_margin,
        retrieval_degraded=retrieval_degraded,
    )


def evaluate_pre_generation_answerability(
    signals: AnswerabilitySignals,
    policy_artifact: CalibrationPolicyArtifact | None = None,
) -> AnswerabilityDecision:
    """Evaluates pre-generation answerability decision based on retrieved signals and calibrated policy.

    Always strictly abstains on empty retrieval.
    """
    version = policy_artifact.policy_version if policy_artifact else "deterministic_v1"

    # Deterministic empty retrieval check
    if signals.retrieved_count == 0:
        return AnswerabilityDecision(
            decision="abstain",
            reason_codes=("no_evidence",),
            policy_version=version,
        )

    # If a calibrated policy is active, evaluate configured signal threshold
    if policy_artifact is not None:
        signal_val: float | None = None
        if policy_artifact.signal_name == "top_reranker_score":
            signal_val = signals.top_reranker_score
        elif policy_artifact.signal_name == "top_rrf_score":
            signal_val = signals.top_rrf_score
        elif policy_artifact.signal_name == "dense_sparse_overlap":
            signal_val = signals.dense_sparse_overlap
        elif policy_artifact.signal_name == "reranker_margin":
            signal_val = signals.reranker_margin

        if signal_val is not None:
            if signal_val < policy_artifact.threshold:
                return AnswerabilityDecision(
                    decision="abstain",
                    reason_codes=("insufficient_evidence",),
                    policy_version=policy_artifact.policy_version,
                )
            return AnswerabilityDecision(
                decision="answer",
                reason_codes=(),
                policy_version=policy_artifact.policy_version,
            )

    return AnswerabilityDecision(
        decision="answer",
        reason_codes=(),
        policy_version=version,
    )


def load_calibration_policy(filepath: str | Path | None) -> CalibrationPolicyArtifact | None:
    """Loads a calibrated policy artifact from a JSON file."""
    if not filepath:
        return None
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        LOGGER.warning("Calibration policy file not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CalibrationPolicyArtifact(
            policy_version=data["policy_version"],
            dataset_hash=data["dataset_hash"],
            corpus_fingerprint=data["corpus_fingerprint"],
            index_fingerprint=data["index_fingerprint"],
            model_name=data["model_name"],
            signal_name=data["signal_name"],
            threshold=float(data["threshold"]),
            cost_objective=float(data["cost_objective"]),
            false_answer_rate=float(data["false_answer_rate"]),
            false_abstention_rate=float(data["false_abstention_rate"]),
            ci_lower=float(data["ci_lower"]),
            ci_upper=float(data["ci_upper"]),
            locked_set_evaluated=bool(data.get("locked_set_evaluated", True)),
            metadata=dict(data.get("metadata", {})),
        )
    except Exception as exc:
        LOGGER.warning("Failed to load calibration policy from %s (type=%s)", path, type(exc).__name__)
        return None


def save_calibration_policy(artifact: CalibrationPolicyArtifact, filepath: str | Path) -> None:
    """Saves a calibrated policy artifact to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact.to_dict(), f, indent=2, ensure_ascii=False)


__all__ = [
    "extract_answerability_signals",
    "evaluate_pre_generation_answerability",
    "load_calibration_policy",
    "save_calibration_policy",
]
