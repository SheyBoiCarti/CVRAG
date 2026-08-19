from __future__ import annotations

from rag.models import AnswerabilityDecision, AnswerabilitySignals
from rag.retrieval.base import ChunkRecord, RankedChunk
from rag.retrieval.service import RetrievalResult
from rag.verification.answerability import (
    evaluate_pre_generation_answerability,
    extract_answerability_signals,
)
from rag.verification.models import CalibrationPolicyArtifact


def _make_dummy_ranked_chunk(chunk_id: str, rank: int, score: float | None = None) -> RankedChunk:
    return RankedChunk(
        chunk=ChunkRecord(
            chunk_id=chunk_id,
            content=f"Content for {chunk_id}",
            source="Custom_CV.md",
            page=1,
            chunk_index=rank - 1,
        ),
        rank=rank,
        score=score,
        methods=("dense",),
    )


def test_extract_signals_empty_retrieval():
    ret_res = RetrievalResult(
        query="unknown query",
        context=(),
        fused_candidates=(),
        degraded_reason_codes=(),
        index_fingerprint="fp1",
    )
    signals = extract_answerability_signals(ret_res)
    assert signals.retrieved_count == 0
    assert signals.dense_sparse_overlap is None
    assert signals.top_rrf_score is None
    assert signals.top_reranker_score is None
    assert signals.reranker_margin is None
    assert signals.retrieval_degraded is False


def test_extract_signals_hybrid_reranked():
    c1 = _make_dummy_ranked_chunk("c1", 1, score=0.9)
    c2 = _make_dummy_ranked_chunk("c2", 2, score=0.6)
    c3 = _make_dummy_ranked_chunk("c3", 3, score=0.3)

    ret_res = RetrievalResult(
        query="experience",
        context=(c1, c2),
        fused_candidates=(c1, c2, c3),
        degraded_reason_codes=(),
        index_fingerprint="fp1",
        dense_candidates=(c1, c2),
        sparse_candidates=(c1, c3),
        reranked_candidates=(c1, c2),
    )
    signals = extract_answerability_signals(ret_res)
    assert signals.retrieved_count == 2
    assert signals.dense_sparse_overlap == 0.5  # 1 out of 2 final chunks (c1) in both dense & sparse
    assert signals.top_reranker_score == 0.9
    assert round(signals.reranker_margin or 0.0, 2) == 0.30
    assert signals.retrieval_degraded is False


def test_evaluate_pre_generation_empty_retrieval_strictly_abstains():
    signals = AnswerabilitySignals(
        retrieved_count=0,
        dense_sparse_overlap=None,
        top_rrf_score=None,
        top_reranker_score=None,
        reranker_margin=None,
        retrieval_degraded=False,
    )
    decision = evaluate_pre_generation_answerability(signals)
    assert decision.decision == "abstain"
    assert decision.reason_codes == ("no_evidence",)


def test_evaluate_pre_generation_uncalibrated_nonempty():
    signals = AnswerabilitySignals(
        retrieved_count=3,
        dense_sparse_overlap=1.0,
        top_rrf_score=0.03,
        top_reranker_score=None,
        reranker_margin=None,
        retrieval_degraded=False,
    )
    decision = evaluate_pre_generation_answerability(signals, policy_artifact=None)
    assert decision.decision == "answer"
    assert decision.reason_codes == ()


def test_evaluate_pre_generation_calibrated_threshold():
    policy = CalibrationPolicyArtifact(
        policy_version="2026-08-19-calibrated",
        dataset_hash="hash123",
        corpus_fingerprint="corpus1",
        index_fingerprint="index1",
        model_name="nvidia/llama-nemotron-rerank-1b-v2",
        signal_name="top_reranker_score",
        threshold=0.75,
        cost_objective=1.5,
        false_answer_rate=0.02,
        false_abstention_rate=0.05,
        ci_lower=0.70,
        ci_upper=0.80,
    )

    # Score above threshold -> answer
    sig_pass = AnswerabilitySignals(retrieved_count=2, top_reranker_score=0.85)
    dec_pass = evaluate_pre_generation_answerability(sig_pass, policy_artifact=policy)
    assert dec_pass.decision == "answer"

    # Score below threshold -> abstain with insufficient_evidence
    sig_fail = AnswerabilitySignals(retrieved_count=2, top_reranker_score=0.60)
    dec_fail = evaluate_pre_generation_answerability(sig_fail, policy_artifact=policy)
    assert dec_fail.decision == "abstain"
    assert dec_fail.reason_codes == ("insufficient_evidence",)
    assert dec_fail.policy_version == "2026-08-19-calibrated"
