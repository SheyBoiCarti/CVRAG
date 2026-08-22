"""Evaluation runner orchestrating batch evaluation, metrics aggregation, and artifact export.

Architecture & Role:
-------------------
This module implements the automated RAG evaluation framework:
1. Benchmark Harness:
   - Iterates over golden test cases (`cv_golden.jsonl`) spanning straightforward, exact-term, multi-hop,
     conversational follow-up, ambiguous, and unanswerable categories.
2. Multi-Stage Metric Scoring:
   - Retrieval: Precision@K, Recall@K, Reciprocal Rank (MRR), NDCG, and Hit Rate.
   - Citations: Citation Precision, Recall, F1, and Grounding Coverage.
   - Answers: Token F1, Exact Match, Required/Forbidden Fact Matchers, and LLM-as-a-Judge semantic grading.
   - Latency & Reliability: Stage-by-stage timings (retrieval, generation, verification) and degradation tracking.
3. Reproducible Baseline Artifacts:
   - Exports complete evaluation run snapshots (`RunArtifact`) for regression tracking and baseline diffing (`evaluation.compare`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics.answers import (
    AnswerCaseMetrics,
    aggregate_answer_metrics,
    evaluate_answer,
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
)
from evaluation.schema import GoldenCase, RunArtifact
from rag.models import ChatTurn
from rag.pipeline import RAGPipeline
from rag.settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)

PRESET_CONFIGS = {
    "dense-baseline": {
        "RETRIEVAL_MODE": "dense",
        "RERANK_ENABLED": False,
    },
    "hybrid-no-rerank": {
        "RETRIEVAL_MODE": "hybrid",
        "RERANK_ENABLED": False,
    },
    "hybrid-rerank": {
        "RETRIEVAL_MODE": "hybrid",
        "RERANK_ENABLED": True,
    },
}


def load_dataset(dataset_path: str | Path) -> tuple[list[GoldenCase], str]:
    """Loads JSONL golden dataset and computes SHA-256 hash of raw file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {path}")

    with open(path, "rb") as f:
        raw_bytes = f.read()
    dataset_hash = hashlib.sha256(raw_bytes).hexdigest()

    cases: list[GoldenCase] = []
    text = raw_bytes.decode("utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(GoldenCase.from_dict(json.loads(line)))

    return cases, dataset_hash


def get_git_metadata() -> tuple[str, bool]:
    """Retrieves current git commit hash and dirty status."""
    try:
        rev = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        is_dirty = len(status) > 0
        return rev, is_dirty
    except Exception:
        return "unknown", False


def compute_latency_percentiles(durations_ms: Sequence[float]) -> dict[str, float]:
    """Computes latency percentiles p50, p90, p95, p99 if at least 5 samples exist."""
    if len(durations_ms) < 5:
        return {}
    arr = np.array(durations_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }


def build_pipeline_from_preset(
    preset_name: str,
    base_settings: Settings | None = None,
    provider_factories: Any | None = None,
) -> RAGPipeline:
    """Instantiates a RAGPipeline wired according to a declared named preset."""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Allowed: {list(PRESET_CONFIGS.keys())}"
        )

    overrides = PRESET_CONFIGS[preset_name]
    settings = base_settings or get_settings()

    settings_dict = {
        k: getattr(settings, k)
        for k in settings.__dataclass_fields__
        if hasattr(settings, k)
    }
    settings_dict.update(overrides)
    new_settings = Settings(**settings_dict)

    return RAGPipeline(settings=new_settings, provider_factories=provider_factories)


class EvaluationRunner:
    """Executes evaluation across golden test cases and compiles deterministic RunArtifact."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        ks: Sequence[int] = (1, 3, 5, 10),
        judge: Any | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.ks = tuple(ks)
        self.judge = judge

    def run(
        self,
        cases: Sequence[GoldenCase],
        dataset_hash: str = "",
        run_id: str | None = None,
        timestamp: str | None = None,
    ) -> RunArtifact:
        start_time = time.perf_counter()
        actual_run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        actual_timestamp = timestamp or datetime.now(timezone.utc).isoformat()

        executed_cases: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        retrieval_metrics_all: list[RetrievalCaseMetrics] = []
        answer_metrics_all: list[AnswerCaseMetrics] = []
        citation_metrics_all: list[CitationCaseMetrics] = []

        retrieval_metrics_by_cat: dict[str, list[RetrievalCaseMetrics]] = {}
        answer_metrics_by_cat: dict[str, list[AnswerCaseMetrics]] = {}
        citation_metrics_by_cat: dict[str, list[CitationCaseMetrics]] = {}

        total_latencies: list[float] = []
        retrieval_latencies: list[float] = []
        generation_latencies: list[float] = []

        corpus_fp = ""
        index_fp = ""

        for case in cases:
            retrieval_metrics_by_cat.setdefault(case.category, [])
            answer_metrics_by_cat.setdefault(case.category, [])
            citation_metrics_by_cat.setdefault(case.category, [])

            chat_history = [
                ChatTurn(user=turn[0], assistant=turn[1])
                for turn in case.history
            ]

            try:
                answer_result, trace = self.pipeline.answer_with_trace(
                    question=case.question,
                    history=chat_history,
                    case_id=case.id,
                )
                if not corpus_fp:
                    corpus_fp = trace.corpus_fingerprint
                if not index_fp:
                    index_fp = trace.index_fingerprint

                ret_chunks = (
                    trace.reranked_results
                    if trace.reranked_results
                    else (trace.fused_results if trace.fused_results else trace.dense_results)
                )

                ret_metrics = evaluate_retrieval(case, ret_chunks, ks=self.ks)
                ans_metrics = evaluate_answer(case, answer_result)
                cit_metrics = evaluate_citations(case, answer_result, ret_chunks)

                retrieval_metrics_all.append(ret_metrics)
                answer_metrics_all.append(ans_metrics)
                citation_metrics_all.append(cit_metrics)

                retrieval_metrics_by_cat[case.category].append(ret_metrics)
                answer_metrics_by_cat[case.category].append(ans_metrics)
                citation_metrics_by_cat[case.category].append(cit_metrics)

                timings = dict(trace.timings_ms)
                if "total_ms" in timings:
                    total_latencies.append(timings["total_ms"])
                if "retrieval_ms" in timings:
                    retrieval_latencies.append(timings["retrieval_ms"])
                if "generation_ms" in timings and timings["generation_ms"] > 0:
                    generation_latencies.append(timings["generation_ms"])

                case_record = {
                    "case_id": case.id,
                    "category": case.category,
                    "question": case.question,
                    "answerable": case.answerable,
                    "retrieval_query": trace.retrieval_query,
                    "answer": answer_result.answer,
                    "decision": answer_result.decision,
                    "citations": [c.to_dict() for c in answer_result.citations],
                    "retrieval_metrics": ret_metrics.to_dict(),
                    "answer_metrics": ans_metrics.to_dict(),
                    "citation_metrics": cit_metrics.to_dict(),
                    "timings_ms": timings,
                }

                if self.judge is not None:
                    try:
                        judge_res = self.judge.judge(
                            case=case,
                            answer=answer_result.answer,
                            evidence_texts=[c.chunk.content for c in ret_chunks],
                        )
                        case_record["judge_metrics"] = judge_res.to_dict()
                    except Exception as j_exc:
                        LOGGER.warning("Judge evaluation failed for case %s: %s", case.id, j_exc)
                        case_record["judge_metrics"] = {"error": str(j_exc)}

                executed_cases.append(case_record)

            except Exception as exc:
                LOGGER.error("Evaluation execution failed for case %s: %s", case.id, exc)
                failures.append({
                    "case_id": case.id,
                    "category": case.category,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })

        # Category aggregates
        category_aggregates: dict[str, dict[str, Any]] = {}
        for cat in retrieval_metrics_by_cat:
            cat_ret = aggregate_retrieval_metrics(retrieval_metrics_by_cat[cat])
            cat_ans = aggregate_answer_metrics(answer_metrics_by_cat[cat])
            cat_cit = aggregate_citation_metrics(citation_metrics_by_cat[cat])
            category_aggregates[cat] = {
                "retrieval": cat_ret,
                "answers": cat_ans,
                "citations": cat_cit,
                "total_cases": len(retrieval_metrics_by_cat[cat]),
            }

        # Overall macro aggregates
        macro_aggregates = {
            "retrieval": aggregate_retrieval_metrics(retrieval_metrics_all),
            "answers": aggregate_answer_metrics(answer_metrics_all),
            "citations": aggregate_citation_metrics(citation_metrics_all),
            "total_executed_cases": len(executed_cases),
            "total_failed_cases": len(failures),
        }

        latencies = {
            "total": compute_latency_percentiles(total_latencies),
            "retrieval": compute_latency_percentiles(retrieval_latencies),
            "generation": compute_latency_percentiles(generation_latencies),
        }

        git_rev, git_dirty = get_git_metadata()
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        content = {
            "dataset_hash": dataset_hash,
            "corpus_fingerprint": corpus_fp or "unknown",
            "index_fingerprint": index_fp or "unknown",
            "configuration": {
                "RETRIEVAL_MODE": self.pipeline.settings.RETRIEVAL_MODE,
                "RERANK_ENABLED": self.pipeline.settings.RERANK_ENABLED,
                "RETRIEVAL_K": self.pipeline.settings.RETRIEVAL_K,
                "DENSE_CANDIDATE_K": self.pipeline.settings.DENSE_CANDIDATE_K,
                "SPARSE_CANDIDATE_K": self.pipeline.settings.SPARSE_CANDIDATE_K,
                "FUSION_CANDIDATE_K": self.pipeline.settings.FUSION_CANDIDATE_K,
                "CHUNK_SIZE": self.pipeline.settings.CHUNK_SIZE,
                "CHUNK_OVERLAP": self.pipeline.settings.CHUNK_OVERLAP,
                "LLM_MODEL": str(self.pipeline.settings.LLM_MODEL),
                "EMBEDDING_MODEL": str(self.pipeline.settings.EMBEDDING_MODEL),
            },
            "cases": executed_cases,
            "category_aggregates": category_aggregates,
            "macro_aggregates": macro_aggregates,
            "failures": failures,
            "latencies": latencies,
        }

        return RunArtifact.create(
            content=content,
            run_id=actual_run_id,
            timestamp=actual_timestamp,
            git_revision=git_rev,
            git_dirty=git_dirty,
            duration_ms=duration_ms,
            extra_metadata={"platform": platform.platform(), "python": platform.python_version()},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG Pipeline Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/datasets/cv_golden.jsonl",
        help="Path to golden JSONL dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hybrid-no-rerank",
        choices=["dense-baseline", "hybrid-no-rerank", "hybrid-rerank"],
        help="Named configuration preset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation run artifact JSON",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Enable optional LLM judge evaluation",
    )

    args = parser.parse_args()

    cases, dataset_hash = load_dataset(args.dataset)
    print(f"Loaded {len(cases)} test cases from {args.dataset} (hash={dataset_hash[:8]}...)")

    pipeline = build_pipeline_from_preset(args.config)
    print(f"Initialized pipeline with preset '{args.config}' (mode={pipeline.settings.RETRIEVAL_MODE}, rerank={pipeline.settings.RERANK_ENABLED})")

    judge = None
    if args.judge:
        from evaluation.judges.nvidia import NvidiaJudge
        judge = NvidiaJudge()
        print("Optional LLM judge enabled (NVIDIA)")

    runner = EvaluationRunner(pipeline=pipeline, judge=judge)
    artifact = runner.run(cases=cases, dataset_hash=dataset_hash)

    print("\n--- Evaluation Run Summary ---")
    print(f"Content Fingerprint: {artifact.content_fingerprint}")
    macro = artifact.content["macro_aggregates"]
    print(f"Executed Cases: {macro['total_executed_cases']}, Failures: {macro['total_failed_cases']}")

    ret = macro["retrieval"]
    if ret.get("applicable_cases", 0) > 0:
        print(f"Retrieval -> Hit@1: {ret.get('hit_at_1', 0.0):.3f}, Hit@5: {ret.get('hit_at_5', 0.0):.3f}, Recall@5: {ret.get('recall_at_5', 0.0):.3f}, MRR: {ret.get('mrr', 0.0):.3f}")

    ans = macro["answers"]
    print(f"Answers   -> Fact Coverage: {ans.get('required_fact_coverage', 0.0):.3f}, Correct Abstention: {ans.get('correct_abstention_rate', 0.0):.3f}")

    cit = macro["citations"]
    if cit.get("applicable_cases", 0) > 0:
        print(f"Citations -> ID Validity: {cit.get('citation_id_validity', 0.0):.3f}, Claim Coverage: {cit.get('claim_citation_coverage', 0.0):.3f}")

    if args.output:
        out_path = Path(args.output)
        artifact.save(out_path)
        print(f"\nSaved evaluation artifact to {out_path}")


if __name__ == "__main__":
    main()
