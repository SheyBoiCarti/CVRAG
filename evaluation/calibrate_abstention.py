"""Answerability threshold calibration CLI and cross-validation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence
import uuid

import numpy as np

from evaluation.calibration import (
    calibrate_answerability_threshold,
    compute_bootstrap_ci,
    evaluate_calibration_pool_and_locked_set,
    stratified_k_fold_split,
)
from evaluation.schema import GoldenCase
from rag.verification.answerability import save_calibration_policy
from rag.verification.models import CalibrationPolicyArtifact

LOGGER = logging.getLogger(__name__)


def _compute_dataset_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of the dataset file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def calibrate_abstention_policy(
    dataset_path: str | Path,
    signal_name: str = "top_reranker_score",
    output_path: str | Path | None = None,
    score_extractor: Callable[[GoldenCase], float] | None = None,
    cost_weights: tuple[float, float] = (5.0, 1.0),
    n_splits: int = 5,
    locked_fraction: float = 0.25,
    n_bootstrap: int = 1000,
    model_name: str = "nvidia/llama-nemotron-rerank-1b-v2",
    corpus_fingerprint: str = "default-corpus",
    index_fingerprint: str = "default-index",
    seed: int = 42,
) -> CalibrationPolicyArtifact:
    """Runs stratified 5-fold cross-validation and 1000-sample bootstrap analysis to calibrate answerability threshold."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dataset_hash = _compute_dataset_hash(path)

    # Load cases
    cases: list[GoldenCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                cases.append(GoldenCase.from_dict(json.loads(line_str)))

    if not cases:
        raise ValueError("Golden evaluation dataset is empty.")

    # Default score extractor if none provided
    def _default_extractor(case: GoldenCase) -> float:
        meta = case.metadata or {}
        if signal_name in meta:
            return float(meta[signal_name])
        return 1.0 if case.answerable else 0.0

    extractor = score_extractor or _default_extractor

    # 1. Split calibration pool and locked regression holdout set
    cal_pool, locked_set = evaluate_calibration_pool_and_locked_set(
        cases,
        locked_fraction=locked_fraction,
        seed=seed,
    )

    # 2. Stratified K-Fold on calibration pool
    splits = stratified_k_fold_split(cal_pool, n_splits=n_splits, seed=seed)
    fold_thresholds: list[float] = []
    fold_costs: list[float] = []

    for train_cases, val_cases in splits:
        train_scores = [(c.answerable, extractor(c)) for c in train_cases]
        val_scores = [(c.answerable, extractor(c)) for c in val_cases]

        train_outcome = calibrate_answerability_threshold(
            train_scores,
            cost_weights=cost_weights,
        )
        fold_thresholds.append(train_outcome.optimal_threshold)

        # Evaluate on validation fold
        val_outcome = calibrate_answerability_threshold(
            val_scores,
            candidate_thresholds=[train_outcome.optimal_threshold],
            cost_weights=cost_weights,
        )
        fold_costs.append(val_outcome.optimal_cost)

    # 3. Overall calibration on full pool
    pool_scores = [(c.answerable, extractor(c)) for c in cal_pool]
    pool_outcome = calibrate_answerability_threshold(
        pool_scores,
        cost_weights=cost_weights,
    )

    # 4. Bootstrap confidence intervals on validation costs and scores
    ci_lower, ci_upper = compute_bootstrap_ci(
        fold_thresholds,
        n_resamples=n_bootstrap,
        confidence=0.95,
        seed=seed,
    )

    # 5. Evaluate locked regression set
    locked_scores = [(c.answerable, extractor(c)) for c in locked_set]
    locked_outcome = calibrate_answerability_threshold(
        locked_scores,
        candidate_thresholds=[pool_outcome.optimal_threshold],
        cost_weights=cost_weights,
    )

    policy_version = f"calibrated-{signal_name}-{dataset_hash}"
    artifact = CalibrationPolicyArtifact(
        policy_version=policy_version,
        dataset_hash=dataset_hash,
        corpus_fingerprint=corpus_fingerprint,
        index_fingerprint=index_fingerprint,
        model_name=model_name,
        signal_name=signal_name,
        threshold=pool_outcome.optimal_threshold,
        cost_objective=pool_outcome.optimal_cost,
        false_answer_rate=pool_outcome.false_answer_rate,
        false_abstention_rate=pool_outcome.false_abstention_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        locked_set_evaluated=bool(locked_set),
        metadata={
            "fold_thresholds": fold_thresholds,
            "fold_costs": fold_costs,
            "locked_cost": locked_outcome.optimal_cost,
            "locked_false_answer_count": locked_outcome.false_answer_count,
            "locked_false_abstention_count": locked_outcome.false_abstention_count,
        },
    )

    if output_path is not None:
        save_calibration_policy(artifact, output_path)

    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate answerability threshold on golden dataset.")
    parser.add_argument("--dataset", type=str, default="evaluation/datasets/cv_golden.jsonl", help="Path to golden dataset JSONL")
    parser.add_argument("--signal", type=str, default="top_reranker_score", help="Candidate signal name to calibrate")
    parser.add_argument("--output", type=str, default="artifacts/calibration_policy.json", help="Path to write policy JSON artifact")
    parser.add_argument("--model", type=str, default="nvidia/llama-nemotron-rerank-1b-v2", help="Model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Starting threshold calibration on %s for signal '%s'", args.dataset, args.signal)

    artifact = calibrate_abstention_policy(
        dataset_path=args.dataset,
        signal_name=args.signal,
        output_path=args.output,
        model_name=args.model,
        seed=args.seed,
    )
    print(f"Calibration completed! Optimal threshold: {artifact.threshold:.4f} (95% CI: [{artifact.ci_lower:.4f}, {artifact.ci_upper:.4f}])")
    print(f"Policy artifact saved to: {args.output}")


if __name__ == "__main__":
    main()
