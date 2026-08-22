"""Answerability calibration protocol, stratified cross-validation, and bootstrap analysis."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluation.schema import GoldenCase


@dataclass(frozen=True)
class CalibrationOutcome:
    """Outcome of threshold calibration optimization."""

    optimal_threshold: float
    optimal_cost: float
    false_answer_count: int
    false_abstention_count: int
    false_answer_rate: float
    false_abstention_rate: float
    threshold_costs: Mapping[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimal_threshold": self.optimal_threshold,
            "optimal_cost": self.optimal_cost,
            "false_answer_count": self.false_answer_count,
            "false_abstention_count": self.false_abstention_count,
            "false_answer_rate": self.false_answer_rate,
            "false_abstention_rate": self.false_abstention_rate,
            "threshold_costs": {str(k): v for k, v in self.threshold_costs.items()},
        }


def stratified_k_fold_split(
    cases: Sequence[GoldenCase],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[GoldenCase], list[GoldenCase]]]:
    """Partitions cases into N stratified folds preserving category and answerability balance."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    # Group cases by (category, answerable) strata
    strata: dict[tuple[str, bool], list[GoldenCase]] = {}
    for case in cases:
        key = (case.category, case.answerable)
        strata.setdefault(key, []).append(case)

    rng = random.Random(seed)
    folds: list[list[GoldenCase]] = [[] for _ in range(n_splits)]

    for group in strata.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        for idx, item in enumerate(shuffled):
            folds[idx % n_splits].append(item)

    splits: list[tuple[list[GoldenCase], list[GoldenCase]]] = []
    for i in range(n_splits):
        val_set = folds[i]
        train_set = [item for j, f in enumerate(folds) if j != i for item in f]
        splits.append((train_set, val_set))

    return splits


def calibrate_answerability_threshold(
    case_scores: Sequence[tuple[bool, float]],
    candidate_thresholds: Sequence[float] | None = None,
    cost_weights: tuple[float, float] = (5.0, 1.0),
) -> CalibrationOutcome:
    """Selects the cost-optimal confidence threshold for answerability.

    Cost = cost_weights[0] * false_answer + cost_weights[1] * false_abstention.
    Ties choose the more conservative (higher) threshold.
    """
    w_false_ans, w_false_abs = cost_weights
    if not candidate_thresholds:
        candidate_thresholds = np.linspace(0.0, 1.0, 101).tolist()

    best_thresh = candidate_thresholds[0]
    best_cost = float("inf")
    best_fa = 0
    best_fab = 0
    costs_by_thresh: dict[float, float] = {}

    total_ans = sum(1 for is_ans, _ in case_scores if is_ans)
    total_unans = sum(1 for is_ans, _ in case_scores if not is_ans)

    # Sort candidate thresholds ascending
    for thresh in sorted(candidate_thresholds):
        # Decision: Answer if score >= thresh, Abstain if score < thresh
        false_answers = 0
        false_abstentions = 0
        for is_answerable, score in case_scores:
            predicted_answer = score >= thresh
            if not is_answerable and predicted_answer:
                false_answers += 1
            elif is_answerable and not predicted_answer:
                false_abstentions += 1

        cost = (w_false_ans * false_answers) + (w_false_abs * false_abstentions)
        costs_by_thresh[thresh] = cost

        # If cost is <= best_cost, select higher threshold (conservative tie-break)
        if cost <= best_cost:
            best_cost = cost
            best_thresh = thresh
            best_fa = false_answers
            best_fab = false_abstentions

    fa_rate = (best_fa / total_unans) if total_unans > 0 else 0.0
    fab_rate = (best_fab / total_ans) if total_ans > 0 else 0.0

    return CalibrationOutcome(
        optimal_threshold=best_thresh,
        optimal_cost=best_cost,
        false_answer_count=best_fa,
        false_abstention_count=best_fab,
        false_answer_rate=fa_rate,
        false_abstention_rate=fab_rate,
        threshold_costs=costs_by_thresh,
    )


def compute_bootstrap_ci(
    scores: Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes non-parametric bootstrap confidence interval with fixed random seed."""
    if not scores:
        return 0.0, 0.0

    arr = np.array(scores, dtype=float)
    n = len(arr)
    rng = np.random.RandomState(seed)

    bootstrap_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - (alpha / 2.0)) * 100.0

    ci_lower = float(np.percentile(bootstrap_means, lower_pct))
    ci_upper = float(np.percentile(bootstrap_means, upper_pct))
    return ci_lower, ci_upper


def evaluate_calibration_pool_and_locked_set(
    cases: Sequence[GoldenCase],
    locked_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[GoldenCase], list[GoldenCase]]:
    """Splits full dataset into a calibration pool and a held-out locked regression set."""
    strata: dict[tuple[str, bool], list[GoldenCase]] = {}
    for case in cases:
        key = (case.category, case.answerable)
        strata.setdefault(key, []).append(case)

    rng = random.Random(seed)
    cal_pool: list[GoldenCase] = []
    locked_set: list[GoldenCase] = []

    for group in strata.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_locked = int(len(shuffled) * locked_fraction)
        locked_set.extend(shuffled[:n_locked])
        cal_pool.extend(shuffled[n_locked:])

    return cal_pool, locked_set


__all__ = [
    "CalibrationOutcome",
    "calibrate_answerability_threshold",
    "compute_bootstrap_ci",
    "evaluate_calibration_pool_and_locked_set",
    "stratified_k_fold_split",
]
