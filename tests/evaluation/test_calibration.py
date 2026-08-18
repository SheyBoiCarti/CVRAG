import pytest
from evaluation.calibration import (
    stratified_k_fold_split,
    calibrate_answerability_threshold,
    compute_bootstrap_ci,
    evaluate_calibration_pool_and_locked_set,
    CalibrationOutcome,
)
from evaluation.schema import GoldenCase


def test_stratified_split_preserves_distribution():
    cases = []
    for i in range(25):
        cases.append(GoldenCase.from_dict({
            "id": f"sf-{i}", "category": "straightforward", "question": "q", "history": [],
            "answerable": True, "reference_answer": "ans", "required_facts": [], "forbidden_facts": [], "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "Wirk"}]],
        }))
        cases.append(GoldenCase.from_dict({
            "id": f"un-{i}", "category": "unanswerable", "question": "q", "history": [],
            "answerable": False, "reference_answer": "ans", "required_facts": [], "forbidden_facts": [], "evidence_groups": [],
        }))

    folds = stratified_k_fold_split(cases, n_splits=5, seed=42)
    assert len(folds) == 5

    for train_cases, val_cases in folds:
        assert len(train_cases) == 40
        assert len(val_cases) == 10
        val_unans = sum(1 for c in val_cases if not c.answerable)
        assert val_unans == 5


def test_threshold_calibration_cost_objective():
    # 5 answerable cases with scores [0.9, 0.8, 0.7, 0.4, 0.3]
    # 5 unanswerable cases with scores [0.6, 0.5, 0.3, 0.2, 0.1]
    case_scores = [
        (True, 0.9), (True, 0.8), (True, 0.7), (True, 0.4), (True, 0.3),
        (False, 0.6), (False, 0.5), (False, 0.3), (False, 0.2), (False, 0.1),
    ]

    outcome = calibrate_answerability_threshold(
        case_scores=case_scores,
        candidate_thresholds=[0.2, 0.4, 0.65, 0.85],
        cost_weights=(5.0, 1.0),
    )

    assert isinstance(outcome, CalibrationOutcome)
    # Threshold 0.65 yields:
    # Ans >= 0.65: [0.9, 0.8, 0.7] -> 3 answered, 2 false abstentions (0.4, 0.3) -> cost 2*1 = 2
    # Unans >= 0.65: None -> 0 false answers -> cost 0
    # Total cost = 2.
    assert outcome.optimal_threshold == pytest.approx(0.65)
    assert outcome.optimal_cost == pytest.approx(2.0)


def test_bootstrap_confidence_interval_deterministic():
    scores = [1.0] * 80 + [0.0] * 20
    low, high = compute_bootstrap_ci(scores, n_resamples=500, confidence=0.95, seed=42)
    assert 0.70 <= low <= 0.82
    assert 0.78 <= high <= 0.90
    assert low < high


def test_locked_set_partitioning_and_evaluation():
    cases = []
    # Build 40 answerable and 40 unanswerable (total 80)
    for i in range(40):
        cases.append(GoldenCase.from_dict({
            "id": f"ans-{i}", "category": "straightforward", "question": "q", "history": [],
            "answerable": True, "reference_answer": "ans", "required_facts": [], "forbidden_facts": [],
            "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "Wirk"}]],
        }))
        cases.append(GoldenCase.from_dict({
            "id": f"unans-{i}", "category": "unanswerable", "question": "q", "history": [],
            "answerable": False, "reference_answer": "ans", "required_facts": [], "forbidden_facts": [],
            "evidence_groups": [],
        }))

    cal_pool, locked_set = evaluate_calibration_pool_and_locked_set(cases, locked_fraction=0.25, seed=42)
    assert len(cal_pool) == 60
    assert len(locked_set) == 20
    # Ensure locked set has 10 answerable and 10 unanswerable
    assert sum(1 for c in locked_set if c.answerable) == 10
    assert sum(1 for c in locked_set if not c.answerable) == 10
