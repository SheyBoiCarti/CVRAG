import json
from pathlib import Path

import pytest

from evaluation.schema import GoldenCase

DATASET_PATH = Path(__file__).parents[2] / "evaluation" / "datasets" / "cv_golden.jsonl"


def test_cv_golden_dataset_parses_and_validates():
    assert DATASET_PATH.exists(), f"Dataset file missing at {DATASET_PATH}"
    cases: list[GoldenCase] = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                case = GoldenCase.from_dict(data)
                cases.append(case)
            except Exception as exc:
                pytest.fail(f"Failed parsing line {line_num} in cv_golden.jsonl: {exc}")

    assert len(cases) >= 60, f"Expected at least 60 cases, got {len(cases)}"

    categories: dict[str, int] = {}
    answerable_count = 0
    unanswerable_count = 0
    ids: set[str] = set()

    for case in cases:
        assert case.id not in ids, f"Duplicate ID '{case.id}'"
        ids.add(case.id)
        categories[case.category] = categories.get(case.category, 0) + 1
        if case.answerable:
            answerable_count += 1
            assert len(case.evidence_groups) >= 1, f"Answerable case {case.id} must have at least 1 evidence group"
        else:
            unanswerable_count += 1
            assert len(case.evidence_groups) == 0, f"Unanswerable case {case.id} must have 0 evidence groups"

    required_categories = [
        "straightforward",
        "exact_term",
        "multi_hop",
        "conversational_follow_up",
        "ambiguous",
        "unanswerable",
    ]
    for req_cat in required_categories:
        count = categories.get(req_cat, 0)
        assert count >= 8, f"Category '{req_cat}' has {count} cases (< 8 required)"

    assert answerable_count >= 30, f"Need at least 30 answerable cases, got {answerable_count}"
    assert unanswerable_count >= 30, f"Need at least 30 unanswerable cases, got {unanswerable_count}"
