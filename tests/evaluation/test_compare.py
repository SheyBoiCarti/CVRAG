import json

import pytest

from evaluation.compare import ComparisonReport, compare_runs, load_artifact


def test_compare_detects_improvements_and_regressions():
    base_dict = {
        "content_fingerprint": "hash1",
        "content": {
            "category_aggregates": {
                "straightforward": {
                    "retrieval": {"hit_at_5": 0.8, "mrr": 0.7},
                    "answers": {"required_fact_coverage": 0.8, "correct_abstention_rate": 0.0},
                },
                "unanswerable": {
                    "retrieval": {"applicable_cases": 0},
                    "answers": {"correct_abstention_rate": 1.0, "false_answer_rate": 0.0},
                },
            },
            "macro_aggregates": {
                "retrieval": {"hit_at_5": 0.8, "mrr": 0.7},
                "answers": {"required_fact_coverage": 0.8, "correct_abstention_rate": 1.0},
                "citations": {"citation_id_validity": 1.0},
                "total_executed_cases": 2,
                "total_failed_cases": 0,
            },
            "cases": [
                {
                    "case_id": "c1",
                    "category": "straightforward",
                    "retrieval_metrics": {"hit_at_5": 0.0, "mrr": 0.0},
                    "answer_metrics": {"required_fact_coverage": 0.0},
                },
                {
                    "case_id": "u1",
                    "category": "unanswerable",
                    "retrieval_metrics": {"is_applicable": False},
                    "answer_metrics": {"correct_abstention": 1.0},
                },
            ],
        },
        "run_metadata": {"run_id": "r1"},
    }

    cand_dict = {
        "content_fingerprint": "hash2",
        "content": {
            "category_aggregates": {
                "straightforward": {
                    "retrieval": {"hit_at_5": 1.0, "mrr": 1.0},
                    "answers": {"required_fact_coverage": 1.0, "correct_abstention_rate": 0.0},
                },
                "unanswerable": {
                    "retrieval": {"applicable_cases": 0},
                    "answers": {"correct_abstention_rate": 0.9, "false_answer_rate": 0.1},
                },
            },
            "macro_aggregates": {
                "retrieval": {"hit_at_5": 1.0, "mrr": 1.0},
                "answers": {"required_fact_coverage": 1.0, "correct_abstention_rate": 0.9},
                "citations": {"citation_id_validity": 1.0},
                "total_executed_cases": 2,
                "total_failed_cases": 0,
            },
            "cases": [
                {
                    "case_id": "c1",
                    "category": "straightforward",
                    "retrieval_metrics": {"hit_at_5": 1.0, "mrr": 1.0},
                    "answer_metrics": {"required_fact_coverage": 1.0},
                },
                {
                    "case_id": "u1",
                    "category": "unanswerable",
                    "retrieval_metrics": {"is_applicable": False},
                    "answer_metrics": {"correct_abstention": 0.9},
                },
            ],
        },
        "run_metadata": {"run_id": "r2"},
    }

    report = compare_runs(base_dict, cand_dict)

    assert isinstance(report, ComparisonReport)
    assert report.macro_deltas["retrieval"]["hit_at_5"] == pytest.approx(0.2)
    assert report.macro_deltas["retrieval"]["mrr"] == pytest.approx(0.3)
    assert report.macro_deltas["answers"]["required_fact_coverage"] == pytest.approx(0.2)
    assert report.macro_deltas["answers"]["correct_abstention_rate"] == pytest.approx(-0.1)

    assert "c1" in report.improved_case_ids
    assert "u1" in report.regressed_case_ids
    assert len(report.missing_in_candidate) == 0
    assert len(report.added_in_candidate) == 0


def test_load_artifact(tmp_path):
    artifact_data = {
        "content_fingerprint": "hash123",
        "content": {"cases": []},
        "run_metadata": {"run_id": "r1"},
    }
    file_path = tmp_path / "art.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(artifact_data, f)

    loaded = load_artifact(file_path)
    assert loaded["content_fingerprint"] == "hash123"
