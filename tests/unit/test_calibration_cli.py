from __future__ import annotations

import json
from pathlib import Path

from evaluation.calibrate_abstention import calibrate_abstention_policy
from evaluation.calibrate_answerability import calibrate_answerability_policy
from evaluation.schema import GoldenCase
from rag.verification.answerability import load_calibration_policy
from rag.verification.models import CalibrationPolicyArtifact


def _sample_cases() -> list[GoldenCase]:
    cases = []
    # 20 answerable cases
    for i in range(20):
        cases.append(
            GoldenCase(
                id=f"ans_{i}",
                question=f"Query {i}",
                category="straightforward",
                answerable=True,
                tags=(f"score_{0.80 + (i % 5) * 0.03:.2f}",),
            )
        )
    # 10 unanswerable cases
    for i in range(10):
        cases.append(
            GoldenCase(
                id=f"unans_{i}",
                question=f"Unans Query {i}",
                category="unanswerable",
                answerable=False,
                tags=(f"score_{0.20 + (i % 4) * 0.05:.2f}",),
            )
        )
    return cases


def test_calibrate_abstention_policy(tmp_path: Path):
    dataset_file = tmp_path / "dataset.jsonl"
    cases = _sample_cases()
    with open(dataset_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c.to_dict()) + "\n" for c in cases)

    output_policy_file = tmp_path / "policy.json"

    artifact = calibrate_abstention_policy(
        dataset_path=dataset_file,
        signal_name="top_reranker_score",
        output_path=output_policy_file,
        score_extractor=lambda c: float(c.tags[0].split("_")[1]),
        seed=42,
    )

    assert isinstance(artifact, CalibrationPolicyArtifact)
    assert artifact.signal_name == "top_reranker_score"
    assert 0.3 <= artifact.threshold <= 0.8
    assert artifact.ci_lower <= artifact.threshold or artifact.ci_lower <= artifact.ci_upper
    assert output_policy_file.exists()

    # Test loading the saved policy
    loaded = load_calibration_policy(output_policy_file)
    assert loaded is not None
    assert loaded.policy_version == artifact.policy_version
    assert loaded.threshold == artifact.threshold


def test_calibrate_answerability_policy(tmp_path: Path):
    dataset_file = tmp_path / "dataset.jsonl"
    cases = _sample_cases()
    with open(dataset_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c.to_dict()) + "\n" for c in cases)

    output_policy_file = tmp_path / "policy_ans.json"

    artifact = calibrate_answerability_policy(
        dataset_path=dataset_file,
        signal_name="top_reranker_score",
        output_path=output_policy_file,
        score_extractor=lambda c: float(c.tags[0].split("_")[1]),
        seed=42,
    )

    assert isinstance(artifact, CalibrationPolicyArtifact)
    assert artifact.signal_name == "top_reranker_score"
    assert 0.3 <= artifact.threshold <= 0.8
    assert output_policy_file.exists()

