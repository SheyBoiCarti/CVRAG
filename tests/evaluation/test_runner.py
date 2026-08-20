import json
from pathlib import Path

from evaluation.runner import EvaluationRunner, build_pipeline_from_preset, load_dataset
from evaluation.schema import GoldenCase, RunArtifact, compute_content_hash
from rag.pipeline import RAGPipeline
from rag.settings import Settings
from tests.fakes import FakeChatModel, FakeEmbeddings, FakeProviderFactories


def test_runner_executes_cases_and_builds_artifact(tmp_path):
    cases = [
        GoldenCase.from_dict({
            "id": "sf-001",
            "category": "straightforward",
            "question": "What is Sheharyar's role at Wirk?",
            "history": [],
            "answerable": True,
            "reference_answer": "Software Engineer & Game Developer",
            "required_facts": [{"matcher": "token_phrase", "value": "Software Engineer", "aliases": []}],
            "forbidden_facts": [],
            "evidence_groups": [[{"source": "Custom_CV.md", "page": 1, "anchor": "Wirk"}]],
        }),
        GoldenCase.from_dict({
            "id": "un-001",
            "category": "unanswerable",
            "question": "What is his phone number?",
            "history": [],
            "answerable": False,
            "reference_answer": "I do not have access to that information.",
            "required_facts": [],
            "forbidden_facts": [],
            "evidence_groups": [],
        }),
    ]

    factories = FakeProviderFactories(
        embeddings=FakeEmbeddings(),
        chat_model=FakeChatModel(
            responses=[
                "Sheharyar works as a Software Engineer at Wirk [S1].",
                "I couldn't find that information in the CV.",
            ]
        ),
    )
    pipeline = RAGPipeline(settings=Settings(), provider_factories=factories)
    runner = EvaluationRunner(pipeline=pipeline)

    artifact = runner.run(
        cases=cases,
        dataset_hash="dataset-hash-123",
        run_id="test-run-id",
        timestamp="2026-08-18T12:00:00Z",
    )

    assert isinstance(artifact, RunArtifact)
    assert artifact.run_metadata["run_id"] == "test-run-id"
    assert artifact.run_metadata["timestamp"] == "2026-08-18T12:00:00Z"
    assert artifact.content["dataset_hash"] == "dataset-hash-123"
    assert len(artifact.content["cases"]) == 2
    assert "straightforward" in artifact.content["category_aggregates"]
    assert "unanswerable" in artifact.content["category_aggregates"]
    assert "macro_aggregates" in artifact.content
    assert artifact.content_fingerprint == compute_content_hash(artifact.content)

    # Test artifact save and roundtrip
    out_file = tmp_path / "baseline.json"
    artifact.save(out_file)
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["content_fingerprint"] == artifact.content_fingerprint


def test_build_pipeline_from_preset():
    pipe_dense = build_pipeline_from_preset("dense-baseline")
    assert pipe_dense.settings.RETRIEVAL_MODE == "dense"
    assert pipe_dense.settings.RERANK_ENABLED is False

    pipe_hybrid = build_pipeline_from_preset("hybrid-no-rerank")
    assert pipe_hybrid.settings.RETRIEVAL_MODE == "hybrid"
    assert pipe_hybrid.settings.RERANK_ENABLED is False

    pipe_rerank = build_pipeline_from_preset("hybrid-rerank")
    assert pipe_rerank.settings.RETRIEVAL_MODE == "hybrid"
    assert pipe_rerank.settings.RERANK_ENABLED is True


def test_load_dataset():
    dataset_path = Path(__file__).parents[2] / "evaluation" / "datasets" / "cv_golden.jsonl"
    cases, d_hash = load_dataset(dataset_path)
    assert len(cases) >= 60
    assert d_hash != ""
