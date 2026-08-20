from pathlib import Path

from evaluation.runner import EvaluationRunner, load_dataset
from evaluation.schema import RunArtifact
from rag.pipeline import RAGPipeline
from rag.settings import Settings
from tests.fakes import FakeChatModel, FakeEmbeddings, FakeProviderFactories

DATASET_PATH = Path(__file__).parents[2] / "evaluation" / "datasets" / "cv_golden.jsonl"


def test_full_offline_evaluation_run_on_golden_dataset(tmp_path):
    cases, d_hash = load_dataset(DATASET_PATH)
    assert len(cases) >= 60

    factories = FakeProviderFactories(
        embeddings=FakeEmbeddings(),
        chat_model=FakeChatModel(
            default_response="Sheharyar worked at Wirk (Pty) Ltd as a Software Engineer [S1]."
        ),
    )
    settings = Settings(
        RETRIEVAL_MODE="hybrid",
        RERANK_ENABLED=False,
    )
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)
    runner = EvaluationRunner(pipeline=pipeline)

    artifact = runner.run(
        cases=cases,
        dataset_hash=d_hash,
        run_id="offline-integration-run",
        timestamp="2026-08-18T12:00:00Z",
    )

    assert isinstance(artifact, RunArtifact)
    assert len(artifact.content["cases"]) == len(cases)
    assert artifact.content["macro_aggregates"]["total_failed_cases"] == 0

    # Verify all 6 categories present in aggregates
    for cat in ["straightforward", "exact_term", "multi_hop", "conversational_follow_up", "ambiguous", "unanswerable"]:
        assert cat in artifact.content["category_aggregates"]

    # Test baseline artifact save
    out_file = tmp_path / "baseline_integration.json"
    artifact.save(out_file)
    assert out_file.exists()


def test_offline_runner_preset_execution():
    cases, d_hash = load_dataset(DATASET_PATH)
    # Take a small subset for speed in unit run
    subset = cases[:6]

    factories = FakeProviderFactories(
        embeddings=FakeEmbeddings(),
        chat_model=FakeChatModel(default_response="Answer with citations [S1]."),
    )
    settings = Settings(
        RETRIEVAL_MODE="dense",
        RERANK_ENABLED=False,
    )
    pipeline = RAGPipeline(settings=settings, provider_factories=factories)
    runner = EvaluationRunner(pipeline=pipeline)

    artifact = runner.run(
        cases=subset,
        dataset_hash=d_hash,
        run_id="preset-dense-run",
        timestamp="2026-08-18T12:00:00Z",
    )

    assert len(artifact.content["cases"]) == 6
    assert artifact.content["configuration"]["RETRIEVAL_MODE"] == "dense"
