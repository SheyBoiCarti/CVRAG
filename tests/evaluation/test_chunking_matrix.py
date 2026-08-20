from __future__ import annotations

from pathlib import Path

from evaluation.chunking_matrix import (
    CHUNKING_MATRIX_CONFIGS,
    ChunkingMatrixRunner,
    generate_markdown_matrix_report,
)
from evaluation.runner import load_dataset
from rag.settings import Settings
from tests.fakes import FakeProviderFactories


def test_chunking_matrix_configs_defined() -> None:
    expected_keys = {
        "recursive_bullet_v1_700_140",
        "fixed_character_v1_500_100",
        "fixed_character_v1_700_140",
        "fixed_character_v1_1000_200",
        "structure_aware_v1_700_140",
        "semantic_v1_035_200_900_1",
    }
    assert set(CHUNKING_MATRIX_CONFIGS.keys()) == expected_keys

    for key, cfg in CHUNKING_MATRIX_CONFIGS.items():
        assert "CHUNK_STRATEGY" in cfg


def test_chunking_matrix_runner_single_config(tmp_path: Path) -> None:
    dataset_path = Path(__file__).resolve().parent.parent.parent / "evaluation" / "datasets" / "cv_golden.jsonl"
    cases, dataset_hash = load_dataset(dataset_path)

    # Use first 3 cases for fast test
    subset_cases = cases[:3]

    base_settings = Settings(
        VECTOR_DB_DIR=tmp_path / "vector_db",
        DOCUMENTS_DIR=tmp_path / "docs",
    )
    docs_dir = base_settings.DOCUMENTS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "cv.md").write_text("# John Doe\n\nSoftware Engineer with Python and RAG experience.\n", encoding="utf-8")

    runner = ChunkingMatrixRunner(
        base_settings=base_settings,
        cases=subset_cases,
        dataset_hash=dataset_hash,
        provider_factories=FakeProviderFactories(),
    )

    result = runner.evaluate_config("recursive_bullet_v1_700_140", CHUNKING_MATRIX_CONFIGS["recursive_bullet_v1_700_140"])
    assert "chunk_stats" in result
    assert "retrieval_macro" in result
    assert "build_duration_ms" in result
    assert result["chunk_stats"]["count"] > 0


def test_generate_markdown_matrix_report() -> None:
    mock_results = {
        "recursive_bullet_v1_700_140": {
            "strategy": "recursive_bullet_v1",
            "chunk_stats": {"count": 10, "mean_chars": 500.0, "p50_chars": 510.0, "p90_chars": 680.0},
            "build_duration_ms": 120.0,
            "retrieval_macro": {"mrr": 0.85, "hit_rate": 0.90, "all_evidence_recall": 0.80},
            "citation_macro": {"citation_f1": 0.92},
            "latencies": {"retrieval": {"p50": 15.0, "p90": 25.0}},
        },
        "fixed_character_v1_700_140": {
            "strategy": "fixed_character_v1",
            "chunk_stats": {"count": 12, "mean_chars": 700.0, "p50_chars": 700.0, "p90_chars": 700.0},
            "build_duration_ms": 110.0,
            "retrieval_macro": {"mrr": 0.80, "hit_rate": 0.88, "all_evidence_recall": 0.75},
            "citation_macro": {"citation_f1": 0.90},
            "latencies": {"retrieval": {"p50": 16.0, "p90": 26.0}},
        },
    }

    report = generate_markdown_matrix_report(mock_results)
    assert "# Chunking Experiment Matrix Report" in report
    assert "recursive_bullet_v1_700_140" in report
    assert "fixed_character_v1_700_140" in report
    assert "MRR" in report
