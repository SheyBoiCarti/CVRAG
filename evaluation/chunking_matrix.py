"""Chunking experiment matrix runner evaluating strategies across golden dataset."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.judges.base import AnswerJudge
from evaluation.runner import EvaluationRunner, load_dataset
from evaluation.schema import GoldenCase, canonical_json_dumps
from rag.pipeline import RAGPipeline
from rag.provider_factories import ProviderFactories
from rag.settings import Settings

LOGGER = logging.getLogger(__name__)

CHUNKING_MATRIX_CONFIGS: dict[str, dict[str, Any]] = {
    "recursive_bullet_v1_700_140": {
        "CHUNK_STRATEGY": "recursive_bullet_v1",
        "CHUNK_SIZE": 700,
        "CHUNK_OVERLAP": 140,
    },
    "fixed_character_v1_500_100": {
        "CHUNK_STRATEGY": "fixed_character_v1",
        "CHUNK_SIZE": 500,
        "CHUNK_OVERLAP": 100,
    },
    "fixed_character_v1_700_140": {
        "CHUNK_STRATEGY": "fixed_character_v1",
        "CHUNK_SIZE": 700,
        "CHUNK_OVERLAP": 140,
    },
    "fixed_character_v1_1000_200": {
        "CHUNK_STRATEGY": "fixed_character_v1",
        "CHUNK_SIZE": 1000,
        "CHUNK_OVERLAP": 200,
    },
    "structure_aware_v1_700_140": {
        "CHUNK_STRATEGY": "structure_aware_v1",
        "STRUCTURE_MAX_CHARS": 700,
        "STRUCTURE_OVERLAP_CHARS": 140,
    },
    "semantic_v1_035_200_900_1": {
        "CHUNK_STRATEGY": "semantic_v1",
        "SEMANTIC_BREAK_THRESHOLD": 0.35,
        "SEMANTIC_MIN_CHARS": 200,
        "SEMANTIC_MAX_CHARS": 900,
        "SEMANTIC_OVERLAP_UNITS": 1,
    },
}


class ChunkingMatrixRunner:
    """Orchestrates multi-strategy chunking experiments and metrics comparison."""

    def __init__(
        self,
        base_settings: Settings,
        cases: list[GoldenCase],
        dataset_hash: str,
        provider_factories: ProviderFactories | None = None,
        judge: AnswerJudge | None = None,
    ) -> None:
        self.base_settings = base_settings
        self.cases = cases
        self.dataset_hash = dataset_hash
        self.provider_factories = provider_factories
        self.judge = judge

    def evaluate_config(self, config_name: str, config_params: Mapping[str, Any]) -> dict[str, Any]:
        """Builds index and evaluates a single chunking strategy configuration."""
        LOGGER.info("Evaluating chunking configuration: %s", config_name)
        cfg_settings = replace(self.base_settings, **config_params)

        from rag.ingestion.indexer import IndexSynchronizer
        from rag.provider_factories import LiveProviderFactories

        factories = self.provider_factories or LiveProviderFactories()
        embeddings = factories.create_embeddings(cfg_settings)
        indexer = IndexSynchronizer(settings=cfg_settings, embeddings=embeddings)

        t0 = time.perf_counter()
        active_index = indexer.ensure_active_index()
        build_duration_ms = (time.perf_counter() - t0) * 1000.0

        pipeline = RAGPipeline(
            settings=cfg_settings,
            provider_factories=self.provider_factories,
        )

        chunks = active_index.chunks
        lengths = [len(c.content) for c in chunks] if chunks else [0]
        chunk_stats = {
            "count": len(chunks),
            "min_chars": int(np.min(lengths)),
            "max_chars": int(np.max(lengths)),
            "mean_chars": float(np.mean(lengths)),
            "p50_chars": float(np.percentile(lengths, 50)),
            "p90_chars": float(np.percentile(lengths, 90)),
        }

        eval_runner = EvaluationRunner(
            pipeline=pipeline,
            judge=self.judge,
        )
        artifact = eval_runner.run(cases=self.cases, dataset_hash=self.dataset_hash)

        return {
            "config_name": config_name,
            "strategy": cfg_settings.CHUNK_STRATEGY,
            "configuration": dict(config_params),
            "effective_parameters": cfg_settings.chunk_parameters(),
            "build_duration_ms": round(build_duration_ms, 2),
            "chunk_stats": chunk_stats,
            "retrieval_macro": artifact.content["macro_aggregates"]["retrieval"],
            "retrieval_by_category": {
                cat: data["retrieval"] for cat, data in artifact.content["category_aggregates"].items()
            },
            "answer_macro": artifact.content["macro_aggregates"]["answers"],
            "citation_macro": artifact.content["macro_aggregates"]["citations"],
            "latencies": artifact.content["latencies"],
            "total_cases": artifact.content["macro_aggregates"]["total_executed_cases"],
        }

    def run_matrix(self) -> dict[str, Any]:
        """Runs all chunking matrix configurations and collects comprehensive results."""
        results: dict[str, Any] = {}
        for name, config_params in CHUNKING_MATRIX_CONFIGS.items():
            results[name] = self.evaluate_config(name, config_params)
        return results


def generate_markdown_matrix_report(results: Mapping[str, Any]) -> str:
    """Generates a Markdown comparison table and analysis from matrix results."""
    lines: list[str] = [
        "# Chunking Experiment Matrix Report",
        "",
        "## 1. Summary Comparison",
        "",
        "| Configuration | Strategy | Chunks | Mean Len | Build (ms) | MRR | Hit Rate | All-Evidence Recall | Cit F1 | Ret Lat p50 (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for name, res in results.items():
        stats = res.get("chunk_stats", {})
        ret = res.get("retrieval_macro", {})
        cit = res.get("citation_macro", {})
        lat = res.get("latencies", {}).get("retrieval", {})

        count = stats.get("count", 0)
        mean_len = round(stats.get("mean_chars", 0.0), 1)
        build_ms = round(res.get("build_duration_ms", 0.0), 1)
        mrr = round(ret.get("mrr", 0.0), 4)
        hit_rate = round(ret.get("hit_rate", 0.0), 4)
        all_recall = round(ret.get("all_evidence_recall", 0.0), 4)
        cit_f1 = round(cit.get("citation_f1", 0.0), 4)
        lat_p50 = round(lat.get("p50", 0.0), 1)

        lines.append(
            f"| `{name}` | {res.get('strategy', '')} | {count} | {mean_len} | {build_ms} | {mrr} | {hit_rate} | {all_recall} | {cit_f1} | {lat_p50} |"
        )

    lines.extend([
        "",
        "## 2. Chunk Distribution Analysis",
        "",
        "| Configuration | Chunk Count | Min Chars | Max Chars | Mean Chars | p50 Chars | p90 Chars |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for name, res in results.items():
        stats = res.get("chunk_stats", {})
        lines.append(
            f"| `{name}` | {stats.get('count', 0)} | {stats.get('min_chars', 0)} | {stats.get('max_chars', 0)} | {round(stats.get('mean_chars', 0.0), 1)} | {round(stats.get('p50_chars', 0.0), 1)} | {round(stats.get('p90_chars', 0.0), 1)} |"
        )

    lines.extend([
        "",
        "## 3. Strategy Selection Decision & Recommendation",
        "",
        "- **Selected Default Strategy:** `recursive_bullet_v1` (`chunk_size=700`, `chunk_overlap=140`)",
        "- **Decision Rationale:**",
        "  1. `recursive_bullet_v1` provides guaranteed backward compatibility and deterministic chunk boundaries matching established golden anchors.",
        "  2. `structure_aware_v1` produces granular section-bounded chunks and serves as an effective candidate for hierarchical/structured markdown retrieval.",
        "  3. `semantic_v1` provides adaptive embedding-distance boundary segmentation while isolating non-deterministic variations.",
        "  4. Under the Spec 04 promotion gate criteria, `recursive_bullet_v1` remains the default baseline until live hybrid+reranker cross-validation is completed.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chunking Experiment Matrix")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/datasets/cv_golden.jsonl",
        help="Path to golden JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/baselines",
        help="Directory to save experiment matrix JSON and Markdown report",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Use deterministic offline fakes for execution",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    cases, dataset_hash = load_dataset(dataset_path)

    base_settings = Settings(
        RETRIEVAL_MODE="hybrid",
        RERANK_ENABLED=False,
    )
    from tests.fakes import FakeProviderFactories

    runner = ChunkingMatrixRunner(
        base_settings=base_settings,
        cases=cases,
        dataset_hash=dataset_hash,
        provider_factories=FakeProviderFactories() if args.offline else None,
    )

    results = runner.run_matrix()
    report_md = generate_markdown_matrix_report(results)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "chunking_matrix_results.json"
    md_path = out_dir / "chunking_experiment_report.md"

    json_path.write_text(canonical_json_dumps(results), encoding="utf-8")
    md_path.write_text(report_md, encoding="utf-8")
    print(f"Matrix evaluation complete. Results written to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
