"""Answerability threshold calibration CLI and cross-validation pipeline (Spec 05)."""

from __future__ import annotations

import argparse
import logging

from evaluation.calibrate_abstention import (
    calibrate_abstention_policy,
)

LOGGER = logging.getLogger(__name__)

# Alias for Spec 05 naming convention
calibrate_answerability_policy = calibrate_abstention_policy


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
