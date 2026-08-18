"""Evaluation comparison tool analyzing performance deltas and regressions between runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_artifact(artifact_path: str | Path) -> dict[str, Any]:
    """Loads an evaluation run artifact JSON file."""
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation artifact not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_dict_deltas(base_dict: Mapping[str, Any], cand_dict: Mapping[str, Any]) -> dict[str, float]:
    """Computes candidate - baseline numerical differences for matching keys."""
    deltas: dict[str, float] = {}
    all_keys = set(base_dict.keys()) | set(cand_dict.keys())
    for k in all_keys:
        b_val = base_dict.get(k)
        c_val = cand_dict.get(k)
        if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
            deltas[k] = round(float(c_val - b_val), 4)
    return deltas


@dataclass(frozen=True)
class ComparisonReport:
    """Detailed comparison between a baseline run and a candidate run."""

    base_fingerprint: str
    cand_fingerprint: str
    macro_deltas: Mapping[str, dict[str, float]]
    category_deltas: Mapping[str, dict[str, dict[str, float]]]
    improved_case_ids: tuple[str, ...]
    regressed_case_ids: tuple[str, ...]
    unchanged_case_ids: tuple[str, ...]
    missing_in_candidate: tuple[str, ...]
    added_in_candidate: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_fingerprint": self.base_fingerprint,
            "cand_fingerprint": self.cand_fingerprint,
            "macro_deltas": dict(self.macro_deltas),
            "category_deltas": dict(self.category_deltas),
            "improved_cases_count": len(self.improved_case_ids),
            "regressed_cases_count": len(self.regressed_case_ids),
            "unchanged_cases_count": len(self.unchanged_case_ids),
            "missing_in_candidate": list(self.missing_in_candidate),
            "added_in_candidate": list(self.added_in_candidate),
            "improved_case_ids": list(self.improved_case_ids),
            "regressed_case_ids": list(self.regressed_case_ids),
        }


def compare_runs(
    base_artifact: Mapping[str, Any],
    cand_artifact: Mapping[str, Any],
) -> ComparisonReport:
    """Compares baseline and candidate evaluation artifacts."""
    base_fp = base_artifact.get("content_fingerprint", "unknown")
    cand_fp = cand_artifact.get("content_fingerprint", "unknown")

    base_content = base_artifact.get("content", {})
    cand_content = cand_artifact.get("content", {})

    base_macro = base_content.get("macro_aggregates", {})
    cand_macro = cand_content.get("macro_aggregates", {})

    macro_deltas = {
        "retrieval": _compute_dict_deltas(base_macro.get("retrieval", {}), cand_macro.get("retrieval", {})),
        "answers": _compute_dict_deltas(base_macro.get("answers", {}), cand_macro.get("answers", {})),
        "citations": _compute_dict_deltas(base_macro.get("citations", {}), cand_macro.get("citations", {})),
    }

    # Category deltas
    base_cats = base_content.get("category_aggregates", {})
    cand_cats = cand_content.get("category_aggregates", {})
    all_cats = sorted(set(base_cats.keys()) | set(cand_cats.keys()))

    category_deltas: dict[str, dict[str, dict[str, float]]] = {}
    for cat in all_cats:
        b_cat = base_cats.get(cat, {})
        c_cat = cand_cats.get(cat, {})
        category_deltas[cat] = {
            "retrieval": _compute_dict_deltas(b_cat.get("retrieval", {}), c_cat.get("retrieval", {})),
            "answers": _compute_dict_deltas(b_cat.get("answers", {}), c_cat.get("answers", {})),
            "citations": _compute_dict_deltas(b_cat.get("citations", {}), c_cat.get("citations", {})),
        }

    # Per-case comparison
    base_cases = {c["case_id"]: c for c in base_content.get("cases", [])}
    cand_cases = {c["case_id"]: c for c in cand_content.get("cases", [])}

    all_ids = set(base_cases.keys()) | set(cand_cases.keys())
    missing_in_cand = sorted(set(base_cases.keys()) - set(cand_cases.keys()))
    added_in_cand = sorted(set(cand_cases.keys()) - set(base_cases.keys()))

    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []

    for cid in sorted(all_ids):
        if cid in missing_in_cand or cid in added_in_cand:
            continue
        b_case = base_cases[cid]
        c_case = cand_cases[cid]

        # Check retrieval delta (Hit@5, MRR)
        b_ret = b_case.get("retrieval_metrics", {})
        c_ret = c_case.get("retrieval_metrics", {})
        ret_improved = False
        ret_regressed = False
        b_ret_app = b_ret.get("is_applicable", b_case.get("category") != "unanswerable")
        c_ret_app = c_ret.get("is_applicable", c_case.get("category") != "unanswerable")
        if b_ret_app and c_ret_app:
            b_hit = b_ret.get("hit_at_k", {}).get("5", b_ret.get("hit_at_5", 0.0))
            c_hit = c_ret.get("hit_at_k", {}).get("5", c_ret.get("hit_at_5", 0.0))
            b_mrr = b_ret.get("mrr", 0.0)
            c_mrr = c_ret.get("mrr", 0.0)
            if c_hit > b_hit or c_mrr > b_mrr:
                ret_improved = True
            elif c_hit < b_hit or c_mrr < b_mrr:
                ret_regressed = True

        # Check answer delta
        b_ans = b_case.get("answer_metrics", {})
        c_ans = c_case.get("answer_metrics", {})
        ans_improved = False
        ans_regressed = False
        b_ans_app = b_ans.get("is_answerable", b_case.get("category") != "unanswerable")
        c_ans_app = c_ans.get("is_answerable", c_case.get("category") != "unanswerable")
        if b_ans_app and c_ans_app:
            b_cov = b_ans.get("required_fact_coverage", 0.0)
            c_cov = c_ans.get("required_fact_coverage", 0.0)
            if c_cov > b_cov:
                ans_improved = True
            elif c_cov < b_cov:
                ans_regressed = True
        else:
            # Unanswerable abstention check
            b_abs = b_ans.get("correct_abstention", 0.0)
            c_abs = c_ans.get("correct_abstention", 0.0)
            if c_abs > b_abs:
                ans_improved = True
            elif c_abs < b_abs:
                ans_regressed = True

        if ret_regressed or ans_regressed:
            regressed.append(cid)
        elif ret_improved or ans_improved:
            improved.append(cid)
        else:
            unchanged.append(cid)

    return ComparisonReport(
        base_fingerprint=base_fp,
        cand_fingerprint=cand_fp,
        macro_deltas=macro_deltas,
        category_deltas=category_deltas,
        improved_case_ids=tuple(improved),
        regressed_case_ids=tuple(regressed),
        unchanged_case_ids=tuple(unchanged),
        missing_in_candidate=tuple(missing_in_cand),
        added_in_candidate=tuple(added_in_cand),
    )


def print_comparison_table(report: ComparisonReport) -> None:
    """Formats and prints an ASCII comparison report."""
    print("=================================================================")
    print("                     EVALUATION COMPARISON REPORT                ")
    print("=================================================================")
    print(f"Baseline Fingerprint:  {report.base_fingerprint}")
    print(f"Candidate Fingerprint: {report.cand_fingerprint}")
    print("-----------------------------------------------------------------")
    print(f"Cases: {len(report.improved_case_ids)} Improved | {len(report.regressed_case_ids)} Regressed | {len(report.unchanged_case_ids)} Unchanged")
    if report.missing_in_candidate:
        print(f"Missing in candidate ({len(report.missing_in_candidate)}): {', '.join(report.missing_in_candidate[:5])}...")
    if report.added_in_candidate:
        print(f"Added in candidate ({len(report.added_in_candidate)}): {', '.join(report.added_in_candidate[:5])}...")

    print("\n--- MACRO METRIC DELTAS ---")
    for section, deltas in report.macro_deltas.items():
        print(f"\n[{section.upper()}]")
        for k, v in sorted(deltas.items()):
            if k == "applicable_cases":
                continue
            symbol = "+" if v > 0 else ("-" if v < 0 else " ")
            print(f"  {k:<28} : {symbol}{v:+.4f}")

    print("\n--- CATEGORY BREAKDOWN ---")
    for cat, metrics in report.category_deltas.items():
        print(f"\nCategory: {cat}")
        for section, deltas in metrics.items():
            if not deltas:
                continue
            delta_strs = [f"{k}={'+' if v > 0 else ''}{v:.3f}" for k, v in deltas.items() if k != "applicable_cases" and v != 0]
            if delta_strs:
                print(f"  {section:<10}: {', '.join(delta_strs)}")
            else:
                print(f"  {section:<10}: No changes")

    if report.regressed_case_ids:
        print(f"\n[!] WARNING: Regressions detected in {len(report.regressed_case_ids)} cases: {', '.join(report.regressed_case_ids)}")
    else:
        print("\n[+] No regressions detected across test suite.")
    print("=================================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation run artifacts")
    parser.add_argument("baseline", type=str, help="Path to baseline evaluation JSON artifact")
    parser.add_argument("candidate", type=str, help="Path to candidate evaluation JSON artifact")
    parser.add_argument("--json", action="store_true", default=False, help="Output comparison result as JSON")

    args = parser.parse_args()

    base_artifact = load_artifact(args.baseline)
    cand_artifact = load_artifact(args.candidate)

    report = compare_runs(base_artifact, cand_artifact)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_comparison_table(report)


if __name__ == "__main__":
    main()
