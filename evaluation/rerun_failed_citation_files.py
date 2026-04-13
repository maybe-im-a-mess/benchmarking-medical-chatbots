import argparse
from pathlib import Path
from typing import Dict, List

from evaluation.evaluate_citation import (
    DEFAULT_JUDGE_MODEL,
    calculate_metrics,
    evaluate_conversation_file,
    load_json,
    save_json,
)


def find_failed_files(pair_rows: List[Dict], failure_marker: str) -> List[str]:
    failed = {
        row.get("file")
        for row in pair_rows
        if failure_marker in (row.get("judge_rationale") or "")
    }
    return sorted([f for f in failed if f])


def rebuild_global_summary(per_file_rows: List[Dict], judge_model: str) -> Dict:
    total_citations = sum(r["citation_precision"]["total_citations"] for r in per_file_rows)
    strict_supported = sum(r["citation_precision"]["supported_citations_strict"] for r in per_file_rows)
    relaxed_supported = sum(r["citation_precision"]["supported_citations_relaxed"] for r in per_file_rows)

    claims_with_citations = sum(r["support_coverage"]["claims_with_citations"] for r in per_file_rows)
    total_factual_claims = sum(r["support_coverage"]["total_factual_claims"] for r in per_file_rows)

    full_count = sum(r["support_distribution"]["full_support"]["count"] for r in per_file_rows)
    partial_count = sum(r["support_distribution"]["partial_support"]["count"] for r in per_file_rows)
    no_count = sum(r["support_distribution"]["no_support"]["count"] for r in per_file_rows)

    return {
        "judge_model": judge_model,
        "files_evaluated": len(per_file_rows),
        "citation_precision": {
            "strict_full_only": round(strict_supported / total_citations, 4) if total_citations else 0.0,
            "relaxed_full_plus_partial": round(relaxed_supported / total_citations, 4) if total_citations else 0.0,
            "supported_citations_strict": strict_supported,
            "supported_citations_relaxed": relaxed_supported,
            "total_citations": total_citations,
        },
        "support_coverage": {
            "claims_with_citations": claims_with_citations,
            "total_factual_claims": total_factual_claims,
            "coverage": round(claims_with_citations / total_factual_claims, 4)
            if total_factual_claims
            else 0.0,
        },
        "support_distribution": {
            "full_support": {
                "count": full_count,
                "percentage": round(full_count / total_citations, 4) if total_citations else 0.0,
            },
            "partial_support": {
                "count": partial_count,
                "percentage": round(partial_count / total_citations, 4) if total_citations else 0.0,
            },
            "no_support": {
                "count": no_count,
                "percentage": round(no_count / total_citations, 4) if total_citations else 0.0,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun failed citation-judge files and merge fresh results into existing JSON outputs."
    )
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=Path("data/evaluation_results/citation_pair_judgments.json"),
    )
    parser.add_argument(
        "--per-file-path",
        type=Path,
        default=Path("data/evaluation_results/citation_metrics_per_file.json"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("data/evaluation_results/citation_metrics.json"),
    )
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--failure-marker",
        type=str,
        default="Judge call failed",
        help="Substring used to detect failed rows in judge_rationale.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for rerun count (useful for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be rerun without changing files.",
    )
    args = parser.parse_args()

    if not args.pairs_path.exists() or not args.per_file_path.exists():
        raise FileNotFoundError(
            "Expected existing outputs are missing. Run evaluate_citation.py first to create them."
        )

    pairs = load_json(args.pairs_path)
    per_file = load_json(args.per_file_path)

    failed_files = find_failed_files(pairs, args.failure_marker)
    if args.max_files is not None:
        failed_files = failed_files[: args.max_files]

    print(f"Detected {len(failed_files)} failed files.")
    for fp in failed_files:
        print(f" - {fp}")

    if args.dry_run or not failed_files:
        print("No changes written.")
        return

    for file_str in failed_files:
        target = Path(file_str)
        print(f"Rerunning: {target}")

        refreshed = evaluate_conversation_file(target, args.judge_model)
        refreshed_metrics = calculate_metrics(
            refreshed["citation_pairs"],
            refreshed["total_factual_claims"],
            refreshed["claims_with_citations"],
        )

        new_file_row = {
            "file": str(target),
            "total_factual_claims": refreshed["total_factual_claims"],
            "claims_with_citations": refreshed["claims_with_citations"],
            **refreshed_metrics,
        }

        pairs = [r for r in pairs if r.get("file") != str(target)]
        pairs.extend(refreshed["citation_pairs"])

        per_file = [r for r in per_file if r.get("file") != str(target)]
        per_file.append(new_file_row)

    per_file = sorted(per_file, key=lambda r: r.get("file", ""))
    summary = rebuild_global_summary(per_file, args.judge_model)

    save_json(args.pairs_path, pairs)
    save_json(args.per_file_path, per_file)
    save_json(args.summary_path, summary)

    print("Merge complete.")
    print(f"Updated files: {len(failed_files)}")
    print(f"Total pair rows: {len(pairs)}")


if __name__ == "__main__":
    main()
