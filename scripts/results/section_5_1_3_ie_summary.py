#!/usr/bin/env python3
"""
Section 5.1.3 — IE Summary and Method Selection

Combines thresholded F1 and SUSWIR results to recommend the best
model + method combination for the medical information extraction task.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.results._common import (
    IE_PER_DOC_PATH,
    IE_SUSWIR_PATH,
    TABLES_DIR,
    display_method,
    display_model,
    load_json,
    print_table,
    save_csv,
)


def main():
    out_dir = TABLES_DIR / "section_5_1_3"

    # ── Load both evaluation results ───────────────────────────────────
    ie_raw = load_json(IE_PER_DOC_PATH)
    ie_df = pd.DataFrame(ie_raw)
    ie_agg = (
        ie_df.groupby(["model", "method"])
        .agg(mean_f1=("f1", "mean"), mean_precision=("precision", "mean"), mean_recall=("recall", "mean"))
        .reset_index()
    )

    suswir_raw = load_json(IE_SUSWIR_PATH)
    suswir_df = pd.DataFrame(suswir_raw)
    suswir_agg = (
        suswir_df.groupby(["model", "method"])
        .agg(mean_suswir=("suswir_score", "mean"))
        .reset_index()
    )

    # ── Merge and rank ─────────────────────────────────────────────────
    merged = ie_agg.merge(suswir_agg, on=["model", "method"], how="inner")
    merged["f1_rank"] = merged["mean_f1"].rank(ascending=False).astype(int)
    merged["suswir_rank"] = merged["mean_suswir"].rank(ascending=False).astype(int)
    merged["combined_rank"] = (merged["f1_rank"] + merged["suswir_rank"]) / 2
    merged = merged.sort_values("combined_rank").reset_index(drop=True)

    merged["Model"] = merged["model"].map(display_model)
    merged["Method"] = merged["method"].map(display_method)

    table = merged[
        ["Model", "Method", "mean_f1", "f1_rank", "mean_suswir", "suswir_rank", "combined_rank"]
    ].copy()
    table.columns = [
        "Model", "Method", "Mean F1", "F1 Rank", "Mean SUSWIR", "SUSWIR Rank", "Combined Rank",
    ]
    for col in ["Mean F1", "Mean SUSWIR", "Combined Rank"]:
        table[col] = table[col].round(4)

    print_table(table, "Combined Ranking: F1 + SUSWIR")
    save_csv(table, out_dir / "ie_combined_ranking.csv")

    # ── Summary recommendation ─────────────────────────────────────────
    best = merged.iloc[0]
    runner = merged.iloc[1]

    print("=" * 60)
    print("  RECOMMENDATION")
    print("=" * 60)
    print(f"\n  🏆 Best overall combination:")
    print(f"     {display_model(best['model'])} + {display_method(best['method'])}")
    print(f"     F1 = {best['mean_f1']:.4f} (rank {best['f1_rank']})")
    print(f"     SUSWIR = {best['mean_suswir']:.4f} (rank {best['suswir_rank']})")
    print(f"\n  🥈 Runner-up:")
    print(f"     {display_model(runner['model'])} + {display_method(runner['method'])}")
    print(f"     F1 = {runner['mean_f1']:.4f} (rank {runner['f1_rank']})")
    print(f"     SUSWIR = {runner['mean_suswir']:.4f} (rank {runner['suswir_rank']})")

    # Check if proprietary vs open-source matters
    prop = merged[merged["model"] == "gpt-5-mini"].iloc[0]
    best_open = merged[merged["model"] != "gpt-5-mini"].iloc[0]
    print(f"\n  💰 Cost consideration:")
    print(f"     Best proprietary: {display_model(prop['model'])} + {display_method(prop['method'])} "
          f"(combined rank {prop['combined_rank']:.1f})")
    print(f"     Best open-source: {display_model(best_open['model'])} + {display_method(best_open['method'])} "
          f"(combined rank {best_open['combined_rank']:.1f})")

    if best_open["combined_rank"] <= prop["combined_rank"]:
        print("     → Open-source matches or outperforms proprietary — cost savings achievable")
    else:
        gap = best_open["combined_rank"] - prop["combined_rank"]
        print(f"     → Proprietary leads by {gap:.1f} combined rank positions")

    print("\n✅ Section 5.1.3 complete.\n")


if __name__ == "__main__":
    main()
