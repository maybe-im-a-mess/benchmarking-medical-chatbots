#!/usr/bin/env python3
"""
Section 5.1.2 — Adapted SUSWIR Results

Reads SUSWIR evaluation results and produces:
  • Table 3: SUSWIR factor breakdown (16 combos)
  • Analysis: high-REF / low-RDF combos and vice versa
  • Ranking comparison with thresholded F1 (Spearman)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
    setup_plot_style,
    get_colors,
    save_figure,
)
import matplotlib.pyplot as plt


def main():
    setup_plot_style()
    out_dir = TABLES_DIR / "section_5_1_2"

    # ── Load SUSWIR data ───────────────────────────────────────────────
    raw = load_json(IE_SUSWIR_PATH)
    df = pd.DataFrame(raw)

    # Aggregate over documents
    agg = (
        df.groupby(["model", "method"])
        .agg(
            mean_suswir=("suswir_score", "mean"),
            mean_ssf=("ssf", "mean"),
            mean_rdf=("rdf", "mean"),
            mean_ref=("ref", "mean"),
            mean_baf=("baf", "mean"),
            doc_count=("suswir_score", "count"),
        )
        .reset_index()
    )
    agg = agg.sort_values("mean_suswir", ascending=False).reset_index(drop=True)

    agg["Model"] = agg["model"].map(display_model)
    agg["Method"] = agg["method"].map(display_method)
    table = agg[["Model", "Method", "mean_suswir", "mean_ssf", "mean_rdf", "mean_ref", "mean_baf"]].copy()
    table.columns = ["Model", "Method", "SUSWIR", "SSF", "RDF", "REF", "BAF"]
    for col in ["SUSWIR", "SSF", "RDF", "REF", "BAF"]:
        table[col] = table[col].round(4)

    print_table(table, "Table 3: Adapted SUSWIR Results — All Combinations (sorted by SUSWIR)")
    save_csv(table, out_dir / "suswir_results.csv")

    # ── Analysis: REF vs RDF behaviour ─────────────────────────────────
    print("=" * 60)
    print("  SUSWIR FACTOR ANALYSIS")
    print("=" * 60)

    # High relevance, low redundancy (ideal)
    print("\n🌟 High REF + High RDF (low redundancy) — best extraction behaviour:")
    ideal = agg.copy()
    ideal["ref_rdf_combined"] = ideal["mean_ref"] + ideal["mean_rdf"]
    ideal = ideal.sort_values("ref_rdf_combined", ascending=False)
    for _, r in ideal.head(5).iterrows():
        print(f"  {display_model(r['model']):20s} + {display_method(r['method']):8s}  "
              f"REF={r['mean_ref']:.3f}  RDF={r['mean_rdf']:.3f}  SUSWIR={r['mean_suswir']:.3f}")

    # High redundancy (low RDF = many redundant statements)
    print("\n⚠️  Low RDF (high redundancy) — over-extracting similar statements:")
    redundant = agg.sort_values("mean_rdf", ascending=True)
    for _, r in redundant.head(3).iterrows():
        print(f"  {display_model(r['model']):20s} + {display_method(r['method']):8s}  "
              f"RDF={r['mean_rdf']:.3f}  REF={r['mean_ref']:.3f}")

    # Low relevance
    print("\n⚠️  Low REF (low source coverage) — under-extracting:")
    low_ref = agg.sort_values("mean_ref", ascending=True)
    for _, r in low_ref.head(3).iterrows():
        print(f"  {display_model(r['model']):20s} + {display_method(r['method']):8s}  "
              f"REF={r['mean_ref']:.3f}  RDF={r['mean_rdf']:.3f}")

    # ── Ranking comparison with thresholded F1 ─────────────────────────
    print("\n" + "=" * 60)
    print("  RANKING AGREEMENT: SUSWIR vs. Thresholded F1")
    print("=" * 60)

    ie_raw = load_json(IE_PER_DOC_PATH)
    ie_df = pd.DataFrame(ie_raw)
    ie_agg = (
        ie_df.groupby(["model", "method"])
        .agg(mean_f1=("f1", "mean"))
        .reset_index()
    )

    merged = agg.merge(ie_agg, on=["model", "method"], how="inner")
    merged["suswir_rank"] = merged["mean_suswir"].rank(ascending=False).astype(int)
    merged["f1_rank"] = merged["mean_f1"].rank(ascending=False).astype(int)

    rho, pval = spearmanr(merged["suswir_rank"], merged["f1_rank"])
    print(f"\n  Spearman ρ = {rho:.4f}  (p = {pval:.4f})")
    if rho > 0.7:
        print("  → Strong agreement: both metrics rank combinations similarly")
    elif rho > 0.4:
        print("  → Moderate agreement: partially consistent rankings")
    else:
        print("  → Weak agreement: metrics capture different aspects of quality")

    rank_table = merged[["model", "method", "mean_f1", "f1_rank", "mean_suswir", "suswir_rank"]].copy()
    rank_table["Model"] = rank_table["model"].map(display_model)
    rank_table["Method"] = rank_table["method"].map(display_method)
    rank_table = rank_table[["Model", "Method", "mean_f1", "f1_rank", "mean_suswir", "suswir_rank"]]
    rank_table.columns = ["Model", "Method", "Mean F1", "F1 Rank", "Mean SUSWIR", "SUSWIR Rank"]
    rank_table = rank_table.sort_values("F1 Rank")
    for col in ["Mean F1", "Mean SUSWIR"]:
        rank_table[col] = rank_table[col].round(4)

    print_table(rank_table, "Ranking comparison: F1 vs SUSWIR")
    save_csv(rank_table, out_dir / "suswir_vs_f1_ranking.csv")

    # ── Figure: SUSWIR factor radar / grouped bar ──────────────────────
    factors = ["SSF", "RDF", "REF", "BAF"]
    models = sorted(agg["model"].unique())
    methods = sorted(agg["method"].unique())
    colors = get_colors(len(methods))

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = agg[agg["model"] == model].sort_values("method")
        x = np.arange(len(factors))
        n = len(sub)
        w = 0.8 / n
        for j, (_, row) in enumerate(sub.iterrows()):
            vals = [row[f"mean_{f.lower()}"] for f in factors]
            ax.bar(x + j * w - (n - 1) * w / 2, vals, w,
                   label=display_method(row["method"]), color=colors[j], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(factors)
        ax.set_ylim(0, 1.05)
        ax.set_title(display_model(model), fontsize=11)
        if model == models[0]:
            ax.set_ylabel("Factor Score")
    axes[-1].legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.suptitle("SUSWIR Factor Breakdown by Model × Method", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir / "suswir_factor_breakdown.png")

    print("\n✅ Section 5.1.2 complete.\n")


if __name__ == "__main__":
    main()
