#!/usr/bin/env python3
"""
Section 5.1.1 — Thresholded Semantic Matching Results

Reads per-document IE evaluation results and produces:
  • Table 1: Main results (16 model x method combos) with mean P / R / F1
  • Table 2: Per-document breakdown for the best-performing combination
  • Key findings printed to stdout
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.results._common import (
    IE_PER_DOC_PATH,
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
    out_dir = TABLES_DIR / "section_5_1_1"

    # ── Load data ──────────────────────────────────────────────────────
    raw = load_json(IE_PER_DOC_PATH)
    df = pd.DataFrame(raw)

    # ── Table 1: Main results (mean over documents) ────────────────────
    agg = (
        df.groupby(["model", "method"])
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            doc_count=("f1", "count"),
        )
        .reset_index()
    )
    agg = agg.sort_values("mean_f1", ascending=False).reset_index(drop=True)

    # Add display names
    agg["Model"] = agg["model"].map(display_model)
    agg["Method"] = agg["method"].map(display_method)
    table1 = agg[["Model", "Method", "mean_precision", "mean_recall", "mean_f1"]].copy()
    table1.columns = ["Model", "Method", "Mean Precision", "Mean Recall", "Mean F1"]
    for col in ["Mean Precision", "Mean Recall", "Mean F1"]:
        table1[col] = table1[col].round(4)

    print_table(table1, "Table 1: Thresholded Semantic Matching — All Combinations (sorted by F1)")
    save_csv(table1, out_dir / "thresholded_matching_main.csv")

    # ── Table 2: Per-document breakdown for best combo ─────────────────
    best_row = agg.iloc[0]
    best_model, best_method = best_row["model"], best_row["method"]
    best_df = df[(df["model"] == best_model) & (df["method"] == best_method)].copy()
    best_df = best_df[["document", "precision", "recall", "f1", "extracted_count", "gt_count"]].copy()
    best_df.columns = ["Document", "Precision", "Recall", "F1", "Extracted Count", "GT Count"]
    for col in ["Precision", "Recall", "F1"]:
        best_df[col] = best_df[col].round(4)

    print_table(
        best_df,
        f"Table 2: Per-Document Breakdown — Best Combo: "
        f"{display_model(best_model)} + {display_method(best_method)} (F1={best_row['mean_f1']:.4f})",
    )
    save_csv(best_df, out_dir / "thresholded_matching_per_doc.csv")

    # ── Key findings ───────────────────────────────────────────────────
    print("=" * 60)
    print("  KEY FINDINGS")
    print("=" * 60)

    # Best method per model
    print("\n📊 Best method per model:")
    for model_name in df["model"].unique():
        sub = agg[agg["model"] == model_name].sort_values("mean_f1", ascending=False)
        top = sub.iloc[0]
        print(f"  {display_model(model_name):20s} → {display_method(top['method']):8s}  (F1 = {top['mean_f1']:.4f})")

    # Best model per method
    print("\n📊 Best model per method:")
    for method_name in df["method"].unique():
        sub = agg[agg["method"] == method_name].sort_values("mean_f1", ascending=False)
        top = sub.iloc[0]
        print(f"  {display_method(method_name):8s} → {display_model(top['model']):20s}  (F1 = {top['mean_f1']:.4f})")

    # Precision vs. recall trade-off
    print("\n📊 Precision vs. Recall trade-off:")
    method_agg = (
        agg.groupby("method")
        .agg(avg_prec=("mean_precision", "mean"), avg_rec=("mean_recall", "mean"))
        .reset_index()
    )
    for _, r in method_agg.iterrows():
        delta = r["avg_prec"] - r["avg_rec"]
        direction = "precision-biased ⬆P" if delta > 0.02 else ("recall-biased ⬆R" if delta < -0.02 else "balanced ≈")
        print(f"  {display_method(r['method']):8s}: P={r['avg_prec']:.3f}  R={r['avg_rec']:.3f}  ({direction})")

    # Proprietary vs. open-source
    print("\n📊 Proprietary vs. Open-source comparison:")
    proprietary = agg[agg["model"] == "gpt-5-mini"]
    open_source = agg[agg["model"] != "gpt-5-mini"]
    prop_best = proprietary.loc[proprietary["mean_f1"].idxmax()]
    os_best = open_source.loc[open_source["mean_f1"].idxmax()]
    print(f"  Best proprietary:  {display_model(prop_best['model'])} + {display_method(prop_best['method'])}  F1={prop_best['mean_f1']:.4f}")
    print(f"  Best open-source:  {display_model(os_best['model'])} + {display_method(os_best['method'])}  F1={os_best['mean_f1']:.4f}")
    gap = prop_best["mean_f1"] - os_best["mean_f1"]
    if gap > 0:
        print(f"  → Proprietary leads by {gap:.4f} F1 points")
    else:
        print(f"  → Open-source leads by {abs(gap):.4f} F1 points — proprietary may NOT justify cost")

    # ── Figure: Grouped bar chart ─────────────────────────────────────
    models = sorted(agg["model"].unique())
    methods = sorted(agg["method"].unique())
    n_models = len(models)
    n_methods = len(methods)
    x = np.arange(n_models)
    width = 0.8 / n_methods
    colors = get_colors(n_methods)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, method in enumerate(methods):
        vals = []
        for model in models:
            row = agg[(agg["model"] == model) & (agg["method"] == method)]
            vals.append(row["mean_f1"].values[0] if len(row) else 0)
        bars = ax.bar(x + j * width - (n_methods - 1) * width / 2, vals, width,
                      label=display_method(method), color=colors[j], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#444")

    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Thresholded Semantic Matching — Mean F1 by Model x Method")
    ax.set_xticks(x)
    ax.set_xticklabels([display_model(m) for m in models])
    ax.set_ylim(0, 1.0)
    ax.legend(title="Extraction Method", loc="upper right")
    fig.tight_layout()
    save_figure(fig, out_dir / "thresholded_matching_f1_bar.png")

    print("\n✅ Section 5.1.1 complete.\n")


if __name__ == "__main__":
    main()
