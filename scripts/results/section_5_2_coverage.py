#!/usr/bin/env python3
"""
Section 5.2 — Topic Coverage Evaluation

Reads coverage results from all three approaches (Hungarian, Bi-encoder+Entailment,
LLM Judge) for both datasets and produces:
  • Table 4a: Dataset A by mode x approach
  • Table 4b: Dataset B by mode x approach
  • Table 5:  Dataset B breakdown by persona
  • Figure 1: Grouped bar charts
  • Cross-method agreement analysis (Spearman + divergence)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.results._common import (
    COVERAGE_A_HUNGARIAN,
    COVERAGE_A_LLM,
    COVERAGE_A_SBERT,
    COVERAGE_B_HUNGARIAN,
    COVERAGE_B_LLM,
    COVERAGE_B_SBERT,
    TABLES_DIR,
    display_mode,
    display_persona,
    extract_persona,
    load_json,
    print_table,
    save_csv,
    setup_plot_style,
    get_colors,
    save_figure,
    PALETTE,
)
import matplotlib.pyplot as plt


APPROACH_PATHS_A = {
    "Hungarian": COVERAGE_A_HUNGARIAN,
    "Bi-Enc+Entailment": COVERAGE_A_SBERT,
    "LLM Judge": COVERAGE_A_LLM,
}

APPROACH_PATHS_B = {
    "Hungarian": COVERAGE_B_HUNGARIAN,
    "Bi-Enc+Entailment": COVERAGE_B_SBERT,
    "LLM Judge": COVERAGE_B_LLM,
}


def load_coverage_df(paths: dict) -> pd.DataFrame:
    """Load and tag coverage rows from multiple approaches."""
    frames = []
    for approach, path in paths.items():
        rows = load_json(path)
        for r in rows:
            r["approach"] = approach
            r["persona"] = extract_persona(r.get("file", ""))
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def build_mode_approach_table(df: pd.DataFrame) -> pd.DataFrame:
    """2x3 table: mode x approach with mean_hit_rate and mean_wcr."""
    agg = (
        df.groupby(["mode", "approach"])
        .agg(
            mean_hit_rate=("hit_rate", "mean"),
            std_hit_rate=("hit_rate", "std"),
            mean_wcr=("weighted_critical_recall", "mean"),
            std_wcr=("weighted_critical_recall", "std"),
            n=("hit_rate", "count"),
        )
        .reset_index()
    )
    agg["Mode"] = agg["mode"].map(display_mode)

    # Format for display
    table = agg[["Mode", "approach", "mean_hit_rate", "std_hit_rate", "mean_wcr", "std_wcr", "n"]].copy()
    table.columns = ["Mode", "Approach", "Mean Hit Rate", "SD Hit Rate", "Mean WCR", "SD WCR", "n"]
    for col in ["Mean Hit Rate", "SD Hit Rate", "Mean WCR", "SD WCR"]:
        table[col] = table[col].round(4)
    return table


def plot_mode_approach_bars(df: pd.DataFrame, title: str, out_path: Path):
    """Grouped bar chart: mode x approach for hit rate and WCR."""
    approaches = sorted(df["approach"].unique())
    modes = ["passive", "active"]  # naive first, supervised second
    mode_labels = [display_mode(m) for m in modes]
    colors = get_colors(len(approaches))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    metrics = [("hit_rate", "Mean Hit Rate"), ("weighted_critical_recall", "Mean Weighted Critical Recall")]

    for ax, (metric, ylabel) in zip(axes, metrics):
        x = np.arange(len(modes))
        n_app = len(approaches)
        w = 0.75 / n_app
        for j, approach in enumerate(approaches):
            vals = []
            errs = []
            for mode in modes:
                sub = df[(df["mode"] == mode) & (df["approach"] == approach)]
                vals.append(sub[metric].mean() if len(sub) else 0)
                errs.append(sub[metric].std() if len(sub) else 0)
            bars = ax.bar(x + j * w - (n_app - 1) * w / 2, vals, w,
                          yerr=errs, capsize=3, label=approach, color=colors[j],
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#444")
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.0)

    axes[0].legend(title="Approach", loc="upper left", fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_path)


def main():
    setup_plot_style()
    out_dir = TABLES_DIR / "section_5_2"

    # ── Dataset A (60 conversations, GPT-5.4) ─────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET A (60 conversations, GPT-5.4)")
    print("=" * 60)

    df_a = load_coverage_df(APPROACH_PATHS_A)
    table_a = build_mode_approach_table(df_a)
    print_table(table_a, "Table 4a: Topic Coverage — Dataset A, by Mode x Approach")
    save_csv(table_a, out_dir / "coverage_dataset_a.csv")

    plot_mode_approach_bars(df_a, "Topic Coverage — Dataset A (60 conversations)", out_dir / "coverage_dataset_a_bar.png")

    # Key finding: do all three approaches agree on naive vs supervised ranking?
    print("📊 Do all approaches agree that supervised > naive?")
    for approach in df_a["approach"].unique():
        sub = df_a[df_a["approach"] == approach]
        naive_hr = sub[sub["mode"] == "passive"]["hit_rate"].mean()
        super_hr = sub[sub["mode"] == "active"]["hit_rate"].mean()
        agree = "✅ Yes" if super_hr > naive_hr else "❌ No"
        print(f"  {approach:25s}: Naive={naive_hr:.3f}  Supervised={super_hr:.3f}  {agree}")

    # ── Dataset B (216 conversations, GPT-5.4-mini) ───────────────────
    print("\n" + "=" * 60)
    print("  DATASET B (216 conversations, GPT-5.4-mini)")
    print("=" * 60)

    df_b = load_coverage_df(APPROACH_PATHS_B)
    table_b = build_mode_approach_table(df_b)
    print_table(table_b, "Table 4b: Topic Coverage — Dataset B, by Mode x Approach")
    save_csv(table_b, out_dir / "coverage_dataset_b.csv")

    plot_mode_approach_bars(df_b, "Topic Coverage — Dataset B (216 conversations)", out_dir / "coverage_dataset_b_bar.png")

    # Persona breakdown
    persona_agg = (
        df_b.groupby(["persona", "mode", "approach"])
        .agg(
            mean_hit_rate=("hit_rate", "mean"),
            mean_wcr=("weighted_critical_recall", "mean"),
            n=("hit_rate", "count"),
        )
        .reset_index()
    )
    persona_agg["Persona"] = persona_agg["persona"].map(display_persona)
    persona_agg["Mode"] = persona_agg["mode"].map(display_mode)
    persona_table = persona_agg[["Persona", "Mode", "approach", "mean_hit_rate", "mean_wcr", "n"]].copy()
    persona_table.columns = ["Persona", "Mode", "Approach", "Mean Hit Rate", "Mean WCR", "n"]
    for col in ["Mean Hit Rate", "Mean WCR"]:
        persona_table[col] = persona_table[col].round(4)
    persona_table = persona_table.sort_values(["Persona", "Mode", "Approach"])

    print_table(persona_table, "Table 5: Dataset B — Coverage Breakdown by Persona")
    save_csv(persona_table, out_dir / "coverage_persona_breakdown.csv")

    # Key finding: supervised advantage consistent?
    print("📊 Supervised > naive consistent across personas?")
    for persona in sorted(df_b["persona"].unique()):
        sub = df_b[(df_b["persona"] == persona) & (df_b["approach"] == "Hungarian")]
        naive_hr = sub[sub["mode"] == "passive"]["hit_rate"].mean()
        super_hr = sub[sub["mode"] == "active"]["hit_rate"].mean()
        delta = super_hr - naive_hr
        emoji = "✅" if delta > 0 else "⚠️"
        print(f"  {display_persona(persona):25s}: Δ = {delta:+.3f}  {emoji}")

    # ── Cross-method agreement (Section 5.2.3) ────────────────────────
    print("\n" + "=" * 60)
    print("  CROSS-METHOD AGREEMENT ANALYSIS")
    print("=" * 60)

    # Use Dataset B (larger sample) for agreement analysis
    approaches = sorted(df_b["approach"].unique())
    pivot = df_b.pivot_table(index=["file", "mode"], columns="approach", values="hit_rate")
    pivot = pivot.dropna()

    agreement_rows = []
    for i in range(len(approaches)):
        for j in range(i + 1, len(approaches)):
            a, b = approaches[i], approaches[j]
            rho, pval = spearmanr(pivot[a], pivot[b])
            agreement_rows.append({
                "Approach A": a,
                "Approach B": b,
                "Spearman ρ": round(rho, 4),
                "p-value": round(pval, 6),
                "n": len(pivot),
            })
            print(f"  {a} vs {b}: ρ = {rho:.4f}  (p = {pval:.6f})")

    agreement_df = pd.DataFrame(agreement_rows)
    save_csv(agreement_df, out_dir / "cross_method_agreement.csv")

    # Where do they diverge?
    print("\n📊 Divergence analysis:")
    for a in approaches:
        for b in approaches:
            if a >= b:
                continue
            diff = (pivot[a] - pivot[b]).abs()
            n_big_diff = (diff > 0.2).sum()
            print(f"  {a} vs {b}: {n_big_diff}/{len(diff)} conversations differ by >0.2 hit rate")

    # LLM Judge vs embedding approaches
    if "LLM Judge" in pivot.columns:
        emb_cols = [c for c in pivot.columns if c != "LLM Judge"]
        for col in emb_cols:
            llm_higher = (pivot["LLM Judge"] > pivot[col]).sum()
            total = len(pivot)
            print(f"  LLM Judge scores higher than {col} in {llm_higher}/{total} "
                  f"({llm_higher/total*100:.1f}%) conversations")

    # ── Figure: Scatter plots for pairwise agreement ───────────────────
    n_pairs = len(approaches) * (len(approaches) - 1) // 2
    fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4.5), squeeze=False)
    axes = axes[0]
    idx = 0
    for i in range(len(approaches)):
        for j in range(i + 1, len(approaches)):
            a, b = approaches[i], approaches[j]
            ax = axes[idx]

            # Color by mode
            for mode, color, marker in [("passive", PALETTE[2], "o"), ("active", PALETTE[0], "s")]:
                mask = pivot.index.get_level_values("mode") == mode
                ax.scatter(pivot.loc[mask, a], pivot.loc[mask, b],
                           c=color, marker=marker, alpha=0.6, s=30, label=display_mode(mode))

            ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1)
            ax.set_xlabel(a)
            ax.set_ylabel(b)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{a}\nvs {b}", fontsize=10)
            ax.legend(fontsize=8)
            idx += 1

    fig.suptitle("Cross-Method Agreement — Hit Rate (Dataset B)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir / "cross_method_agreement_scatter.png")

    print("\n✅ Section 5.2 complete.\n")


if __name__ == "__main__":
    main()
