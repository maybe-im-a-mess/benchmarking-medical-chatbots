#!/usr/bin/env python3
"""
Section 5.4 — Mandatory Question Compliance Evaluation

Produces:
  • Table 8: Dataset A — naive vs supervised
  • Table 9: Dataset B — naive vs supervised
  • Table 10: Dataset B — breakdown by persona
  • Key findings
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.results._common import (
    MANDATORY_A_SUMMARY, MANDATORY_A_PER_CONV,
    MANDATORY_B_SUMMARY, MANDATORY_B_PER_CONV,
    TABLES_DIR, display_mode, display_persona, extract_persona,
    extract_mode_from_filename, load_json, print_table, save_csv,
    setup_plot_style, save_figure, get_colors, PALETTE,
)
import matplotlib.pyplot as plt


def summary_to_table(summary, label):
    """Extract the mode-split metrics from a mandatory_q_summary.json."""
    rows = []
    for mode_key, mode_label in [("active_mode_metrics","Supervised"),("passive_mode_metrics","Naive")]:
        m = summary.get(mode_key, {})
        rows.append({
            "Dataset": label,
            "Mode": mode_label,
            "n": m.get("files_evaluated", 0),
            "Mean Q Recall": m.get("mean_question_recall", 0),
            "Strict Compliance": m.get("strict_compliance_rate", 0),
            "Acceptable Compliance": m.get("acceptable_compliance_rate", 0),
            "Mean 1st Q Turn": m.get("mean_first_mandatory_question_turn"),
            "Judge Failure Rate": m.get("judge_failure_rate", 0),
        })
    return pd.DataFrame(rows)


def per_conv_persona_table(per_conv, label):
    """Build persona × mode breakdown from per-conversation data."""
    rows = []
    for entry in per_conv:
        filepath = entry.get("file", "")
        filename = Path(filepath).name
        rows.append({
            "persona": extract_persona(filename),
            "mode": entry.get("mode", extract_mode_from_filename(filename)),
            "question_recall": entry.get("question_recall", 0),
            "strict_compliance": entry.get("strict_compliance", 0),
            "acceptable_compliance": entry.get("acceptable_compliance", 0),
            "first_q_turn": entry.get("first_mandatory_question_turn"),
            "judge_failed": entry.get("judge_failed", False),
        })
    df = pd.DataFrame(rows)

    agg = (
        df.groupby(["persona", "mode"])
        .agg(
            n=("persona", "count"),
            mean_recall=("question_recall", "mean"),
            strict_rate=("strict_compliance", "mean"),
            acceptable_rate=("acceptable_compliance", "mean"),
            mean_first_turn=("first_q_turn", lambda x: x.dropna().mean() if x.dropna().any() else None),
            judge_fail_rate=("judge_failed", "mean"),
        )
        .reset_index()
    )
    agg["Persona"] = agg["persona"].map(display_persona)
    agg["Mode"] = agg["mode"].map(display_mode)
    agg["Dataset"] = label
    return agg


def main():
    setup_plot_style()
    out_dir = TABLES_DIR / "section_5_4"

    # ── Dataset A ──────────────────────────────────────────────────────
    sum_a = load_json(MANDATORY_A_SUMMARY)
    t8 = summary_to_table(sum_a, "A (60, GPT-5.4)")
    for c in ["Mean Q Recall","Strict Compliance","Acceptable Compliance","Judge Failure Rate"]:
        t8[c] = t8[c].round(4)
    print_table(t8, "Table 8: Mandatory Q Compliance — Dataset A")
    save_csv(t8, out_dir / "mandatory_q_dataset_a.csv")

    # ── Dataset B ──────────────────────────────────────────────────────
    sum_b = load_json(MANDATORY_B_SUMMARY)
    t9 = summary_to_table(sum_b, "B (216, GPT-5.4-mini)")
    for c in ["Mean Q Recall","Strict Compliance","Acceptable Compliance","Judge Failure Rate"]:
        t9[c] = t9[c].round(4)
    print_table(t9, "Table 9: Mandatory Q Compliance — Dataset B")
    save_csv(t9, out_dir / "mandatory_q_dataset_b.csv")

    # ── Dataset B by persona ───────────────────────────────────────────
    per_conv_b = load_json(MANDATORY_B_PER_CONV)
    persona_agg = per_conv_persona_table(per_conv_b, "B")
    pt = persona_agg[["Persona","Mode","n","mean_recall","strict_rate","acceptable_rate","mean_first_turn","judge_fail_rate"]].copy()
    pt.columns = ["Persona","Mode","n","Mean Recall","Strict Compliance","Acceptable Compliance","Mean 1st Turn","Judge Fail Rate"]
    for c in ["Mean Recall","Strict Compliance","Acceptable Compliance","Judge Fail Rate"]:
        pt[c] = pt[c].round(4)
    pt["Mean 1st Turn"] = pt["Mean 1st Turn"].round(2)
    pt = pt.sort_values(["Persona","Mode"])
    print_table(pt, "Table 10: Mandatory Q Compliance — Dataset B by Persona")
    save_csv(pt, out_dir / "mandatory_q_persona_breakdown.csv")

    # ── Key findings ───────────────────────────────────────────────────
    print("="*60+"\n  KEY FINDINGS\n"+"="*60)

    for label, s in [("A", sum_a), ("B", sum_b)]:
        act = s.get("active_mode_metrics",{})
        pas = s.get("passive_mode_metrics",{})
        gap = act.get("mean_question_recall",0) - pas.get("mean_question_recall",0)
        print(f"\n  Dataset {label}:")
        print(f"    Supervised recall: {act.get('mean_question_recall',0):.4f}  |  "
              f"Naive recall: {pas.get('mean_question_recall',0):.4f}  |  Δ = {gap:+.4f}")
        print(f"    Supervised strict compliance: {act.get('strict_compliance_rate',0):.4f}  |  "
              f"Naive: {pas.get('strict_compliance_rate',0):.4f}")

    # Persona variation
    print("\n  📊 Persona variation (Dataset B, Supervised mode):")
    sup = persona_agg[persona_agg["mode"] == "active"].sort_values("mean_recall")
    for _, r in sup.iterrows():
        print(f"    {r['Persona']:25s}: recall={r['mean_recall']:.3f}  strict={r['strict_rate']:.3f}")

    # ── Figure: Bar chart comparing modes ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (label, s) in zip(axes, [("Dataset A", sum_a), ("Dataset B", sum_b)]):
        act = s.get("active_mode_metrics",{})
        pas = s.get("passive_mode_metrics",{})
        metrics = ["mean_question_recall","strict_compliance_rate","acceptable_compliance_rate"]
        metric_labels = ["Mean Recall","Strict\nCompliance","Acceptable\nCompliance"]
        x = np.arange(len(metrics))
        w = 0.35
        v_naive = [pas.get(m,0) for m in metrics]
        v_super = [act.get(m,0) for m in metrics]
        b1 = ax.bar(x - w/2, v_naive, w, label="Naive", color=PALETTE[2], edgecolor="white")
        b2 = ax.bar(x + w/2, v_super, w, label="Supervised", color=PALETTE[0], edgecolor="white")
        for bars in [b1, b2]:
            for b, v in zip(bars, [v_naive, v_super][bars is b2]):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{v:.2f}",
                        ha="center",va="bottom",fontsize=9,color="#333")
        ax.set_xticks(x); ax.set_xticklabels(metric_labels)
        ax.set_ylim(0,1.15); ax.set_ylabel("Rate"); ax.set_title(label)
        ax.legend(fontsize=9)
    fig.suptitle("Mandatory Question Compliance",fontsize=13,fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir / "mandatory_q_comparison.png")

    print("\n✅ Section 5.4 complete.\n")


if __name__ == "__main__":
    main()
