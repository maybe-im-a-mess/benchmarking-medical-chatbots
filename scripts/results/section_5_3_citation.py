#!/usr/bin/env python3
"""
Section 5.3 — Citation Faithfulness Evaluation

Reads citation evaluation results for both datasets and produces:
  • Table 6: Aggregate citation metrics (Dataset A vs B)
  • Table 7: Citation metrics split by mode (naive vs supervised)
  • Key findings and support distribution figure
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.results._common import (
    CITATION_A_METRICS, CITATION_A_PER_FILE,
    CITATION_B_METRICS, CITATION_B_PER_FILE,
    TABLES_DIR, display_mode, extract_mode_from_filename,
    load_json, print_table, save_csv, setup_plot_style, save_figure, PALETTE,
)
import matplotlib.pyplot as plt


def flatten_metrics(m, label):
    cp, sc, sd = m.get("citation_precision",{}), m.get("support_coverage",{}), m.get("support_distribution",{})
    return {"Dataset":label, "Files":m.get("files_evaluated",0),
            "Strict Precision":cp.get("strict_full_only",0),
            "Relaxed Precision":cp.get("relaxed_full_plus_partial",0),
            "Support Coverage":sc.get("coverage",0),
            "Full %":sd.get("full_support",{}).get("percentage",0),
            "Partial %":sd.get("partial_support",{}).get("percentage",0),
            "No Support %":sd.get("no_support",{}).get("percentage",0),
            "Total Citations":cp.get("total_citations",0),
            "Total Claims":sc.get("total_factual_claims",0)}


def mode_table(per_file, label):
    rows = []
    for e in per_file:
        mode = extract_mode_from_filename(Path(e.get("file","")).name)
        cp, sc, sd = e.get("citation_precision",{}), e.get("support_coverage",{}), e.get("support_distribution",{})
        rows.append({"mode":mode, "strict":cp.get("strict_full_only",0),
                      "relaxed":cp.get("relaxed_full_plus_partial",0),
                      "coverage":sc.get("coverage",0),
                      "full":sd.get("full_support",{}).get("percentage",0),
                      "partial":sd.get("partial_support",{}).get("percentage",0),
                      "none":sd.get("no_support",{}).get("percentage",0)})
    df = pd.DataFrame(rows)
    agg = df.groupby("mode").agg(n=("mode","count"), strict=("strict","mean"),
            relaxed=("relaxed","mean"), coverage=("coverage","mean"),
            full=("full","mean"), partial=("partial","mean"), none=("none","mean")).reset_index()
    agg["Dataset"], agg["Mode"] = label, agg["mode"].map(display_mode)
    return agg


def main():
    setup_plot_style()
    out_dir = TABLES_DIR / "section_5_3"

    ma, mb = load_json(CITATION_A_METRICS), load_json(CITATION_B_METRICS)
    t6 = pd.DataFrame([flatten_metrics(ma,"A (60, GPT-5.4)"), flatten_metrics(mb,"B (216, GPT-5.4-mini)")])
    for c in ["Strict Precision","Relaxed Precision","Support Coverage","Full %","Partial %","No Support %"]:
        t6[c] = t6[c].round(4)
    print_table(t6, "Table 6: Citation Faithfulness — Aggregate")
    save_csv(t6, out_dir / "citation_results.csv")

    pfa, pfb = load_json(CITATION_A_PER_FILE), load_json(CITATION_B_PER_FILE)
    ma2, mb2 = mode_table(pfa,"A"), mode_table(pfb,"B")
    mall = pd.concat([ma2, mb2], ignore_index=True)
    mt = mall[["Dataset","Mode","n","strict","relaxed","coverage","full","partial","none"]].copy()
    mt.columns = ["Dataset","Mode","n","Strict Prec","Relaxed Prec","Coverage","Full%","Partial%","None%"]
    for c in mt.columns[3:]: mt[c] = mt[c].round(4)
    print_table(mt, "Table 7: Citation by Mode")
    save_csv(mt, out_dir / "citation_by_mode.csv")

    print("="*60+"\n  KEY FINDINGS\n"+"="*60)
    for label, m in [("A", ma), ("B", mb)]:
        sd = m.get("support_distribution",{})
        f = sd.get("full_support",{}).get("percentage",0)*100
        p = sd.get("partial_support",{}).get("percentage",0)*100
        n = sd.get("no_support",{}).get("percentage",0)*100
        cov = m.get("support_coverage",{}).get("coverage",0)*100
        print(f"\n  Dataset {label}: Full={f:.1f}%  Partial={p:.1f}%  None={n:.1f}%  Coverage={cov:.1f}%")

    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    for ax,(label,m) in zip(axes,[("Dataset A",ma),("Dataset B",mb)]):
        sd = m.get("support_distribution",{})
        vals = [sd.get(k,{}).get("percentage",0) for k in ["full_support","partial_support","no_support"]]
        cats = ["Full","Partial","None"]
        cols = [PALETTE[1],PALETTE[3],PALETTE[2]]
        bars = ax.bar(cats, vals, color=cols, edgecolor="white", width=0.6)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v*100:.1f}%",
                    ha="center",va="bottom",fontsize=10,fontweight="bold",color="#333")
        ax.set_ylim(0,1); ax.set_ylabel("Proportion"); ax.set_title(label)
    fig.suptitle("Citation Support Distribution",fontsize=13,fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir / "citation_support_distribution.png")
    print("\n✅ Section 5.3 complete.\n")

if __name__ == "__main__":
    main()
