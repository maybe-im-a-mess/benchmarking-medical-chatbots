import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path("data/evaluation_results")
PLOTS_DIR = RESULTS_ROOT / "plots"

BASELINE_PATH = RESULTS_ROOT / "coverage" / "coverage_evaluation_thr75_recalc_thr60.json"
SBERT_PATH = RESULTS_ROOT / "coverage_sbert" / "coverage_sbert_evaluation_thr50.json"
LLM_JUDGE_PATH = RESULTS_ROOT / "coverage_llm_judge" / "coverage_llm_judge_evaluation.json"

METHODS = {
    "Semantic + Hungarian": BASELINE_PATH,
    "SBERT + Cross-Encoder": SBERT_PATH,
    "LLM Judge": LLM_JUDGE_PATH,
}


def load_rows(path: Path):
    if not path.exists():
        print(f"Missing file, skipping: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def method_metrics(rows):
    if not rows:
        return {"mean_hit_rate": 0.0, "mean_wcr": 0.0, "std_hit_rate": 0.0, "std_wcr": 0.0}

    hit_rates = np.array([float(r.get("hit_rate", 0.0)) for r in rows], dtype=float)
    wcr = np.array([float(r.get("weighted_critical_recall", 0.0)) for r in rows], dtype=float)

    return {
        "mean_hit_rate": float(np.mean(hit_rates)),
        "mean_wcr": float(np.mean(wcr)),
        "std_hit_rate": float(np.std(hit_rates)),
        "std_wcr": float(np.std(wcr)),
    }


def by_mode(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r.get("mode") or "unknown").lower()].append(r)

    out = {}
    for mode, items in grouped.items():
        out[mode] = {
            "mean_hit_rate": mean([float(x.get("hit_rate", 0.0)) for x in items]),
            "mean_wcr": mean([float(x.get("weighted_critical_recall", 0.0)) for x in items]),
        }
    return out


def by_procedure(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.get("procedure") or "unknown"].append(r)

    out = {}
    for proc, items in grouped.items():
        out[proc] = {
            "mean_hit_rate": mean([float(x.get("hit_rate", 0.0)) for x in items]),
            "mean_wcr": mean([float(x.get("weighted_critical_recall", 0.0)) for x in items]),
        }
    return out


def align_by_file(method_rows):
    aligned = defaultdict(dict)
    for method_name, rows in method_rows.items():
        for r in rows:
            file_name = r.get("file")
            if not file_name:
                continue
            aligned[file_name][method_name] = {
                "hit_rate": float(r.get("hit_rate", 0.0)),
                "wcr": float(r.get("weighted_critical_recall", 0.0)),
                "mode": r.get("mode"),
                "procedure": r.get("procedure"),
            }
    return aligned


def plot_overall_comparison(method_rows):
    names = list(method_rows.keys())
    hit_means = []
    hit_stds = []
    wcr_means = []
    wcr_stds = []

    for n in names:
        m = method_metrics(method_rows[n])
        hit_means.append(m["mean_hit_rate"])
        hit_stds.append(m["std_hit_rate"])
        wcr_means.append(m["mean_wcr"])
        wcr_stds.append(m["std_wcr"])

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, hit_means, width, yerr=hit_stds, capsize=4, label="Hit Rate")
    ax.bar(x + width / 2, wcr_means, width, yerr=wcr_stds, capsize=4, label="Weighted Critical Recall")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Overall Coverage Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.legend()
    fig.tight_layout()

    out = PLOTS_DIR / "coverage_overall_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_mode_split(method_rows):
    modes = ["active", "passive"]
    names = list(method_rows.keys())
    x = np.arange(len(names))
    width = 0.36

    active_vals = []
    passive_vals = []

    for name in names:
        mode_map = by_mode(method_rows[name])
        active_vals.append(mode_map.get("active", {}).get("mean_hit_rate", 0.0))
        passive_vals.append(mode_map.get("passive", {}).get("mean_hit_rate", 0.0))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, active_vals, width, label="Active")
    ax.bar(x + width / 2, passive_vals, width, label="Passive")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Hit Rate")
    ax.set_title("Coverage by Dialogue Mode")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.legend()
    fig.tight_layout()

    out = PLOTS_DIR / "coverage_mode_split_hit_rate.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_procedure_heatmaps(method_rows):
    procedures = sorted(
        {
            r.get("procedure")
            for rows in method_rows.values()
            for r in rows
            if r.get("procedure")
        }
    )
    names = list(method_rows.keys())

    hit_matrix = np.zeros((len(procedures), len(names)), dtype=float)
    wcr_matrix = np.zeros((len(procedures), len(names)), dtype=float)

    for j, name in enumerate(names):
        proc_map = by_procedure(method_rows[name])
        for i, proc in enumerate(procedures):
            hit_matrix[i, j] = proc_map.get(proc, {}).get("mean_hit_rate", 0.0)
            wcr_matrix[i, j] = proc_map.get(proc, {}).get("mean_wcr", 0.0)

    def draw_heatmap(matrix, title, out_name):
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=10)
        ax.set_yticks(np.arange(len(procedures)))
        ax.set_yticklabels(procedures)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Score")
        fig.tight_layout()

        out = PLOTS_DIR / out_name
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"Saved: {out}")

    draw_heatmap(hit_matrix, "Procedure-Level Mean Hit Rate", "coverage_procedure_heatmap_hit_rate.png")
    draw_heatmap(wcr_matrix, "Procedure-Level Mean Weighted Critical Recall", "coverage_procedure_heatmap_wcr.png")


def plot_pairwise_scatter(method_rows):
    aligned = align_by_file(method_rows)
    names = list(method_rows.keys())
    pairs = [
        (names[0], names[1]),
        (names[0], names[2]),
        (names[1], names[2]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, (a, b) in zip(axes, pairs):
        xs = []
        ys = []
        for _, data in aligned.items():
            if a in data and b in data:
                xs.append(data[a]["hit_rate"])
                ys.append(data[b]["hit_rate"])

        ax.scatter(xs, ys, alpha=0.7)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{a} vs {b}")
        ax.set_xlabel(f"{a}\nHit Rate")
        ax.set_ylabel(f"{b}\nHit Rate")

    fig.suptitle("Conversation-Level Agreement (Hit Rate)")
    fig.tight_layout()

    out = PLOTS_DIR / "coverage_pairwise_scatter_hit_rate.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    method_rows = {name: load_rows(path) for name, path in METHODS.items()}
    available = {k: v for k, v in method_rows.items() if v}

    if len(available) < 2:
        print("Need at least two result files to create comparison plots.")
        return

    # Keep plot ordering stable based on METHODS declaration.
    method_rows = {k: available[k] for k in METHODS.keys() if k in available}

    print("Building plots for methods:")
    for k, rows in method_rows.items():
        print(f"  - {k}: {len(rows)} conversations")

    plot_overall_comparison(method_rows)
    plot_mode_split(method_rows)
    plot_procedure_heatmaps(method_rows)

    if len(method_rows) >= 3:
        plot_pairwise_scatter(method_rows)

    print("Done.")


if __name__ == "__main__":
    main()
