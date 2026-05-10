"""
Shared utilities for thesis results chapter scripts.

Provides consistent path constants, display-name mappings, persona extraction,
and CSV/figure saving helpers used across all section scripts.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # code/
DATA_DIR = PROJECT_ROOT / "data"

# Information extraction evaluation
IE_RESULTS_DIR = DATA_DIR / "evaluation_results" / "informtion_extraction"
IE_PER_DOC_PATH = IE_RESULTS_DIR / "per_document_thr55.json"
IE_SUSWIR_PATH = IE_RESULTS_DIR / "suswir_evaluation.json"

# Dataset A  (v1, 60 conversations, GPT-5.4)
EVAL_A_DIR = DATA_DIR / "evaluation_results"
COVERAGE_A_HUNGARIAN = EVAL_A_DIR / "coverage" / "coverage_evaluation_thr75.json"
COVERAGE_A_SBERT = EVAL_A_DIR / "coverage_sbert" / "coverage_sbert_evaluation_thr50.json"
COVERAGE_A_LLM = EVAL_A_DIR / "coverage_llm_judge" / "coverage_llm_judge_evaluation.json"
CITATION_A_METRICS = EVAL_A_DIR / "citation_metrics.json"
CITATION_A_PER_FILE = EVAL_A_DIR / "citation_metrics_per_file.json"
MANDATORY_A_SUMMARY = EVAL_A_DIR / "mandatory_q_summary.json"
MANDATORY_A_PER_CONV = EVAL_A_DIR / "mandatory_q_per_conversation.json"

# Dataset B  (v3, 216 conversations, GPT-5.4-mini)
EVAL_B_DIR = DATA_DIR / "evaluation_results" / "v3"
COVERAGE_B_HUNGARIAN = EVAL_B_DIR / "coverage" / "coverage_evaluation_thr75.json"
COVERAGE_B_SBERT = EVAL_B_DIR / "coverage_sbert" / "coverage_sbert_evaluation_thr50.json"
COVERAGE_B_LLM = EVAL_B_DIR / "coverage_llm_judge" / "coverage_llm_judge_evaluation.json"
CITATION_B_METRICS = EVAL_B_DIR / "citation_metrics.json"
CITATION_B_PER_FILE = EVAL_B_DIR / "citation_metrics_per_file.json"
MANDATORY_B_SUMMARY = EVAL_B_DIR / "mandatory_q_summary.json"
MANDATORY_B_PER_CONV = EVAL_B_DIR / "mandatory_q_per_conversation.json"

# Conversation dirs (for qualitative excerpts)
CONV_A_DIR = DATA_DIR / "conversations" / "v1"
CONV_B_DIR = DATA_DIR / "conversations" / "patient_agent_probabilistic_v3"

# Output
TABLES_DIR = DATA_DIR / "evaluation_results" / "thesis_tables"

# ---------------------------------------------------------------------------
# Display-name mappings
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "gpt-5-mini": "GPT-5 Mini",
    "qwen3-32b": "Qwen3-32B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "ministral-3-14b-reasoning": "Ministral-14B",
}

METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "naive": "Naïve",
    "cot": "CoT",
    "atomic": "Atomic",
    "uie": "UIE",
}

# Mode mapping:  internal active/passive → thesis terminology
MODE_DISPLAY_NAMES: Dict[str, str] = {
    "active": "Supervised",
    "passive": "Naive",
}

APPROACH_DISPLAY_NAMES: Dict[str, str] = {
    "hungarian": "Semantic + Hungarian",
    "sbert": "Bi-Encoder + Entailment",
    "llm_judge": "LLM Judge",
}

PERSONA_DISPLAY_NAMES: Dict[str, str] = {
    "allergy_risk": "Allergy Risk",
    "anesthesia_risk": "Anesthesia Risk",
    "anticoagulation_risk": "Anticoagulation Risk",
    "baseline": "Baseline",
    "hypertension_risk": "Hypertension Risk",
    "induction_risk": "Induction Risk",
    "language_barrier_risk": "Language Barrier",
    "trauma_history_risk": "Trauma History",
    "version_contraindication": "Version Contraindication",
}


def display_model(raw: str) -> str:
    return MODEL_DISPLAY_NAMES.get(raw, raw)


def display_method(raw: str) -> str:
    return METHOD_DISPLAY_NAMES.get(raw, raw)


def display_mode(raw: str) -> str:
    return MODE_DISPLAY_NAMES.get(raw.lower(), raw)


def display_persona(raw: str) -> str:
    return PERSONA_DISPLAY_NAMES.get(raw, raw)


# ---------------------------------------------------------------------------
# Persona / mode extraction from filenames
# ---------------------------------------------------------------------------
def extract_persona(filename: str) -> str:
    """Extract the patient persona tag from a conversation filename.

    Expected patterns:
        Procedure_active_persona_tag_NNN.json
        Procedure_passive_persona_tag_NNN.json
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p in ("active", "passive"):
            persona_parts = parts[i + 1 : -1]  # everything between mode and number
            return "_".join(persona_parts)
    return "unknown"


def extract_mode_from_filename(filename: str) -> str:
    """Return 'active' or 'passive' extracted from the filename."""
    stem = Path(filename).stem
    if "_active_" in stem:
        return "active"
    if "_passive_" in stem:
        return "passive"
    return "unknown"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"  💾 Saved CSV: {path}")


def save_figure(fig: mpl.figure.Figure, path: Path, dpi: int = 300) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  🎨 Saved figure: {path}")


# ---------------------------------------------------------------------------
# Plot style — clean academic with a tiny dash of personality ✨
# ---------------------------------------------------------------------------
# A soft pastel-ish palette that still looks professional
PALETTE = [
    "#6C5CE7",  # soft purple
    "#00B894",  # minty green
    "#FD79A8",  # blush pink
    "#0984E3",  # calm blue
    "#FDCB6E",  # warm gold
    "#E17055",  # coral
    "#636E72",  # slate
    "#74B9FF",  # sky blue
    "#A29BFE",  # lavender
]

def setup_plot_style():
    """Apply a consistent, slightly charming academic plot style."""
    plt.rcParams.update({
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,

        # Grid & background
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,

        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": "#555555",
        "ytick.color": "#555555",

        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 100,

        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#DDDDDD",
        "legend.fontsize": 10,

        # Lines & patches
        "patch.edgecolor": "#FFFFFF",
        "lines.linewidth": 2.0,
    })


def get_colors(n: int = None) -> List[str]:
    """Return the palette colors, cycling if n exceeds length."""
    if n is None:
        return PALETTE
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


# ---------------------------------------------------------------------------
# Pretty-print table to stdout
# ---------------------------------------------------------------------------
def print_table(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
    print(df.to_string(index=False))
    print()
