import json
from pathlib import Path
import argparse

DEFAULT_IMPORTANCE_WEIGHTS = {
    "Critical": 4.0,
    "High": 3.0,
    "Medium": 2.0,
    "Low": 1.0,
}

def _default_paths():
    return (
        Path("data/evaluation_results/coverage/coverage_evaluation_thr75.json"),
        0.60,
        Path("data/evaluation_results/coverage/coverage_evaluation_thr75_recalc_thr60.json"),
    )

def load_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def recalculate_hit_rate(results_file: str, new_threshold: float = 0.60, importance_weights: dict = None):
    data = load_results(Path(results_file))
    if importance_weights is None:
        importance_weights = DEFAULT_IMPORTANCE_WEIGHTS
    print(f"--- Recalculating with Threshold {new_threshold} ---")

    updated = []
    for conv in data:
        new_hits = 0
        total_topics = conv.get("total_topics", 0)

        if total_topics == 0:
            updated_pairs = []
            new_hit_rate = 0.0
            new_wcr = 0.0
        else:
            updated_pairs = []
            for pair in conv.get("matched_pairs", []):
                pair_updated = dict(pair)
                similarity = float(pair_updated.get("similarity", 0.0))
                is_hit = similarity >= new_threshold
                pair_updated["hit"] = is_hit
                if is_hit:
                    new_hits += 1
                updated_pairs.append(pair_updated)
            
            new_hit_rate = new_hits / total_topics if total_topics else 0.0

            total_weight = 0.0
            hit_weight = 0.0
            for pair in updated_pairs:
                imp = pair.get("importance", "Medium")
                w = float(importance_weights.get(imp, importance_weights.get("Medium", 2.0)))
                total_weight += w
                if pair.get("hit"):
                    hit_weight += w

            new_wcr = (hit_weight / total_weight) if total_weight else 0.0

        print(f"File: {conv['file']}")
        print(f"Old Hit Rate: {conv['hit_rate']} -> New Hit Rate: {round(new_hit_rate, 4)}")
        print(f"Old WCR: {conv.get('weighted_critical_recall')} -> New WCR: {round(new_wcr, 4)}\n")
        
        conv_updated = dict(conv)
        conv_updated["old_threshold"] = conv.get("threshold")
        conv_updated["threshold"] = new_threshold
        conv_updated["old_hit_rate"] = conv.get("hit_rate")
        conv_updated["hit_rate"] = round(new_hit_rate, 4)
        conv_updated["hits"] = int(new_hits)
        conv_updated["matched_pairs"] = updated_pairs
        conv_updated["old_weighted_critical_recall"] = conv.get("weighted_critical_recall")
        conv_updated["weighted_critical_recall"] = round(float(new_wcr), 4)
        updated.append(conv_updated)
    return updated

def _parse_args():
    p = argparse.ArgumentParser(description="Recalculate hit flags and weighted critical recall using a new threshold")
    p.add_argument("input_file", nargs="?", default=str(_default_paths()[0]), help="Path to input aggregate coverage JSON")
    p.add_argument("--threshold", "-t", type=float, default=_default_paths()[1], help="New similarity threshold (e.g. 0.6)")
    p.add_argument("--output", "-o", default=str(_default_paths()[2]), help="Path to output JSON file")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    recalculated = recalculate_hit_rate(str(args.input_file), args.threshold)
    out_path = Path(args.output)
    save_results(out_path, recalculated)
    print(f"Saved recalculated results to: {out_path}")
