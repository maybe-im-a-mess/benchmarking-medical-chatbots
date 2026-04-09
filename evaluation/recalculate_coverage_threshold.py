import json
from pathlib import Path

INPUT_FILE = Path("data/evaluation_results/coverage/coverage_evaluation_thr75.json")
NEW_THRESHOLD = 0.60
OUTPUT_FILE = Path("data/evaluation_results/coverage/coverage_evaluation_thr75_recalc_thr60.json")


def load_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def recalculate_hit_rate(results_file: str, new_threshold: float = 0.60):
    data = load_results(Path(results_file))

    print(f"--- Recalculating with Threshold {new_threshold} ---")

    updated = []
    for conv in data:
        new_hits = 0
        total_topics = conv.get("total_topics", 0)

        if total_topics == 0:
            updated.append(conv)
            continue

        updated_pairs = []
        for pair in conv.get("matched_pairs", []):
            pair_updated = dict(pair)
            similarity = float(pair_updated.get("similarity", 0.0))
            is_hit = similarity >= new_threshold
            pair_updated["hit"] = is_hit
            if is_hit:
                new_hits += 1
            updated_pairs.append(pair_updated)

        new_hit_rate = new_hits / total_topics

        print(f"File: {conv['file']}")
        print(f"Old Hit Rate: {conv['hit_rate']} -> New Hit Rate: {round(new_hit_rate, 4)}\n")

        conv_updated = dict(conv)
        conv_updated["old_threshold"] = conv.get("threshold")
        conv_updated["threshold"] = new_threshold
        conv_updated["old_hit_rate"] = conv.get("hit_rate")
        conv_updated["hit_rate"] = round(new_hit_rate, 4)
        conv_updated["hits"] = int(new_hits)
        conv_updated["matched_pairs"] = updated_pairs
        updated.append(conv_updated)

    return updated


if __name__ == "__main__":
    recalculated = recalculate_hit_rate(str(INPUT_FILE), NEW_THRESHOLD)
    save_results(OUTPUT_FILE, recalculated)
    print(f"Saved recalculated results to: {OUTPUT_FILE}")