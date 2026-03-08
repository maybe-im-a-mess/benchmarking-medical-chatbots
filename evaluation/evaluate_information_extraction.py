import json
from pathlib import Path
from typing import Dict, List

import numpy as np

GROUND_TRUTH_PATH = Path("data/ground_truth.json")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("data/evaluation_results")

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_THRESHOLD = 0.55
METHODS = ["naive", "cot", "atomic", "uie"]

_model_cache = {}


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out = {}
    for doc_key, topics in raw.items():
        stem = Path(doc_key).stem
        facts = []
        for topic in topics:
            for sub in topic.get("sub_topics", []):
                content = sub.get("content", "").strip()
                if content:
                    facts.append(content)
        out[stem] = facts
    return out


def load_extracted(file_path: Path) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for item in data.get("extracted_data", []):
        stmt = item.get("statement", "").strip()
        if stmt:
            out.append(stmt)
    return out


def get_embedding_model(model_name: str = EMBEDDING_MODEL):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def encode(texts: List[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    model = get_embedding_model(model_name)
    return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=True))


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def semantic_metrics(
    extracted: List[str],
    gt_embeddings: np.ndarray,
    gt_count: int,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = EMBEDDING_MODEL,
) -> Dict[str, float]:
    if not extracted or gt_count == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched_extracted": 0,
            "matched_gt": 0,
            "extracted_count": len(extracted),
            "gt_count": gt_count,
        }

    emb_ext = encode(extracted, model_name)
    sim = cosine_sim_matrix(emb_ext, gt_embeddings)

    matched_extracted = int((sim.max(axis=1) >= threshold).sum())
    matched_gt = int((sim.max(axis=0) >= threshold).sum())

    precision = matched_extracted / len(extracted)
    recall = matched_gt / gt_count
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched_extracted": matched_extracted,
        "matched_gt": matched_gt,
        "extracted_count": len(extracted),
        "gt_count": gt_count,
    }


def build_gt_embeddings(
    ground_truth: Dict[str, List[str]],
    model_name: str = EMBEDDING_MODEL,
) -> Dict[str, Dict[str, object]]:
    out = {}
    for doc, facts in ground_truth.items():
        if facts:
            out[doc] = {
                "facts": facts,
                "emb": encode(facts, model_name),
                "count": len(facts)
            }
        else:
            out[doc] = {"facts": [], "emb": None, "count": 0}
    return out


def evaluate_model_dir(
    model_dir: Path,
    gt_cache: Dict[str, Dict[str, object]],
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = EMBEDDING_MODEL,
) -> List[Dict]:
    results = []
    for method in METHODS:
        for doc, gt_info in gt_cache.items():
            file_path = model_dir / f"{doc}_{method}.json"
            if not file_path.exists():
                continue

            extracted = load_extracted(file_path)
            metrics = semantic_metrics(
                extracted,
                gt_info["emb"],
                gt_info["count"],
                threshold,
                model_name,
            )
            results.append({
                "model": model_dir.name,
                "document": doc,
                "method": method,
                "threshold": threshold,
                **metrics
            })
    return results


def discover_model_dirs(processed_dir: Path) -> List[Path]:
    model_dirs = []
    for d in processed_dir.iterdir():
        if not d.is_dir() or d.name == "legacy":
            continue
        if any(d.glob("*.json")):
            model_dirs.append(d)
        else:
            for sub in d.iterdir():
                if sub.is_dir() and any(sub.glob("*.json")):
                    model_dirs.append(sub)
    return sorted(model_dirs, key=lambda x: x.name)


def summarize_by_method(rows: List[Dict], model_key: str, threshold: float) -> List[Dict]:
    by_method: Dict[str, List[Dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    out = []
    for method, items in by_method.items():
        prec = [i["precision"] for i in items]
        rec = [i["recall"] for i in items]
        f1s = [i["f1"] for i in items]
        out.append({
            "model": model_key,
            "method": method,
            "threshold": threshold,
            "mean_precision": round(float(np.mean(prec)), 4),
            "mean_recall": round(float(np.mean(rec)), 4),
            "mean_f1": round(float(np.mean(f1s)), 4),
            "doc_count": len(items)
        })
    return sorted(out, key=lambda x: x["method"])


def save_json(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_evaluation(
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = EMBEDDING_MODEL,
    model_keys: List[str] = None,
):
    ground_truth = load_ground_truth()
    gt_cache = build_gt_embeddings(ground_truth, model_name)

    if model_keys:
        all_dirs = discover_model_dirs(processed_dir)
        model_dirs = [d for d in all_dirs if d.name in model_keys]
    else:
        model_dirs = discover_model_dirs(processed_dir)

    all_rows = []

    for model_dir in model_dirs:
        rows = evaluate_model_dir(model_dir, gt_cache, threshold, model_name)
        if not rows:
            continue

        all_rows.extend(rows)

    if all_rows:
        tag = f"thr{int(threshold * 100)}"
        save_json(results_dir / f"per_document_{tag}.json", all_rows)

    return all_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic evaluation")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--model-keys", nargs="*", default=None)
    args = parser.parse_args()

    rows = run_evaluation(
        threshold=args.threshold,
        model_name=args.model,
        model_keys=args.model_keys,
    )

    print(f"Saved results for {len(rows)} document runs")
