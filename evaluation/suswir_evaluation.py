import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

# Configuration
SOURCE_DOCS_DIR = Path("data/raw_md_files")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("data/evaluation_results")
CACHE_DIR = Path("data/embedding_cache")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
METHODS = ["naive", "cot", "atomic", "uie"]

_model_cache = {}

def get_embedding_model(model_name: str = EMBEDDING_MODEL):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def encode(texts: List[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    model = get_embedding_model(model_name)
    return np.array(model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False))

def get_cache_key(texts: List[str]) -> str:
    combined = "\n".join(texts)
    return hashlib.md5(combined.encode()).hexdigest()

def encode_cached(texts: List[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = get_cache_key(texts)
    cache_file = CACHE_DIR / f"{cache_key}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    embeddings = encode(texts, model_name)
    np.save(cache_file, embeddings)
    return embeddings

def load_source_text(doc_name: str) -> str:
    path = SOURCE_DOCS_DIR / f"{doc_name}.md"
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_extracted(file_path: Path) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item.get("statement", "").strip() for item in data.get("extracted_data", []) if item.get("statement")]

def calculate_suswir(extracted: List[str], source_text: str) -> Dict[str, float]:
    if not extracted or not source_text:
        return {"suswir_score": 0.0, "ssf": 0.0, "rdf": 0.0, "ref": 0.0, "baf": 0.0}

    # Prepare Embeddings
    source_sentences = [s.strip() for s in source_text.split('.') if len(s.strip()) > 10]
    if not source_sentences:
        return {"suswir_score": 0.0, "ssf": 0.0, "rdf": 0.0, "ref": 0.0, "baf": 0.0}

    emb_ext = encode_cached(extracted)
    emb_src = encode_cached(source_sentences)
    emb_src_full = encode_cached([source_text])[0]

    # 1. Semantic Similarity Factor (SSF)
    ssf = np.mean([np.dot(e, emb_src_full) for e in emb_ext])

    # 2. Redundancy Factor (RDF)
    n = len(extracted)
    if n <= 1:
        rdf = 1.0
    else:
        sim_matrix = emb_ext @ emb_ext.T
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        rdf = np.mean(upper_tri < 0.5)

    # 3. Relevance Factor (REF)
    sim_matrix_ref = emb_ext @ emb_src.T
    covered_source_sentences = (np.max(sim_matrix_ref, axis=0) > 0.6).sum()
    ref = covered_source_sentences / len(source_sentences)

    # 4. Bias Avoidance Factor (BAF)
    max_sim_per_ext = np.max(sim_matrix_ref, axis=1)
    baf = np.mean(max_sim_per_ext)

    # Final SUSWIR Score (Equal weighting)
    suswir_score = (ssf + rdf + ref + baf) / 4

    return {
        "suswir_score": round(float(suswir_score), 4),
        "ssf": round(float(ssf), 4),
        "rdf": round(float(rdf), 4),
        "ref": round(float(ref), 4),
        "baf": round(float(baf), 4)
    }

def run_evaluation():
    all_results = []
    model_dirs = []
    for d in PROCESSED_DIR.iterdir():
        if not d.is_dir() or d.name == "legacy":
            continue
        if any(d.glob("*.json")):
            model_dirs.append(d)
        else:
            for sub in d.iterdir():
                if sub.is_dir() and any(sub.glob("*.json")):
                    model_dirs.append(sub)

    print(f"Found {len(model_dirs)} model directories: {[d.name for d in model_dirs]}")

    for model_dir in model_dirs:
        for method in METHODS:
            for file_path in model_dir.glob(f"*_{method}.json"):
                doc_name = file_path.stem.replace(f"_{method}", "")
                source_text = load_source_text(doc_name)
                extracted = load_extracted(file_path)
                print(f"  Processing: {model_dir.name} / {doc_name} / {method} → {len(extracted)} statements")

                metrics = calculate_suswir(extracted, source_text)
                all_results.append({
                    "model": model_dir.name,
                    "document": doc_name,
                    "method": method,
                    **metrics
                })

    print(f"Total results: {len(all_results)}")
    save_path = RESULTS_DIR / "suswir_evaluation.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"SUSWIR Evaluation saved to {save_path}")

def inspect_rdf_pairs(extracted: List[str], low: float = 0.5, high: float = 0.7) -> List[Dict]:
    """Returns pairs of extracted statements whose similarity falls in [low, high]."""
    if len(extracted) < 2:
        return []

    emb_ext = encode_cached(extracted)
    sim_matrix = emb_ext @ emb_ext.T
    rows, cols = np.triu_indices(len(extracted), k=1)

    borderline_pairs = []
    for r, c in zip(rows, cols):
        sim = sim_matrix[r, c]
        if low <= sim <= high:
            borderline_pairs.append({
                "statement_a": extracted[r],
                "statement_b": extracted[c],
                "similarity": round(float(sim), 4)
            })

    return sorted(borderline_pairs, key=lambda x: x["similarity"], reverse=True)

if __name__ == "__main__":
    run_evaluation()

    # To inspect borderline RDF pairs after evaluation, uncomment and set a real path:
    # extracted = load_extracted(Path("data/processed/YOUR_MODEL/YOUR_FILE.json"))
    # pairs = inspect_rdf_pairs(extracted)
    # for p in pairs:
    #     print(f"[{p['similarity']}]\n  A: {p['statement_a']}\n  B: {p['statement_b']}\n")