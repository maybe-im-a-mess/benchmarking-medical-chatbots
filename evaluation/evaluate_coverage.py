import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

GROUND_TRUTH_PATH = Path("data/ground_truth.json")
CONVERSATIONS_DIR = Path(os.getenv("CONVERSATIONS_DIR", "data/conversations/v1"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "data/evaluation_results"))

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
DEFAULT_THRESHOLD = 0.75

IMPORTANCE_WEIGHTS = {
	"Critical": 4.0,
	"High": 3.0,
	"Medium": 2.0,
	"Low": 1.0,
}

_model_cache = {}


def make_safe_stem(name: str) -> str:
	# Keep readable names but avoid problematic path separators.
	return name.replace("/", "_")


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> Dict[str, List[Dict]]:
	with open(path, "r", encoding="utf-8") as f:
		raw = json.load(f)

	out: Dict[str, List[Dict]] = {}
	for doc_key, topics in raw.items():
		stem = Path(doc_key).stem
		facts = []
		for topic in topics:
			for sub in topic.get("sub_topics", []):
				content = sub.get("content", "").strip()
				if not content:
					continue
				facts.append({
					"fact_id": sub.get("fact_id"),
					"content": content,
					"importance": sub.get("importance", "Medium"),
				})
		out[stem] = facts
	return out


def normalize_procedure_key(procedure: str, candidates: List[str]) -> str:
	if not procedure:
		return ""
	for c in candidates:
		if c.lower() == procedure.lower():
			return c
	return ""


def get_embedding_model(model_name: str = EMBEDDING_MODEL):
	if model_name not in _model_cache:
		from sentence_transformers import SentenceTransformer

		print(f"Loading embedding model: {model_name} (device={EMBEDDING_DEVICE})")
		try:
			_model_cache[model_name] = SentenceTransformer(model_name, device=EMBEDDING_DEVICE)
		except RuntimeError as e:
			# Safety fallback for device OOM cases (e.g., Apple MPS with large models).
			if EMBEDDING_DEVICE != "cpu":
				print(f"Device '{EMBEDDING_DEVICE}' failed ({e}). Falling back to CPU.")
				_model_cache[model_name] = SentenceTransformer(model_name, device="cpu")
			else:
				raise
	return _model_cache[model_name]


def encode(texts: List[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
	if not texts:
		return np.empty((0, 0), dtype=float)
	model = get_embedding_model(model_name)
	return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def hungarian_max_match(similarity: np.ndarray) -> List[Tuple[int, int, float]]:
	if similarity.size == 0:
		return []

	from scipy.optimize import linear_sum_assignment

	rows, cols = similarity.shape
	n = max(rows, cols)

	padded = np.full((n, n), -1.0, dtype=float)
	padded[:rows, :cols] = similarity

	cost = -padded
	r_idx, c_idx = linear_sum_assignment(cost)

	matches = []
	for r, c in zip(r_idx, c_idx):
		if r < rows and c < cols:
			matches.append((r, c, float(similarity[r, c])))
	return matches


def load_conversation(file_path: Path) -> Dict:
	with open(file_path, "r", encoding="utf-8") as f:
		return json.load(f)


def get_chatbot_utterances(conversation: Dict) -> List[str]:
    utterances = []
    for turn in conversation.get("conversation", []):
        text = (turn.get("chatbot_response") or "").strip()
        if text:
            # 1. Split the text by newlines (for bullet points) and sentence endings (.!?)
            raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
            
            for sentence in raw_sentences:
                # 2. Clean up bullet point characters and whitespace
                clean_sentence = sentence.strip(' \t\n\r-*•')
                
                # 3. Filter out tiny fragments (like "Ja.", "Okay.", or empty strings)
                if len(clean_sentence) > 10:
                    utterances.append(clean_sentence)
                    
    return utterances


def evaluate_single_conversation(
	conv: Dict,
	gt_facts: List[Dict],
	threshold: float = DEFAULT_THRESHOLD,
	model_name: str = EMBEDDING_MODEL,
) -> Dict:
	utterances = get_chatbot_utterances(conv)

	if not gt_facts:
		return {
			"hit_rate": 0.0,
			"weighted_critical_recall": 0.0,
			"hits": 0,
			"total_topics": 0,
			"utterance_count": len(utterances),
			"matched_pairs": [],
		}

	gt_texts = [f["content"] for f in gt_facts]
	gt_weights = [IMPORTANCE_WEIGHTS.get(f.get("importance", "Medium"), 2.0) for f in gt_facts]

	if not utterances:
		return {
			"hit_rate": 0.0,
			"weighted_critical_recall": 0.0,
			"hits": 0,
			"total_topics": len(gt_texts),
			"utterance_count": 0,
			"matched_pairs": [],
		}

	emb_topics = encode(gt_texts, model_name)
	emb_utts = encode(utterances, model_name)
	sim = emb_topics @ emb_utts.T

	matches = hungarian_max_match(sim)
	best_for_topic = {i: 0.0 for i in range(len(gt_texts))}
	for i, j, score in matches:
		best_for_topic[i] = score

	hits = [1 if best_for_topic[i] >= threshold else 0 for i in range(len(gt_texts))]
	hit_rate = float(sum(hits) / len(gt_texts)) if gt_texts else 0.0

	total_weight = float(sum(gt_weights)) if gt_weights else 0.0
	hit_weight = float(sum(w for h, w in zip(hits, gt_weights) if h == 1))
	weighted_critical_recall = (hit_weight / total_weight) if total_weight else 0.0

	matched_pairs = []
	for i in range(len(gt_texts)):
		best_idx = None
		best_score = best_for_topic[i]
		for r, c, score in matches:
			if r == i:
				best_idx = c
				break
		matched_pairs.append({
			"fact_id": gt_facts[i].get("fact_id"),
			"importance": gt_facts[i].get("importance", "Medium"),
			"topic": gt_texts[i],
			"matched_utterance_index": best_idx,
			"matched_utterance": utterances[best_idx] if best_idx is not None else None,
			"similarity": round(float(best_score), 4),
			"hit": best_score >= threshold,
		})

	return {
		"hit_rate": round(hit_rate, 4),
		"weighted_critical_recall": round(float(weighted_critical_recall), 4),
		"hits": int(sum(hits)),
		"total_topics": len(gt_texts),
		"utterance_count": len(utterances),
		"matched_pairs": matched_pairs,
	}


def save_json(path: Path, data) -> None:
	def _json_default(obj):
		# Handle NumPy scalars produced by matching/indexing operations.
		if isinstance(obj, np.integer):
			return int(obj)
		if isinstance(obj, np.floating):
			return float(obj)
		if isinstance(obj, np.bool_):
			return bool(obj)
		raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def run_evaluation(
	conversations_dir: Path = CONVERSATIONS_DIR,
	results_dir: Path = RESULTS_DIR,
	threshold: float = DEFAULT_THRESHOLD,
	model_name: str = EMBEDDING_MODEL,
) -> List[Dict]:
	gt = load_ground_truth()
	tag = f"thr{int(threshold * 100)}"
	coverage_dir = results_dir / "coverage"
	per_conversation_dir = coverage_dir / "per_conversation"
	aggregate_path = coverage_dir / f"coverage_evaluation_{tag}.json"

	# Trigger model load once up front so progress messages are clearer.
	get_embedding_model(model_name)
	print("Embedding model is ready. Starting coverage evaluation...")

	conversation_files = [
		fp for fp in sorted(conversations_dir.glob("*.json"))
		if not fp.name.startswith("conversation_index_")
	]
	total_files = len(conversation_files)

	rows = []
	for idx, conv_file in enumerate(conversation_files, start=1):
		print(f"[{idx}/{total_files}] Evaluating: {conv_file.name}")

		conv = load_conversation(conv_file)
		procedure = (conv.get("metadata", {}).get("procedure") or "").strip()
		proc_key = normalize_procedure_key(procedure, list(gt.keys()))
		gt_facts = gt.get(proc_key, []) if proc_key else []

		metrics = evaluate_single_conversation(
			conv=conv,
			gt_facts=gt_facts,
			threshold=threshold,
			model_name=model_name,
		)

		row = {
			"file": conv_file.name,
			"procedure": procedure,
			"mode": conv.get("metadata", {}).get("mode"),
			"chatbot_model": conv.get("metadata", {}).get("chatbot_model"),
			"patient_model": conv.get("metadata", {}).get("patient_model"),
			"threshold": threshold,
			**metrics,
		}
		rows.append(row)

		# Save one file per conversation for traceability and interruption-safe progress.
		safe_stem = make_safe_stem(conv_file.stem)
		save_json(per_conversation_dir / f"{safe_stem}_{tag}.json", row)

		# Checkpoint after each conversation so interrupted runs keep progress.
		save_json(aggregate_path, rows)

	save_json(aggregate_path, rows)
	print(f"Coverage evaluation finished. Saved {len(rows)} conversation results.")
	return rows


if __name__ == "__main__":
	run_evaluation()
