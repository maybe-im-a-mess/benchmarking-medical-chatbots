import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

MANDATORY_PATH = Path("data/mandatory_questions.json")
CONVERSATIONS_DIR = Path("data/conversations")
RESULTS_DIR = Path("data/evaluation_results")

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_THRESHOLD = 0.75

_model_cache = {}


def load_mandatory_questions(path: Path = MANDATORY_PATH) -> Dict[str, List[Dict]]:
	with open(path, "r", encoding="utf-8") as f:
		raw = json.load(f)

	out: Dict[str, List[Dict]] = {}
	for doc_key, entries in raw.items():
		stem = Path(doc_key).stem
		questions = []
		for entry in entries:
			for q in entry.get("questions", []):
				content = (q.get("content") or "").strip()
				if content:
					questions.append({
						"question_id": q.get("question_id"),
						"content": content,
						"importance": q.get("importance", "High"),
					})
		out[stem] = questions
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

		print(f"Loading embedding model: {model_name}")
		_model_cache[model_name] = SentenceTransformer(model_name)
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
	out = []
	for turn in conversation.get("conversation", []):
		text = (turn.get("chatbot_response") or "").strip()
		if text:
			out.append(text)
	return out


def evaluate_single_conversation(
	conv: Dict,
	required_questions: List[Dict],
	threshold: float = DEFAULT_THRESHOLD,
	model_name: str = EMBEDDING_MODEL,
) -> Dict:
	utterances = get_chatbot_utterances(conv)
	if not required_questions:
		return {
			"mandatory_hit_rate": 0.0,
			"weighted_critical_recall": 0.0,
			"hits": 0,
			"total_required": 0,
			"utterance_count": len(utterances),
			"matched_pairs": [],
			"supervisor_logged_asked": conv.get("metadata", {}).get("mandatory_questions_asked", 0),
		}

	req_texts = [q["content"] for q in required_questions]
	req_weights = [1.0 for _ in required_questions]

	if not utterances:
		return {
			"mandatory_hit_rate": 0.0,
			"weighted_critical_recall": 0.0,
			"hits": 0,
			"total_required": len(req_texts),
			"utterance_count": 0,
			"matched_pairs": [],
			"supervisor_logged_asked": conv.get("metadata", {}).get("mandatory_questions_asked", 0),
		}

	emb_req = encode(req_texts, model_name)
	emb_utt = encode(utterances, model_name)
	sim = emb_req @ emb_utt.T

	matches = hungarian_max_match(sim)
	best_for_req = {i: 0.0 for i in range(len(req_texts))}
	for i, j, score in matches:
		best_for_req[i] = score

	hits = [1 if best_for_req[i] >= threshold else 0 for i in range(len(req_texts))]
	hit_rate = float(sum(hits) / len(req_texts)) if req_texts else 0.0

	total_weight = float(sum(req_weights)) if req_weights else 0.0
	hit_weight = float(sum(w for h, w in zip(hits, req_weights) if h == 1))
	weighted_recall = (hit_weight / total_weight) if total_weight else 0.0

	matched_pairs = []
	for i in range(len(req_texts)):
		best_idx = None
		best_score = best_for_req[i]
		for r, c, score in matches:
			if r == i:
				best_idx = c
				break
		matched_pairs.append({
			"question_id": required_questions[i].get("question_id"),
			"required_question": req_texts[i],
			"matched_utterance_index": best_idx,
			"matched_utterance": utterances[best_idx] if best_idx is not None else None,
			"similarity": round(float(best_score), 4),
			"hit": best_score >= threshold,
		})

	return {
		"mandatory_hit_rate": round(hit_rate, 4),
		"weighted_critical_recall": round(float(weighted_recall), 4),
		"hits": int(sum(hits)),
		"total_required": len(req_texts),
		"utterance_count": len(utterances),
		"matched_pairs": matched_pairs,
		"supervisor_logged_asked": conv.get("metadata", {}).get("mandatory_questions_asked", 0),
	}


def save_json(path: Path, data: List[Dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)


def summarize(rows: List[Dict]) -> List[Dict]:
	by_mode: Dict[str, List[Dict]] = {}
	for row in rows:
		mode = row.get("mode", "unknown")
		by_mode.setdefault(mode, []).append(row)

	out = []
	for mode, items in by_mode.items():
		out.append({
			"mode": mode,
			"conversation_count": len(items),
			"mean_mandatory_hit_rate": round(
				float(np.mean([x["mandatory_hit_rate"] for x in items])), 4
			),
			"mean_weighted_critical_recall": round(
				float(np.mean([x["weighted_critical_recall"] for x in items])), 4
			),
			"mean_supervisor_logged_asked": round(
				float(np.mean([x["supervisor_logged_asked"] for x in items])), 4
			),
		})
	return sorted(out, key=lambda x: x["mode"])


def run_evaluation(
	conversations_dir: Path = CONVERSATIONS_DIR,
	results_dir: Path = RESULTS_DIR,
	threshold: float = DEFAULT_THRESHOLD,
	model_name: str = EMBEDDING_MODEL,
) -> List[Dict]:
	mandatory = load_mandatory_questions()

	rows = []
	for conv_file in sorted(conversations_dir.glob("*.json")):
		if conv_file.name == "conversation_index_all.json":
			continue

		conv = load_conversation(conv_file)
		procedure = (conv.get("metadata", {}).get("procedure") or "").strip()
		proc_key = normalize_procedure_key(procedure, list(mandatory.keys()))
		required = mandatory.get(proc_key, []) if proc_key else []

		metrics = evaluate_single_conversation(
			conv=conv,
			required_questions=required,
			threshold=threshold,
			model_name=model_name,
		)

		rows.append({
			"file": conv_file.name,
			"procedure": procedure,
			"mode": conv.get("metadata", {}).get("mode"),
			"chatbot_model": conv.get("metadata", {}).get("chatbot_model"),
			"threshold": threshold,
			**metrics,
		})

	save_json(results_dir / f"mandatory_q_evaluation_thr{int(threshold * 100)}.json", rows)
	save_json(results_dir / f"mandatory_q_summary_thr{int(threshold * 100)}.json", summarize(rows))
	return rows


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Mandatory question evaluation")
	parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
	parser.add_argument("--model", type=str, default=EMBEDDING_MODEL)
	args = parser.parse_args()

	rows = run_evaluation(threshold=args.threshold, model_name=args.model)
	print(f"Saved mandatory question evaluation for {len(rows)} conversations")
