import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils.llm_config import make_api_call

GROUND_TRUTH_PATH = Path("data/ground_truth.json")
QUESTIONS_DIR = Path("data/comprehension_questions")
CONVERSATIONS_DIR = Path("data/conversations")
RESULTS_DIR = Path("data/evaluation_results")

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
ANSWER_MODEL = "gpt-5-mini"
DEFAULT_THRESHOLD = 0.65

IMPORTANCE_WEIGHTS = {
	"Critical": 4.0,
	"High": 3.0,
	"Medium": 2.0,
	"Low": 1.0,
}

_model_cache = {}


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> Dict[str, Dict[str, Dict]]:
	with open(path, "r", encoding="utf-8") as f:
		raw = json.load(f)

	out: Dict[str, Dict[str, Dict]] = {}
	for doc_key, topics in raw.items():
		stem = Path(doc_key).stem
		fact_map = {}
		for topic in topics:
			for sub in topic.get("sub_topics", []):
				fact_id = sub.get("fact_id")
				content = (sub.get("content") or "").strip()
				if not fact_id or not content:
					continue
				fact_map[fact_id] = {
					"content": content,
					"importance": sub.get("importance", "Medium"),
				}
		out[stem] = fact_map
	return out


def load_questions(questions_dir: Path = QUESTIONS_DIR) -> Dict[str, List[Dict]]:
	out: Dict[str, List[Dict]] = {}
	for fp in sorted(questions_dir.glob("*.json")):
		with open(fp, "r", encoding="utf-8") as f:
			data = json.load(f)
		proc_name = (data.get("procedure_name") or "").strip()
		if proc_name:
			out[proc_name] = data.get("questions", [])
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


def load_conversation(file_path: Path) -> Dict:
	with open(file_path, "r", encoding="utf-8") as f:
		return json.load(f)


def build_conversation_text(conv: Dict) -> str:
	lines = []
	for turn in conv.get("conversation", []):
		p = (turn.get("patient_question") or "").strip()
		c = (turn.get("chatbot_response") or "").strip()
		if p:
			lines.append(f"Patient: {p}")
		if c:
			lines.append(f"Chatbot: {c}")
	return "\n".join(lines)


def generate_patient_answer(
	question: str,
	conversation_text: str,
	answer_model: str = ANSWER_MODEL,
) -> str:
	system_message = (
		"Du bist dieselbe Patientin aus dem Gespräch. "
		"Beantworte die Frage NUR basierend auf dem, was der Chatbot erklärt hat. "
		"Wenn etwas nicht erklärt wurde, antworte ehrlich mit 'Das weiß ich nicht aus dem Gespräch.'. "
		"Antworte kurz in 1-3 Sätzen auf Deutsch."
	)
	prompt = (
		"Hier ist das Gespräch:\n"
		f"{conversation_text}\n\n"
		f"Frage: {question}"
	)

	content = make_api_call(
		prompt=prompt,
		model_name=answer_model,
		temperature=0.2,
		timeout=180,
		system_message=system_message,
	)
	return (content or "").strip()


def score_answer_against_facts(
	answer: str,
	fact_texts: List[str],
	threshold: float,
	model_name: str,
) -> Dict:
	if not fact_texts or not answer:
		return {
			"max_similarity": 0.0,
			"hit": False,
		}

	emb_answer = encode([answer], model_name)
	emb_facts = encode(fact_texts, model_name)
	sims = emb_answer @ emb_facts.T
	max_sim = float(np.max(sims)) if sims.size else 0.0

	return {
		"max_similarity": round(max_sim, 4),
		"hit": max_sim >= threshold,
	}


def evaluate_single_conversation(
	conv: Dict,
	questions: List[Dict],
	fact_map: Dict[str, Dict],
	threshold: float,
	model_name: str,
	answer_model: str,
) -> Dict:
	conversation_text = build_conversation_text(conv)

	if not questions:
		return {
			"comprehension_hit_rate": 0.0,
			"weighted_critical_recall": 0.0,
			"question_count": 0,
			"hits": 0,
			"details": [],
		}

	details = []
	weights = []
	hits = []

	for q in questions:
		q_id = q.get("id")
		q_text = (q.get("question") or "").strip()
		related_fact_ids = q.get("related_fact_ids", [])
		related = [fact_map[fid] for fid in related_fact_ids if fid in fact_map]
		related_texts = [x["content"] for x in related]

		if related:
			q_weight = float(np.mean([
				IMPORTANCE_WEIGHTS.get(x.get("importance", "Medium"), 2.0)
				for x in related
			]))
		else:
			q_weight = 2.0

		patient_answer = generate_patient_answer(
			question=q_text,
			conversation_text=conversation_text,
			answer_model=answer_model,
		)
		score = score_answer_against_facts(
			answer=patient_answer,
			fact_texts=related_texts,
			threshold=threshold,
			model_name=model_name,
		)

		hit = 1 if score["hit"] else 0
		hits.append(hit)
		weights.append(q_weight)

		details.append({
			"question_id": q_id,
			"question": q_text,
			"related_fact_ids": related_fact_ids,
			"patient_answer": patient_answer,
			"max_similarity": score["max_similarity"],
			"hit": bool(hit),
			"weight": round(float(q_weight), 4),
		})

	hit_rate = float(sum(hits) / len(hits)) if hits else 0.0
	total_weight = float(sum(weights)) if weights else 0.0
	hit_weight = float(sum(w for h, w in zip(hits, weights) if h == 1))
	weighted_recall = (hit_weight / total_weight) if total_weight else 0.0

	return {
		"comprehension_hit_rate": round(hit_rate, 4),
		"weighted_critical_recall": round(float(weighted_recall), 4),
		"question_count": len(questions),
		"hits": int(sum(hits)),
		"details": details,
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
			"mean_comprehension_hit_rate": round(
				float(np.mean([x["comprehension_hit_rate"] for x in items])), 4
			),
			"mean_weighted_critical_recall": round(
				float(np.mean([x["weighted_critical_recall"] for x in items])), 4
			),
		})
	return sorted(out, key=lambda x: x["mode"])


def run_evaluation(
	conversations_dir: Path = CONVERSATIONS_DIR,
	results_dir: Path = RESULTS_DIR,
	threshold: float = DEFAULT_THRESHOLD,
	model_name: str = EMBEDDING_MODEL,
	answer_model: str = ANSWER_MODEL,
) -> List[Dict]:
	questions = load_questions()
	ground_truth = load_ground_truth()

	rows = []
	for conv_file in sorted(conversations_dir.glob("*.json")):
		if conv_file.name == "conversation_index_all.json":
			continue

		conv = load_conversation(conv_file)
		procedure = (conv.get("metadata", {}).get("procedure") or "").strip()

		q_key = normalize_procedure_key(procedure, list(questions.keys()))
		gt_key = normalize_procedure_key(procedure, list(ground_truth.keys()))

		proc_questions = questions.get(q_key, []) if q_key else []
		fact_map = ground_truth.get(gt_key, {}) if gt_key else {}

		metrics = evaluate_single_conversation(
			conv=conv,
			questions=proc_questions,
			fact_map=fact_map,
			threshold=threshold,
			model_name=model_name,
			answer_model=answer_model,
		)

		rows.append({
			"file": conv_file.name,
			"procedure": procedure,
			"mode": conv.get("metadata", {}).get("mode"),
			"chatbot_model": conv.get("metadata", {}).get("chatbot_model"),
			"patient_model": conv.get("metadata", {}).get("patient_model"),
			"threshold": threshold,
			"answer_model": answer_model,
			**metrics,
		})

	save_json(results_dir / f"comprehension_evaluation_thr{int(threshold * 100)}.json", rows)
	save_json(results_dir / f"comprehension_summary_thr{int(threshold * 100)}.json", summarize(rows))
	return rows


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Recipient-side comprehension evaluation")
	parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
	parser.add_argument("--model", type=str, default=EMBEDDING_MODEL)
	parser.add_argument("--answer-model", type=str, default=ANSWER_MODEL)
	args = parser.parse_args()

	rows = run_evaluation(
		threshold=args.threshold,
		model_name=args.model,
		answer_model=args.answer_model,
	)
	print(f"Saved comprehension evaluation for {len(rows)} conversations")
