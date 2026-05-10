import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

GROUND_TRUTH_PATH = Path("data/ground_truth.json")
CONVERSATIONS_DIR = Path(os.getenv("CONVERSATIONS_DIR", "data/conversations/v1"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "data/evaluation_results"))

BI_ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/nli-deberta-v3-base"

BI_ENCODER_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
NLI_DEVICE = os.getenv("NLI_DEVICE", BI_ENCODER_DEVICE)

TOP_K = 3
DEFAULT_ENTAILMENT_THRESHOLD = 0.5

IMPORTANCE_WEIGHTS = {
	"Critical": 4.0,
	"High": 3.0,
	"Medium": 2.0,
	"Low": 1.0,
}

_model_cache = {}
_nli_cache = {}


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


def get_embedding_model(model_name: str = BI_ENCODER_MODEL):
	if model_name not in _model_cache:
		from sentence_transformers import SentenceTransformer

		print(f"Loading bi-encoder model: {model_name} (device={BI_ENCODER_DEVICE})")
		try:
			_model_cache[model_name] = SentenceTransformer(model_name, device=BI_ENCODER_DEVICE)
		except RuntimeError as e:
			if BI_ENCODER_DEVICE != "cpu":
				print(f"Device '{BI_ENCODER_DEVICE}' failed ({e}). Falling back to CPU.")
				_model_cache[model_name] = SentenceTransformer(model_name, device="cpu")
			else:
				raise
	return _model_cache[model_name]


def encode(texts: List[str], model_name: str = BI_ENCODER_MODEL) -> np.ndarray:
	if not texts:
		return np.empty((0, 0), dtype=float)
	model = get_embedding_model(model_name)
	return np.array(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def _resolve_torch_device(device_name: str):
	import torch

	device_name = (device_name or "cpu").lower()
	if device_name == "cuda" and torch.cuda.is_available():
		return torch.device("cuda")
	if device_name == "mps" and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def get_nli_model(model_name: str = CROSS_ENCODER_MODEL):
	if model_name not in _nli_cache:
		import torch
		from transformers import AutoModelForSequenceClassification, AutoTokenizer

		device = _resolve_torch_device(NLI_DEVICE)
		print(f"Loading cross-encoder NLI model: {model_name} (device={device.type})")

		try:
			tokenizer = AutoTokenizer.from_pretrained(model_name)
			model = AutoModelForSequenceClassification.from_pretrained(model_name)
			model.to(device)
		except RuntimeError as e:
			if device.type != "cpu":
				print(f"Device '{device.type}' failed ({e}). Falling back to CPU.")
				device = torch.device("cpu")
				tokenizer = AutoTokenizer.from_pretrained(model_name)
				model = AutoModelForSequenceClassification.from_pretrained(model_name)
				model.to(device)
			else:
				raise

		model.eval()
		id2label = getattr(model.config, "id2label", {}) or {}
		entailment_index = None
		for idx, label in id2label.items():
			if "entail" in str(label).lower():
				entailment_index = int(idx)
				break

		if entailment_index is None:
			raise ValueError(
				f"Could not find entailment label in id2label for model '{model_name}': {id2label}"
			)

		_nli_cache[model_name] = {
			"tokenizer": tokenizer,
			"model": model,
			"device": device,
			"entailment_index": entailment_index,
		}

	return _nli_cache[model_name]


def score_entailment_pairs(pairs: List[List[str]], model_name: str = CROSS_ENCODER_MODEL) -> List[float]:
	if not pairs:
		return []

	import torch

	nli = get_nli_model(model_name)
	tokenizer = nli["tokenizer"]
	model = nli["model"]
	device = nli["device"]
	entailment_index = nli["entailment_index"]

	premises = [p[0] for p in pairs]
	hypotheses = [p[1] for p in pairs]

	encoded = tokenizer(
		premises,
		hypotheses,
		padding=True,
		truncation=True,
		max_length=512,
		return_tensors="pt",
	)
	encoded = {k: v.to(device) for k, v in encoded.items()}

	with torch.no_grad():
		logits = model(**encoded).logits
		probs = torch.softmax(logits, dim=-1)
		ent = probs[:, entailment_index].detach().cpu().numpy()

	return [float(x) for x in ent]


def load_conversation(file_path: Path) -> Dict:
	with open(file_path, "r", encoding="utf-8") as f:
		return json.load(f)


def get_chatbot_utterances(conversation: Dict) -> List[str]:
	utterances = []
	for turn in conversation.get("conversation", []):
		text = (turn.get("chatbot_response") or "").strip()
		if not text:
			continue

		# Split by sentence boundaries and newlines to capture atomic mentions.
		raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
		for sentence in raw_sentences:
			clean_sentence = sentence.strip(" \t\n\r-*•")
			if len(clean_sentence) > 10:
				utterances.append(clean_sentence)

	return utterances


def evaluate_single_conversation(
	conv: Dict,
	gt_facts: List[Dict],
	threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
	top_k: int = TOP_K,
	bi_model_name: str = BI_ENCODER_MODEL,
	nli_model_name: str = CROSS_ENCODER_MODEL,
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

	if top_k < 1:
		top_k = 1

	emb_topics = encode(gt_texts, bi_model_name)
	emb_utts = encode(utterances, bi_model_name)
	cosine_sim = emb_topics @ emb_utts.T

	hits = []
	matched_pairs = []

	for i, topic_text in enumerate(gt_texts):
		row_sim = cosine_sim[i]
		k = min(top_k, len(utterances))
		top_indices = np.argsort(-row_sim)[:k]

		# NLI is directional: premise -> hypothesis.
		# We test whether the chatbot utterance supports the ground-truth fact.
		candidate_pairs = [[utterances[int(j)], topic_text] for j in top_indices]
		entailment_scores = score_entailment_pairs(candidate_pairs, nli_model_name)

		best_pos = int(np.argmax(entailment_scores)) if entailment_scores else 0
		best_idx = int(top_indices[best_pos]) if len(top_indices) else None
		best_ent = float(entailment_scores[best_pos]) if entailment_scores else 0.0
		best_cos = float(row_sim[best_idx]) if best_idx is not None else 0.0

		hit = best_ent >= threshold
		hits.append(1 if hit else 0)

		top_candidates = []
		for rank, idx in enumerate(top_indices, start=1):
			top_candidates.append({
				"rank": rank,
				"utterance_index": int(idx),
				"utterance": utterances[int(idx)],
				"bi_encoder_similarity": round(float(row_sim[int(idx)]), 4),
				"entailment_score": round(float(entailment_scores[rank - 1]), 4),
			})

		matched_pairs.append({
			"fact_id": gt_facts[i].get("fact_id"),
			"importance": gt_facts[i].get("importance", "Medium"),
			"topic": topic_text,
			"matched_utterance_index": best_idx,
			"matched_utterance": utterances[best_idx] if best_idx is not None else None,
			"bi_encoder_similarity": round(best_cos, 4),
			"entailment_score": round(best_ent, 4),
			"hit": bool(hit),
			"top_candidates": top_candidates,
		})

	hit_rate = float(sum(hits) / len(gt_texts)) if gt_texts else 0.0
	total_weight = float(sum(gt_weights)) if gt_weights else 0.0
	hit_weight = float(sum(w for h, w in zip(hits, gt_weights) if h == 1))
	weighted_critical_recall = (hit_weight / total_weight) if total_weight else 0.0

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
	threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
	top_k: int = TOP_K,
	bi_model_name: str = BI_ENCODER_MODEL,
	nli_model_name: str = CROSS_ENCODER_MODEL,
) -> List[Dict]:
	gt = load_ground_truth()
	tag = f"thr{int(threshold * 100)}"
	coverage_dir = results_dir / "coverage_sbert"
	per_conversation_dir = coverage_dir / "per_conversation"
	aggregate_path = coverage_dir / f"coverage_sbert_evaluation_{tag}.json"

	# Trigger model loads once up front so progress logs are clearer.
	get_embedding_model(bi_model_name)
	get_nli_model(nli_model_name)
	print("Models are ready. Starting SBERT + Cross-Encoder coverage evaluation...")

	target_file = os.getenv("COVERAGE_TARGET_FILE")
	conversation_files = [
		fp for fp in sorted(conversations_dir.glob("*.json"))
		if not fp.name.startswith("conversation_index_")
	]
	if target_file:
		conversation_files = [fp for fp in conversation_files if fp.name == target_file]

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
			top_k=top_k,
			bi_model_name=bi_model_name,
			nli_model_name=nli_model_name,
		)

		row = {
			"file": conv_file.name,
			"procedure": procedure,
			"mode": conv.get("metadata", {}).get("mode"),
			"chatbot_model": conv.get("metadata", {}).get("chatbot_model"),
			"patient_model": conv.get("metadata", {}).get("patient_model"),
			"threshold": threshold,
			"top_k": int(top_k),
			"bi_encoder_model": bi_model_name,
			"cross_encoder_model": nli_model_name,
			**metrics,
		}
		rows.append(row)

		safe_stem = make_safe_stem(conv_file.stem)
		save_json(per_conversation_dir / f"{safe_stem}_{tag}.json", row)

		# Checkpoint after each conversation for interruption-safe progress.
		save_json(aggregate_path, rows)

	save_json(aggregate_path, rows)
	print(f"SBERT coverage evaluation finished. Saved {len(rows)} conversation results.")
	return rows


if __name__ == "__main__":
	run_evaluation()
