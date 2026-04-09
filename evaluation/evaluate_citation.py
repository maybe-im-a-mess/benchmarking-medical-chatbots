import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

CONVERSATIONS_DIR = Path("data/conversations")
RESULTS_DIR = Path("data/evaluation_results")

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_SUPPORT_THRESHOLD = 0.6

_model_cache = {}


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


def split_into_sentences(text: str) -> List[str]:
	if not text:
		return []
	parts = re.split(r"(?<=[.!?])\s+", text.strip())
	return [p.strip() for p in parts if p.strip()]


def extract_citations_from_sentence(sentence: str) -> List[int]:
	return [int(x) for x in re.findall(r"\[Quelle\s+(\d+)\]", sentence)]


def remove_citation_tags(text: str) -> str:
	return re.sub(r"\[Quelle\s+\d+\]", "", text).strip()


def evaluate_turn_citations(
	turn: Dict,
	procedure: str,
	support_threshold: float,
	model_name: str,
) -> Dict:
	response = turn.get("chatbot_response", "")
	chunks = turn.get("retrieved_chunks", [])

	sentences = split_into_sentences(response)
	cited_sentence_records = []

	total_cited_claims = 0
	supported_cited_claims = 0
	valid_index_claims = 0
	same_doc_claims = 0

	procedure_file = f"{procedure}.md" if procedure else None

	for sent in sentences:
		cited_indices = extract_citations_from_sentence(sent)
		if not cited_indices:
			continue

		claim_text = remove_citation_tags(sent)
		if not claim_text:
			continue

		emb_claim = encode([claim_text], model_name)
		claim_supported = False

		for idx in cited_indices:
			total_cited_claims += 1
			chunk_pos = idx - 1
			if chunk_pos < 0 or chunk_pos >= len(chunks):
				cited_sentence_records.append({
					"sentence": sent,
					"citation_index": idx,
					"valid_index": False,
					"support_similarity": None,
					"supported": False,
					"same_procedure_document": False,
				})
				continue

			valid_index_claims += 1
			chunk = chunks[chunk_pos]
			chunk_text = chunk.get("content", "")
			chunk_file = (chunk.get("meta", {}) or {}).get("file_path")

			emb_chunk = encode([chunk_text], model_name)
			sim = float((emb_claim @ emb_chunk.T)[0, 0]) if emb_chunk.size else 0.0
			supported = sim >= support_threshold
			if supported:
				claim_supported = True

			same_doc = bool(procedure_file and chunk_file == procedure_file)
			if same_doc:
				same_doc_claims += 1

			cited_sentence_records.append({
				"sentence": sent,
				"citation_index": idx,
				"valid_index": True,
				"chunk_file": chunk_file,
				"support_similarity": round(sim, 4),
				"supported": supported,
				"same_procedure_document": same_doc,
			})

		if claim_supported:
			supported_cited_claims += 1

	citation_meta = turn.get("citations", {})
	reported_total_citations = citation_meta.get("total_citations", 0)

	citation_precision = (
		supported_cited_claims / total_cited_claims if total_cited_claims else 0.0
	)
	valid_index_rate = valid_index_claims / total_cited_claims if total_cited_claims else 0.0
	same_doc_rate = same_doc_claims / total_cited_claims if total_cited_claims else 0.0

	return {
		"turn": turn.get("turn"),
		"reported_total_citations": reported_total_citations,
		"parsed_total_cited_claims": total_cited_claims,
		"supported_cited_claims": supported_cited_claims,
		"citation_precision": round(float(citation_precision), 4),
		"valid_index_rate": round(float(valid_index_rate), 4),
		"same_procedure_doc_rate": round(float(same_doc_rate), 4),
		"details": cited_sentence_records,
	}


def aggregate_turn_results(turn_results: List[Dict]) -> Dict:
	total_reported = sum(t["reported_total_citations"] for t in turn_results)
	total_parsed = sum(t["parsed_total_cited_claims"] for t in turn_results)
	total_supported = sum(t["supported_cited_claims"] for t in turn_results)

	weighted_valid = sum(
		t["valid_index_rate"] * t["parsed_total_cited_claims"] for t in turn_results
	)
	weighted_same_doc = sum(
		t["same_procedure_doc_rate"] * t["parsed_total_cited_claims"] for t in turn_results
	)

	if total_parsed > 0:
		citation_accuracy = total_supported / total_parsed
		valid_index_rate = weighted_valid / total_parsed
		same_doc_rate = weighted_same_doc / total_parsed
	else:
		citation_accuracy = 0.0
		valid_index_rate = 0.0
		same_doc_rate = 0.0

	turns_with_no_citations = sum(1 for t in turn_results if t["parsed_total_cited_claims"] == 0)

	return {
		"reported_total_citations": total_reported,
		"parsed_total_cited_claims": total_parsed,
		"supported_cited_claims": total_supported,
		"citation_accuracy": round(float(citation_accuracy), 4),
		"valid_index_rate": round(float(valid_index_rate), 4),
		"same_procedure_doc_rate": round(float(same_doc_rate), 4),
		"turns_with_no_citations": turns_with_no_citations,
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
			"mean_citation_accuracy": round(
				float(np.mean([x["citation_accuracy"] for x in items])), 4
			),
			"mean_same_procedure_doc_rate": round(
				float(np.mean([x["same_procedure_doc_rate"] for x in items])), 4
			),
			"mean_valid_index_rate": round(
				float(np.mean([x["valid_index_rate"] for x in items])), 4
			),
		})
	return sorted(out, key=lambda x: x["mode"])


def run_evaluation(
	conversations_dir: Path = CONVERSATIONS_DIR,
	results_dir: Path = RESULTS_DIR,
	support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
	model_name: str = EMBEDDING_MODEL,
) -> List[Dict]:
	rows = []

	for conv_file in sorted(conversations_dir.glob("*.json")):
		if conv_file.name == "conversation_index_all.json":
			continue

		conv = load_conversation(conv_file)
		procedure = conv.get("metadata", {}).get("procedure", "")
		turn_results = [
			evaluate_turn_citations(
				turn=turn,
				procedure=procedure,
				support_threshold=support_threshold,
				model_name=model_name,
			)
			for turn in conv.get("conversation", [])
		]

		agg = aggregate_turn_results(turn_results)
		rows.append({
			"file": conv_file.name,
			"procedure": procedure,
			"mode": conv.get("metadata", {}).get("mode"),
			"chatbot_model": conv.get("metadata", {}).get("chatbot_model"),
			"support_threshold": support_threshold,
			**agg,
			"turn_details": turn_results,
		})

	save_json(results_dir / f"citation_evaluation_thr{int(support_threshold * 100)}.json", rows)
	save_json(results_dir / f"citation_summary_thr{int(support_threshold * 100)}.json", summarize(rows))
	return rows


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Citation accuracy evaluation")
	parser.add_argument("--threshold", type=float, default=DEFAULT_SUPPORT_THRESHOLD)
	parser.add_argument("--model", type=str, default=EMBEDDING_MODEL)
	args = parser.parse_args()

	rows = run_evaluation(support_threshold=args.threshold, model_name=args.model)
	print(f"Saved citation evaluation for {len(rows)} conversations")
