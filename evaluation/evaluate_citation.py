import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from utils.llm_config import make_api_call


CONVERSATIONS_DIR = Path("data/conversations/v1")
RESULTS_DIR = Path("data/evaluation_results")
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

CITATION_PATTERN = re.compile(r"\[Quelle\s*(\d+)\]")

FULL_SUPPORT = "full_support"
PARTIAL_SUPPORT = "partial_support"
NO_SUPPORT = "no_support"

_judge_cache: Dict[str, Dict[str, str]] = {}


def load_json(path: Path) -> Dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def save_json(path: Path, data) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)


def discover_conversation_files(conversations_dir: Path) -> List[Path]:
	files = [
		p for p in conversations_dir.rglob("*.json")
		if not p.name.startswith("conversation_index_")
	]
	return sorted(files)


def split_response_into_sentences(response_text: str) -> List[str]:
	if not response_text:
		return []

	# Keep bullets and line-level statements visible before sentence split.
	chunks = []
	for raw_line in response_text.replace("\r", "").split("\n"):
		line = raw_line.strip()
		if not line:
			continue
		line = re.sub(r"^[\-•*]\s*", "", line)
		chunks.append(line)

	sentences: List[str] = []
	for chunk in chunks:
		parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9(\[\"„])", chunk)
		for part in parts:
			cleaned = part.strip()
			if cleaned:
				sentences.append(cleaned)

	return sentences


def strip_citations(text: str) -> str:
	without = CITATION_PATTERN.sub("", text)
	return re.sub(r"\s+", " ", without).strip()


def is_factual_sentence(sentence: str) -> bool:
	s = sentence.strip()
	if not s:
		return False

	lower = s.lower()

	if s.endswith("?"):
		return False

	# Ignore conversational filler and empathy-only utterances.
	filler_prefixes = (
		"das ist verständlich",
		"das ist sehr verständlich",
		"ich verstehe",
		"wenn sie möchten",
		"okay",
		"alles klar",
		"danke",
	)
	if lower.startswith(filler_prefixes):
		return False

	# Treat standalone "gern/gerne" intro lines as filler, but keep longer factual lines.
	if re.match(r"^gerne?[\s,.:\-–—]*$", lower):
		return False

	# Very short fragments are usually conversational, not factual claims.
	if len(strip_citations(s)) < 20:
		return False

	return True


def extract_atomic_claims(response_text: str) -> List[Dict]:
	claims = []
	for sentence in split_response_into_sentences(response_text):
		claim_text = strip_citations(sentence)
		citations = [int(x) for x in CITATION_PATTERN.findall(sentence)]

		if not is_factual_sentence(sentence):
			continue

		claims.append({
			"claim_text": claim_text,
			"citations": citations,
			"raw_sentence": sentence,
		})

	return claims


def build_source_map(turn: Dict) -> Dict[int, Dict]:
	mapping: Dict[int, Dict] = {}

	retrieved_chunks = turn.get("retrieved_chunks", []) or []
	citation_mapping = turn.get("citations", {}).get("citation_mapping", []) or []

	for m in citation_mapping:
		idx = m.get("citation_index")
		if not isinstance(idx, int):
			continue

		# In current generation pipeline, citation index maps to retrieved chunk position.
		chunk = retrieved_chunks[idx - 1] if 1 <= idx <= len(retrieved_chunks) else None

		mapping[idx] = {
			"citation_index": idx,
			"document_id": m.get("document_id"),
			"file_path": m.get("file_path"),
			"retrieval_score": m.get("score"),
			"chunk_content": (chunk or {}).get("content"),
			"chunk_meta": (chunk or {}).get("meta"),
			"chunk_id": (chunk or {}).get("id"),
		}

	return mapping


def _judge_key(claim: str, source_chunk: str, model_name: str) -> str:
	raw = f"{model_name}\n{claim}\n{source_chunk}"
	return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _normalize_label(label: str) -> str:
	if not label:
		return NO_SUPPORT

	lower = label.lower().strip()
	if lower in {"full", "full_support", "fully supported", "supported"}:
		return FULL_SUPPORT
	if lower in {"partial", "partial_support", "partially supported"}:
		return PARTIAL_SUPPORT
	if lower in {"none", "no support", "no_support", "unsupported"}:
		return NO_SUPPORT

	if "full" in lower:
		return FULL_SUPPORT
	if "partial" in lower:
		return PARTIAL_SUPPORT
	return NO_SUPPORT

def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def judge_claim_support(claim: str, source_chunk: str, model_name: str) -> Dict[str, str]:
	if not source_chunk:
		return {
			"support_label": NO_SUPPORT,
			"rationale": "Missing mapped source chunk for this citation.",
			"raw_judge_output": "",
		}

	key = _judge_key(claim, source_chunk, model_name)
	if key in _judge_cache:
		return _judge_cache[key]

	system_message = (
		"You are a strict factual support judge. Compare CLAIM against SOURCE CHUNK only. "
		"Do not use outside knowledge. Return JSON only."
	)

	prompt = f"""
Task: Classify whether SOURCE CHUNK supports CLAIM.

Labels:
- full_support: Claim is directly and fully supported by the source.
- partial_support: Source supports only part of the claim or is less specific.
- no_support: Claim is unsupported or contradicted by the source.

Return JSON with keys:
- label: one of full_support, partial_support, no_support
- rationale: brief explanation (max 2 sentences)

CLAIM:
{claim}

SOURCE CHUNK:
{source_chunk}
""".strip()

	try:
		raw = make_api_call(
			prompt=prompt,
			model_name=model_name,
			temperature=0.0,
			system_message=system_message,
		)
	except Exception as e:
		error_details = str(e)
		if hasattr(e, 'response') and hasattr(e.response, 'text'):
			error_details += f" | Reason: {e.response.text}"

		print(f"API ERROR: {error_details}")  # Print it to your terminal
		result = {
			"support_label": NO_SUPPORT,
			"rationale": f"Judge call failed: {e}",
			"raw_judge_output": "",
		}
		_judge_cache[key] = result
		return result

	label = NO_SUPPORT
	rationale = ""

	try:
		clean_raw = _extract_json_object(raw)
		parsed = json.loads(clean_raw)
		label = _normalize_label(str(parsed.get("label", "")))
		rationale = str(parsed.get("rationale", "")).strip()
	except json.JSONDecodeError:
		label = _normalize_label(raw)
		rationale = "Parser fallback used because judge did not return valid JSON."

	result = {
		"support_label": label,
		"rationale": rationale,
		"raw_judge_output": raw,
	}
	_judge_cache[key] = result
	return result


def evaluate_conversation_file(path: Path, judge_model: str) -> Dict:
	data = load_json(path)
	turns = data.get("conversation", []) or []

	pair_rows: List[Dict] = []
	total_claims = 0
	claims_with_citations = 0

	for turn in turns:
		response_text = turn.get("chatbot_response", "")
		claims = extract_atomic_claims(response_text)
		source_map = build_source_map(turn)

		total_claims += len(claims)
		claims_with_citations += sum(1 for c in claims if c["citations"])

		for claim_idx, claim in enumerate(claims, start=1):
			for citation_idx in claim["citations"]:
				source_info = source_map.get(citation_idx, {})
				source_chunk = source_info.get("chunk_content", "")

				judge = judge_claim_support(
					claim=claim["claim_text"],
					source_chunk=source_chunk,
					model_name=judge_model,
				)

				pair_rows.append({
					"file": str(path),
					"turn": turn.get("turn"),
					"claim_index_in_turn": claim_idx,
					"claim_text": claim["claim_text"],
					"raw_sentence": claim["raw_sentence"],
					"citation_index": citation_idx,
					"support_label": judge["support_label"],
					"judge_rationale": judge["rationale"],
					"judge_raw_output": judge["raw_judge_output"],
					"source": {
						"file_path": source_info.get("file_path"),
						"document_id": source_info.get("document_id"),
						"retrieval_score": source_info.get("retrieval_score"),
						"chunk_id": source_info.get("chunk_id"),
						"chunk_content": source_chunk,
						"chunk_meta": source_info.get("chunk_meta"),
					},
				})

	return {
		"file": str(path),
		"total_factual_claims": total_claims,
		"claims_with_citations": claims_with_citations,
		"citation_pairs": pair_rows,
	}


def calculate_metrics(pair_rows: List[Dict], total_claims: int, claims_with_citations: int) -> Dict:
	total_citations = len(pair_rows)
	full_count = sum(1 for r in pair_rows if r["support_label"] == FULL_SUPPORT)
	partial_count = sum(1 for r in pair_rows if r["support_label"] == PARTIAL_SUPPORT)
	none_count = sum(1 for r in pair_rows if r["support_label"] == NO_SUPPORT)

	strict_supported = full_count
	relaxed_supported = full_count + partial_count

	strict_precision = strict_supported / total_citations if total_citations else 0.0
	relaxed_precision = relaxed_supported / total_citations if total_citations else 0.0
	support_coverage = claims_with_citations / total_claims if total_claims else 0.0

	full_pct = full_count / total_citations if total_citations else 0.0
	partial_pct = partial_count / total_citations if total_citations else 0.0
	none_pct = none_count / total_citations if total_citations else 0.0

	return {
		"citation_precision": {
			"strict_full_only": round(strict_precision, 4),
			"relaxed_full_plus_partial": round(relaxed_precision, 4),
			"supported_citations_strict": strict_supported,
			"supported_citations_relaxed": relaxed_supported,
			"total_citations": total_citations,
		},
		"support_coverage": {
			"claims_with_citations": claims_with_citations,
			"total_factual_claims": total_claims,
			"coverage": round(support_coverage, 4),
		},
		"support_distribution": {
			"full_support": {
				"count": full_count,
				"percentage": round(full_pct, 4),
			},
			"partial_support": {
				"count": partial_count,
				"percentage": round(partial_pct, 4),
			},
			"no_support": {
				"count": none_count,
				"percentage": round(none_pct, 4),
			},
		},
	}


def run_evaluation(
	conversations_dir: Path = CONVERSATIONS_DIR,
	results_dir: Path = RESULTS_DIR,
	judge_model: str = DEFAULT_JUDGE_MODEL,
	max_files: int = None,
) -> Tuple[Dict, List[Dict]]:
	files = discover_conversation_files(conversations_dir)
	if max_files is not None:
		files = files[:max_files]

	all_pairs: List[Dict] = []
	file_summaries: List[Dict] = []
	total_claims = 0
	claims_with_citations = 0

	for i, file_path in enumerate(files, start=1):
		print(f"[{i}/{len(files)}] Evaluating {file_path.name}")
		file_result = evaluate_conversation_file(file_path, judge_model)

		file_pairs = file_result["citation_pairs"]
		file_total_claims = file_result["total_factual_claims"]
		file_claims_with_citations = file_result["claims_with_citations"]

		total_claims += file_total_claims
		claims_with_citations += file_claims_with_citations
		all_pairs.extend(file_pairs)

		file_metrics = calculate_metrics(file_pairs, file_total_claims, file_claims_with_citations)
		file_summaries.append({
			"file": file_result["file"],
			"total_factual_claims": file_total_claims,
			"claims_with_citations": file_claims_with_citations,
			**file_metrics,
		})

	global_metrics = calculate_metrics(all_pairs, total_claims, claims_with_citations)
	summary = {
		"judge_model": judge_model,
		"files_evaluated": len(files),
		**global_metrics,
	}

	save_json(results_dir / "citation_pair_judgments.json", all_pairs)
	save_json(results_dir / "citation_metrics.json", summary)
	save_json(results_dir / "citation_metrics_per_file.json", file_summaries)

	return summary, all_pairs


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Citation support evaluation")
	parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
	parser.add_argument("--max-files", type=int, default=None)
	args = parser.parse_args()

	summary, rows = run_evaluation(
		judge_model=args.judge_model,
		max_files=args.max_files,
	)

	print("\nCitation Evaluation Summary")
	print(json.dumps(summary, indent=2, ensure_ascii=False))
	print(f"Saved {len(rows)} citation pair judgments")
