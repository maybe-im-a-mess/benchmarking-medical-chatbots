import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from utils.llm_config import make_api_call

GROUND_TRUTH_PATH = Path("data/ground_truth.json")
CONVERSATIONS_DIR = Path(os.getenv("CONVERSATIONS_DIR", "data/conversations/v1"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "data/evaluation_results"))

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_TIMEOUT = 180
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

IMPORTANCE_WEIGHTS = {
    "Critical": 4.0,
    "High": 3.0,
    "Medium": 2.0,
    "Low": 1.0,
}


SYSTEM_PROMPT = (
    "You are a clinical evaluator.\n\n"
    "Chatbot Conversation: [Insert full chatbot text here]\n"
    "Required Topic: [Insert single Ground Truth fact here]\n\n"
    "Task: Did the chatbot successfully cover or address this Required Topic? "
    "(Note: Asking a relevant medical question to the patient counts as covering the topic).\n"
    "Answer ONLY with a JSON: {\"is_covered\": true/false}"
)


def make_safe_stem(name: str) -> str:
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
                content = (sub.get("content") or "").strip()
                if not content:
                    continue
                facts.append(
                    {
                        "fact_id": sub.get("fact_id"),
                        "content": content,
                        "importance": sub.get("importance", "Medium"),
                    }
                )
        out[stem] = facts
    return out


def normalize_procedure_key(procedure: str, candidates: List[str]) -> str:
    if not procedure:
        return ""
    for c in candidates:
        if c.lower() == procedure.lower():
            return c
    return ""


def load_conversation(file_path: Path) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chatbot_conversation_text(conversation: Dict) -> str:
    lines = []
    for turn in conversation.get("conversation", []):
        patient_q = (turn.get("patient_question") or "").strip()
        chatbot_r = (turn.get("chatbot_response") or "").strip()

        if patient_q:
            lines.append(f"Patient: {patient_q}")
        if chatbot_r:
            lines.append(f"Chatbot: {chatbot_r}")

    return "\n".join(lines)


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start : end + 1]


def judge_fact_coverage(
    chatbot_conversation_text: str,
    required_fact: str,
    model_name: str = JUDGE_MODEL,
) -> Dict:
    prompt = (
        f"Chatbot Conversation:\n{chatbot_conversation_text}\n\n"
        f"Required Topic:\n{required_fact}"
    )

    raw = ""
    api_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = make_api_call(
                prompt=prompt,
                model_name=model_name,
                temperature=0.0,
                timeout=JUDGE_TIMEOUT,
                system_message=SYSTEM_PROMPT,
            )
            raw = (raw or "").strip()
            api_error = None
            break
        except Exception as e:
            api_error = str(e)
            if attempt < MAX_RETRIES:
                wait_s = RETRY_BACKOFF_SECONDS * attempt
                print(f"  Warning: judge call failed (attempt {attempt}/{MAX_RETRIES}). Retrying in {wait_s:.1f}s...")
                time.sleep(wait_s)
            else:
                print(f"  Warning: judge call failed after {MAX_RETRIES} attempts. Marking fact as not covered.")

    is_covered = False
    parse_error = None

    if api_error is not None:
        parse_error = f"api_error: {api_error}"
        is_covered = False
    else:
        try:
            parsed = json.loads(raw)
            is_covered = bool(parsed.get("is_covered", False))
        except Exception:
            try:
                parsed = json.loads(_extract_json_object(raw))
                is_covered = bool(parsed.get("is_covered", False))
            except Exception as e:
                parse_error = str(e)
                is_covered = False

    return {
        "is_covered": is_covered,
        "raw_judge_response": raw,
        "parse_error": parse_error,
    }


def evaluate_single_conversation(
    conv: Dict,
    gt_facts: List[Dict],
    model_name: str = JUDGE_MODEL,
) -> Dict:
    chatbot_text = build_chatbot_conversation_text(conv)

    if not gt_facts:
        return {
            "hit_rate": 0.0,
            "weighted_critical_recall": 0.0,
            "hits": 0,
            "total_topics": 0,
            "matched_pairs": [],
        }

    hits = []
    weights = []
    matched_pairs = []

    for fact in gt_facts:
        fact_text = fact["content"]
        importance = fact.get("importance", "Medium")
        weight = IMPORTANCE_WEIGHTS.get(importance, 2.0)

        judge = judge_fact_coverage(
            chatbot_conversation_text=chatbot_text,
            required_fact=fact_text,
            model_name=model_name,
        )

        hit = bool(judge["is_covered"])
        hits.append(1 if hit else 0)
        weights.append(weight)

        matched_pairs.append(
            {
                "fact_id": fact.get("fact_id"),
                "importance": importance,
                "topic": fact_text,
                "hit": hit,
                "judge_response": judge["raw_judge_response"],
                "parse_error": judge["parse_error"],
            }
        )

    hit_rate = float(sum(hits) / len(gt_facts)) if gt_facts else 0.0
    total_weight = float(sum(weights)) if weights else 0.0
    hit_weight = float(sum(w for h, w in zip(hits, weights) if h == 1))
    weighted_critical_recall = (hit_weight / total_weight) if total_weight else 0.0

    return {
        "hit_rate": round(hit_rate, 4),
        "weighted_critical_recall": round(float(weighted_critical_recall), 4),
        "hits": int(sum(hits)),
        "total_topics": len(gt_facts),
        "matched_pairs": matched_pairs,
    }


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_evaluation(
    conversations_dir: Path = CONVERSATIONS_DIR,
    results_dir: Path = RESULTS_DIR,
    model_name: str = JUDGE_MODEL,
) -> List[Dict]:
    gt = load_ground_truth()
    coverage_dir = results_dir / "coverage_llm_judge"
    per_conversation_dir = coverage_dir / "per_conversation"
    aggregate_path = coverage_dir / "coverage_llm_judge_evaluation.json"

    target_file = os.getenv("COVERAGE_TARGET_FILE")
    conversation_files = [
        fp for fp in sorted(conversations_dir.glob("*.json")) if not fp.name.startswith("conversation_index_")
    ]
    if target_file:
        conversation_files = [fp for fp in conversation_files if fp.name == target_file]

    rows = []
    processed_files = set()

    if aggregate_path.exists() and not target_file:
        try:
            with open(aggregate_path, "r", encoding="utf-8") as f:
                existing_rows = json.load(f)
            if isinstance(existing_rows, list):
                rows = existing_rows
                processed_files = {r.get("file") for r in rows if r.get("file")}
                if processed_files:
                    print(f"Resuming from checkpoint: {len(processed_files)} conversations already processed.")
        except Exception:
            rows = []
            processed_files = set()

    if processed_files:
        conversation_files = [fp for fp in conversation_files if fp.name not in processed_files]

    total_files = len(conversation_files)

    for idx, conv_file in enumerate(conversation_files, start=1):
        print(f"[{idx}/{total_files}] Evaluating: {conv_file.name}")

        conv = load_conversation(conv_file)
        procedure = (conv.get("metadata", {}).get("procedure") or "").strip()
        proc_key = normalize_procedure_key(procedure, list(gt.keys()))
        gt_facts = gt.get(proc_key, []) if proc_key else []

        metrics = evaluate_single_conversation(
            conv=conv,
            gt_facts=gt_facts,
            model_name=model_name,
        )

        row = {
            "file": conv_file.name,
            "procedure": procedure,
            "mode": conv.get("metadata", {}).get("mode"),
            "chatbot_model": conv.get("metadata", {}).get("chatbot_model"),
            "patient_model": conv.get("metadata", {}).get("patient_model"),
            "judge_model": model_name,
            **metrics,
        }
        rows.append(row)

        safe_stem = make_safe_stem(conv_file.stem)
        save_json(per_conversation_dir / f"{safe_stem}.json", row)

        # Checkpoint after each conversation for interruption-safe progress.
        save_json(aggregate_path, rows)

    save_json(aggregate_path, rows)
    print(f"LLM-judge coverage evaluation finished. Saved {len(rows)} conversation results.")
    return rows


if __name__ == "__main__":
    run_evaluation()
