import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from utils.llm_config import make_api_call


MANDATORY_QUESTIONS_PATH = Path("data/mandatory_questions.json")
CONVERSATIONS_DIR = Path(os.getenv("CONVERSATIONS_DIR", "data/conversations/v1"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "data/evaluation_results"))
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
COMPLIANCE_THRESHOLD = 0.75

JSON_ARRAY_PATTERN = re.compile(r"\[.*?\]", re.DOTALL)
CACHE_PATH = RESULTS_DIR / "mandatory_q_cache_batch.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def load_cache() -> Dict:
    if CACHE_PATH.exists():
        return load_json(CACHE_PATH)
    return {}


def save_cache(cache: Dict) -> None:
    save_json(CACHE_PATH, cache)


def load_mandatory_question_maps(path: Path = MANDATORY_QUESTIONS_PATH) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    raw = load_json(path)
    by_procedure: Dict[str, List[Dict]] = {}
    by_stem: Dict[str, List[Dict]] = {}

    for file_key, entries in raw.items():
        stem = Path(file_key).stem
        if not entries:
            continue

        questions = entries[0].get("questions", []) or []
        procedure_label = entries[0].get("procedure", "")

        by_stem[_norm(stem)] = questions
        if procedure_label:
            by_procedure[_norm(procedure_label)] = questions

    aliases = {
        "narkose": "anästhesie / narkose",
        "drk geburtshilfe infos": "klinik-logistik",
        "geburtshilfliche maßnahmen": "allgemeine einwilligung",
    }

    for alias, target in aliases.items():
        target_key = _norm(target)
        if target_key in by_procedure:
            by_procedure[_norm(alias)] = by_procedure[target_key]
        else:
            print(f"Warning: Alias target '{target}' not found in mandatory questions map")

    return by_procedure, by_stem


def discover_conversation_files(conversations_dir: Path = CONVERSATIONS_DIR) -> List[Path]:
    return sorted(
        [p for p in conversations_dir.rglob("*.json") if not p.name.startswith("conversation_index_")]
    )


def format_conversation_history(turns: List[Dict]) -> str:
    lines: List[str] = []
    for t in turns:
        turn_no = t.get("turn")
        patient = (t.get("patient_question") or "").strip()
        chatbot = (t.get("chatbot_response") or "").strip()
        lines.append(f"Turn {turn_no}")
        lines.append(f"Patient: {patient}")
        lines.append(f"Chatbot: {chatbot}")
        lines.append("")
    return "\n".join(lines).strip()


def _extract_json_array(text: str) -> str:
    match = JSON_ARRAY_PATTERN.search(text or "")
    return match.group(0) if match else (text or "[]")


def _build_batch_cache_key(conversation_history: str, question_specs: List[Dict], model_name: str) -> str:
    payload = {
        "model": model_name,
        "conversation": conversation_history,
        "questions": question_specs,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _build_question_specs(questions: List[Dict]) -> List[Dict]:
    specs: List[Dict] = []
    seen: Dict[str, int] = {}

    for idx, q in enumerate(questions, start=1):
        raw_qid = str(q.get("question_id") or "").strip()
        base_qid = raw_qid if raw_qid else f"missing_question_id_{idx}"

        seen[base_qid] = seen.get(base_qid, 0) + 1
        unique_qid = base_qid if seen[base_qid] == 1 else f"{base_qid}__{seen[base_qid]}"

        specs.append(
            {
                "question_id": unique_qid,
                "content": q.get("content", ""),
            }
        )

    return specs


def judge_questions_batch(conversation_history: str, question_specs: List[Dict], model_name: str) -> Dict[str, Dict]:
    if not question_specs:
        return {}

    expected_qids = [q["question_id"] for q in question_specs]
    expected_qids_set = set(expected_qids)

    system_message = (
        "You are an expert evaluator of clinical dialogue systems.\n"
        "Decide for each mandatory question whether the chatbot actively asked it.\n"
        "Count as asked only when the chatbot explicitly requests the information "
        "(direct question or clear request) and the meaning matches the target question "
        "(or a clear paraphrase).\n"
        "A paraphrase still must clearly request the same user-provided information.\n"
        "Do not count statements, explanations, indirect mentions, or rhetorical questions. "
        "If a user response is not clearly expected, mark not asked.\n"
        "Return JSON only: an array with exactly one object per question_id.\n"
        "Each object must contain exactly: question_id, is_asked (true/false), turn_number "
        "(integer or null).\n"
        "Every question_id must appear exactly once. No extra keys or text."
    )

    question_lines = []
    for q in question_specs:
        question_lines.append(f"- question_id: {q.get('question_id')} | question: {q.get('content')}")

    prompt = f"""
Task: Determine if each mandatory question was asked by the chatbot.

Output format:
- Return a JSON array with exactly {len(question_specs)} objects.
- Each object must contain exactly these keys:
  - question_id
  - is_asked (true/false)
  - turn_number (integer or null)
- Each question_id below must appear exactly once.
- Do not add extra keys or extra text.

Mandatory Questions:
{chr(10).join(question_lines)}

Conversation History:
{conversation_history}
""".strip()

    raw = make_api_call(
        prompt=prompt,
        model_name=model_name,
        temperature=0.0,
        system_message=system_message,
    )

    results: Dict[str, Dict] = {}
    qid_counts: Dict[str, int] = {}

    parse_error = None
    try:
        parsed = json.loads(_extract_json_array(raw))
    except json.JSONDecodeError:
        parsed = []
        parse_error = "invalid_json_output"

    if not isinstance(parsed, list):
        parsed = []
        parse_error = "invalid_json_shape"

    if len(parsed) != len(question_specs):
        print(f"Warning: LLM returned {len(parsed)} items, expected {len(question_specs)}")

    for item in parsed:
        if not isinstance(item, dict):
            continue

        qid = str(item.get("question_id") or "").strip()
        if not qid:
            continue
        if qid not in expected_qids_set:
            continue

        qid_counts[qid] = qid_counts.get(qid, 0) + 1
        if qid_counts[qid] > 1:
            if qid in results:
                results[qid]["judge_error"] = "duplicate_question_id_in_llm_output"
            continue

        turn_number_raw = item.get("turn_number")
        turn_number = turn_number_raw if isinstance(turn_number_raw, int) else None
        item_error = None
        if turn_number_raw is not None and not isinstance(turn_number_raw, int):
            item_error = "invalid_turn_number_type"

        results[qid] = {
            "is_asked": bool(item.get("is_asked", False)),
            "turn_number": turn_number,
            "judge_raw_output": raw,
            "judge_error": item_error,
        }

    for qid in expected_qids:
        if qid not in results:
            missing_error = parse_error if parse_error is not None else "missing_from_llm_output"
            results[qid] = {
                "is_asked": False,
                "turn_number": None,
                "judge_raw_output": raw,
                "judge_error": missing_error,
            }

    return results


def get_expected_questions(conversation_path: Path, metadata_procedure: str, by_proc: Dict[str, List[Dict]], by_stem: Dict[str, List[Dict]]) -> List[Dict]:
    proc_key = _norm(metadata_procedure)
    if proc_key in by_proc:
        return by_proc[proc_key]

    base = conversation_path.stem
    if "_active_" in base:
        stem = base.split("_active_")[0]
    elif "_passive_" in base:
        stem = base.split("_passive_")[0]
    else:
        stem = metadata_procedure

    return by_stem.get(_norm(stem), [])


def evaluate_conversation_file(path: Path, by_proc: Dict[str, List[Dict]], by_stem: Dict[str, List[Dict]], model_name: str, cache: Dict) -> Dict:
    data = load_json(path)
    metadata = data.get("metadata", {}) or {}
    turns = data.get("conversation", []) or []

    procedure = metadata.get("procedure", "")
    expected_questions = get_expected_questions(path, procedure, by_proc, by_stem)
    question_specs = _build_question_specs(expected_questions)
    history_text = format_conversation_history(turns)

    cache_key = _build_batch_cache_key(history_text, question_specs, model_name)

    batch_results: Dict[str, Dict]
    batch_error = None

    if cache_key in cache:
        batch_results = cache[cache_key]
    else:
        try:
            batch_results = judge_questions_batch(history_text, question_specs, model_name)
            cache[cache_key] = batch_results
            save_cache(cache)
        except Exception as e:
            batch_results = {}
            batch_error = str(e)

    question_rows: List[Dict] = []
    asked_turns: List[int] = []

    for spec in question_specs:
        qid = spec["question_id"]
        judge = batch_results.get(qid, {})

        is_asked = bool(judge.get("is_asked", False)) if batch_error is None else False
        turn_number = judge.get("turn_number") if batch_error is None else None

        if isinstance(turn_number, int):
            asked_turns.append(turn_number)

        question_rows.append(
            {
                "file": str(path),
                "procedure": procedure,
                "question_id": qid,
                "target_question": spec.get("content", ""),
                "is_asked": is_asked,
                "turn_number": turn_number,
                "judge_error": batch_error if batch_error is not None else judge.get("judge_error"),
                "judge_raw_output": judge.get("judge_raw_output", ""),
            }
        )

    total_q = len(expected_questions)
    asked_q = sum(1 for r in question_rows if r["is_asked"])
    recall = (asked_q / total_q) if total_q else 0.0

    return {
        "file": str(path),
        "procedure": procedure,
        "mode": metadata.get("mode"),
        "mandatory_questions_total": total_q,
        "mandatory_questions_asked": asked_q,
        "question_recall": round(recall, 4),
        "strict_compliance": 1 if total_q > 0 and recall == 1.0 else 0,
        "acceptable_compliance": 1 if recall >= COMPLIANCE_THRESHOLD else 0,
        "compliance_threshold": COMPLIANCE_THRESHOLD,
        "first_mandatory_question_turn": min(asked_turns) if asked_turns else None,
        "judge_failed": batch_error is not None,
        "question_judgments": question_rows,
    }


def build_global_summary(rows: List[Dict], judge_model: str) -> Dict:
    def aggregate(subset: List[Dict]) -> Dict:
        n = len(subset)
        if n == 0:
            return {
                "files_evaluated": 0,
                "mean_question_recall": 0.0,
                "strict_compliance_rate": 0.0,
                "acceptable_compliance_rate": 0.0,
                "mean_first_mandatory_question_turn": None,
                "judge_failure_rate": 0.0,
            }

        recalls = [r["question_recall"] for r in subset]
        strict_vals = [r["strict_compliance"] for r in subset]
        acceptable_vals = [r["acceptable_compliance"] for r in subset]
        judge_failures = [1 if r.get("judge_failed") else 0 for r in subset]
        first_turns = [
            r["first_mandatory_question_turn"]
            for r in subset
            if r["first_mandatory_question_turn"] is not None
        ]

        return {
            "files_evaluated": n,
            "mean_question_recall": round(sum(recalls) / n, 4),
            "strict_compliance_rate": round(sum(strict_vals) / n, 4),
            "acceptable_compliance_rate": round(sum(acceptable_vals) / n, 4),
            "acceptable_compliance_threshold": COMPLIANCE_THRESHOLD,
            "mean_first_mandatory_question_turn": round(sum(first_turns) / len(first_turns), 4)
            if first_turns
            else None,
            "judge_failure_rate": round(sum(judge_failures) / n, 4),
        }

    active_rows = [r for r in rows if r.get("mode") == "active"]
    passive_rows = [r for r in rows if r.get("mode") == "passive"]

    return {
        "judge_model": judge_model,
        "global_metrics": aggregate(rows),
        "active_mode_metrics": aggregate(active_rows),
        "passive_mode_metrics": aggregate(passive_rows),
    }


def run_evaluation(
    conversations_dir: Path = CONVERSATIONS_DIR,
    mandatory_questions_path: Path = MANDATORY_QUESTIONS_PATH,
    results_dir: Path = RESULTS_DIR,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_files: int = None,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    by_proc, by_stem = load_mandatory_question_maps(mandatory_questions_path)
    files = discover_conversation_files(conversations_dir)
    if max_files is not None:
        files = files[:max_files]

    cache = load_cache()

    per_conversation: List[Dict] = []
    per_question: List[Dict] = []

    for i, fp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Evaluating {fp.name}")
        row = evaluate_conversation_file(fp, by_proc, by_stem, judge_model, cache)
        per_conversation.append({k: v for k, v in row.items() if k != "question_judgments"})
        per_question.extend(row["question_judgments"])

    summary = build_global_summary(per_conversation, judge_model)

    save_json(results_dir / "mandatory_q_per_conversation.json", per_conversation)
    save_json(results_dir / "mandatory_q_judgments.json", per_question)
    save_json(results_dir / "mandatory_q_summary.json", summary)

    return summary, per_conversation, per_question


if __name__ == "__main__":
    summary, conv_rows, q_rows = run_evaluation()

    print("\nMandatory Questions Evaluation Summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved {len(conv_rows)} conversation rows and {len(q_rows)} question judgments")
