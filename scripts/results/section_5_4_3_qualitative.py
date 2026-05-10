#!/usr/bin/env python3
"""
Section 5.4.3 — Qualitative Analysis: Representative Conversation Excerpts

Identifies and extracts:
  • Excerpt 1: A naive-mode conversation where a mandatory question was missed
  • Excerpt 2: A supervised-mode conversation with full compliance
Saves both as formatted markdown.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.results._common import (
    MANDATORY_A_PER_CONV, MANDATORY_B_PER_CONV,
    CONV_A_DIR, CONV_B_DIR, TABLES_DIR,
    load_json, ensure_dir, display_mode,
)


EXCLUDE_PROCEDURES = {
    "DRK Geburtshilfe Infos",
}


def find_example_conversations(per_conv, conv_dir, exclude_procedures=None):
    """Find one naive-fail and one supervised-success example."""
    naive_fail = None
    supervised_success = None
    exclude = {p.strip().lower() for p in (exclude_procedures or [])}
    procedure_cache = {}

    def is_excluded(candidate: Path) -> bool:
        if not exclude:
            return False
        key = str(candidate)
        if key not in procedure_cache:
            conv = load_json(candidate)
            procedure_cache[key] = (conv.get("metadata", {}).get("procedure") or "").strip()
        return procedure_cache[key].lower() in exclude

    for entry in per_conv:
        filepath = entry.get("file", "")
        filename = Path(filepath).name
        mode = entry.get("mode", "")
        recall = entry.get("question_recall", 1.0)
        strict = entry.get("strict_compliance", 0)

        # Naive mode: missed at least one question
        if mode == "passive" and recall < 1.0 and naive_fail is None:
            candidate = conv_dir / filename
            if candidate.exists() and not is_excluded(candidate):
                naive_fail = {"entry": entry, "path": candidate}

        # Supervised mode: strict compliance
        if mode == "active" and strict == 1 and supervised_success is None:
            candidate = conv_dir / filename
            if candidate.exists() and not is_excluded(candidate):
                supervised_success = {"entry": entry, "path": candidate}

        if naive_fail and supervised_success:
            break

    return naive_fail, supervised_success


def format_conversation_excerpt(conv_data, entry, max_turns=None):
    """Format a conversation as a readable markdown excerpt."""
    lines = []
    meta = conv_data.get("metadata", {})
    lines.append(f"**Procedure:** {meta.get('procedure', 'N/A')}")
    lines.append(f"**Mode:** {display_mode(meta.get('mode', ''))}")
    lines.append(f"**Total turns:** {meta.get('total_turns', len(conv_data.get('conversation', [])))}")
    lines.append(f"**Questions asked:** {entry.get('mandatory_questions_asked', '?')}"
                 f"/{entry.get('mandatory_questions_total', '?')}")
    lines.append(f"**Question recall:** {entry.get('question_recall', 0):.2f}")

    persona = meta.get("patient_persona", {})
    if persona:
        lines.append(f"**Patient persona:** {persona.get('name', 'N/A')}, "
                     f"age {persona.get('age', '?')}, "
                     f"anxiety: {persona.get('anxiety_level', '?')}, "
                     f"education: {persona.get('education_level', '?')}")

    lines.append("")
    lines.append("---")
    lines.append("")

    turns = conv_data.get("conversation", [])
    display_turns = turns[:max_turns] if max_turns else turns

    for turn in display_turns:
        turn_no = turn.get("turn", "?")
        patient = (turn.get("patient_question") or "").strip()
        chatbot = (turn.get("chatbot_response") or "").strip()

        lines.append(f"**Turn {turn_no}**")
        if patient:
            lines.append(f"> 🗣️ **Patient:** {patient[:500]}{'...' if len(patient)>500 else ''}")
        if chatbot:
            lines.append(f"> 🤖 **Chatbot:** {chatbot[:500]}{'...' if len(chatbot)>500 else ''}")
        lines.append("")

    if max_turns and len(turns) > max_turns:
        lines.append(f"*... {len(turns) - max_turns} more turns omitted ...*")

    return "\n".join(lines)


def main():
    out_dir = TABLES_DIR / "section_5_4"
    ensure_dir(out_dir)

    # Try Dataset B first (more conversations, more likely to find good examples)
    per_conv_b = load_json(MANDATORY_B_PER_CONV)
    naive_fail, supervised_ok = find_example_conversations(
        per_conv_b,
        CONV_B_DIR,
        exclude_procedures=EXCLUDE_PROCEDURES,
    )

    # Fall back to Dataset A if needed
    if not naive_fail or not supervised_ok:
        per_conv_a = load_json(MANDATORY_A_PER_CONV)
        nf_a, so_a = find_example_conversations(
            per_conv_a,
            CONV_A_DIR,
            exclude_procedures=EXCLUDE_PROCEDURES,
        )
        naive_fail = naive_fail or nf_a
        supervised_ok = supervised_ok or so_a

    md_lines = ["# Qualitative Conversation Excerpts\n"]

    if naive_fail:
        print(f"  📝 Naive failure example: {naive_fail['path'].name}")
        conv = load_json(naive_fail["path"])
        md_lines.append("## Excerpt 1: Naive Mode — Missed Mandatory Question\n")
        md_lines.append(format_conversation_excerpt(conv, naive_fail["entry"], max_turns=8))
        md_lines.append("\n")
    else:
        print("  ⚠️ Could not find a naive-mode failure example")
        md_lines.append("## Excerpt 1: Not available\n")

    if supervised_ok:
        print(f"  📝 Supervised success example: {supervised_ok['path'].name}")
        conv = load_json(supervised_ok["path"])
        md_lines.append("## Excerpt 2: Supervised Mode — Full Compliance\n")
        md_lines.append(format_conversation_excerpt(conv, supervised_ok["entry"], max_turns=8))
    else:
        print("  ⚠️ Could not find a supervised success example")
        md_lines.append("## Excerpt 2: Not available\n")

    out_path = out_dir / "qualitative_excerpts.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  💾 Saved: {out_path}")

    print("\n✅ Section 5.4.3 complete.\n")


if __name__ == "__main__":
    main()
