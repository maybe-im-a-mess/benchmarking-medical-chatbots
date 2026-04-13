from chatbot.embeddings import load_document_store
from chatbot.doctor_agent import DoctorAgent
from chatbot.patient_agent import create_patient
from chatbot.dialogue_manager import DialogueManager
from typing import List, Dict
import time
import os
import json


def _conversation_filename(procedure_name: str, mode: str, persona_type: str, repeat_idx: int) -> str:
    return f"{procedure_name}_{mode}_{persona_type}_{repeat_idx:03d}.json"


def _load_existing_index(index_path: str) -> List[Dict]:
    if not os.path.exists(index_path):
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_index(index_path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _dedupe_index_rows(rows: List[Dict]) -> List[Dict]:
    deduped = {}
    for row in rows:
        filepath = row.get("filepath")
        if filepath:
            deduped[filepath] = row
    return list(deduped.values())


def _upsert_index_row(rows: List[Dict], row: Dict) -> None:
    filepath = row.get("filepath")
    if not filepath:
        rows.append(row)
        return
    for i, existing in enumerate(rows):
        if existing.get("filepath") == filepath:
            rows[i] = row
            return
    rows.append(row)


def generate_single_conversation(
                                 document_store,
                                 procedure_name: str,
                                 persona_type: str,
                                 doc_model: str,
                                 pat_model: str,
                                 max_turns: int,
                                 pat_temperature: float = 0.8,
                                 min_turns: int = 8,
                                 mandatory_questions_path: str = "data/mandatory_questions.json",
                                 mode: str = "active",
                                 output_dir: str = "data/conversations",
                                 repeat_idx: int = 1,
                                 dataset_tag: str = "default",
                                 ):
    chatbot = DoctorAgent(document_store, doc_model)
    patient = create_patient(procedure_name=procedure_name,
                                 persona_type=persona_type,
                                 model=pat_model,
                                 max_questions=max_turns,
                                 temperature=pat_temperature)
        
    # Run conversation
    manager = DialogueManager(
        chatbot_agent=chatbot,
        patient_agent=patient,
        max_turns=max_turns,
        min_turns=min_turns,
        procedure_name=procedure_name,
        mandatory_questions_path=mandatory_questions_path,
        mode=mode
    )
    
    conversation_log = manager.run_conversation()

    filename = _conversation_filename(procedure_name, mode, persona_type, repeat_idx)
    filepath = os.path.join(output_dir, filename)

    # Save conversation
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    manager.save_conversation(
        filepath,
        extra_metadata={
            "dataset_tag": dataset_tag,
            "repeat_index": repeat_idx,
            "patient_temperature_effective": pat_temperature,
            "generator_version": "patient_agent_probabilistic_v2",
        },
    )

    return filepath, conversation_log


def generate_conversation_dataset(document_store,
                                  persona_types: List[str],
                                  procedures: List[str],
                                  dataset_tag: str,
                                  doc_model: str = "gpt-5.4-mini",
                                  pat_model: str = "gpt-5.4-mini",
                                  max_turns: int = 8,
                                  pat_temperature: float = 0.8,
                                  min_turns: int = 8,
                                  mandatory_questions_path: str = "data/mandatory_questions.json",
                                  mode: str = "active",
                                  repeats_per_scenario: int = 1,
                                  output_root: str = "data/conversations",
                                  ):
    """
    Generate a dataset of conversations with different patients.
    
    Parameters:
        document_store: Loaded document store for retrieval
        persona_types: List of patient persona types to simulate
        procedures: List of medical procedures to discuss
        doc_model: Chatbot LLM model
        pat_model: Patient LLM model
        max_turns: Maximum turns per conversation
    """
    print(f"\n{'='*60}")
    print("CONVERSATION DATASET GENERATOR")
    print(f"{'='*60}\n")
    
    output_dir = os.path.join(output_root, dataset_tag)
    index_path = os.path.join(output_dir, "conversation_index_all.json")

    # Load and reuse existing index for resume-safe generation.
    all_conversations = _load_existing_index(index_path)
    all_conversations = _dedupe_index_rows(all_conversations)
    indexed_files = {row.get("filepath") for row in all_conversations if row.get("filepath")}

    if all_conversations:
        _save_index(index_path, all_conversations)

    if indexed_files:
        print(f"Resuming dataset '{dataset_tag}': {len(indexed_files)} conversations already indexed.")

    total_planned = len(procedures) * len(persona_types) * max(1, repeats_per_scenario)
    print(f"Planned conversations for mode '{mode}': {total_planned}")

    # Generate conversations
    for procedure in procedures:
        for persona_type in persona_types:
            for repeat_idx in range(1, max(1, repeats_per_scenario) + 1):
                filename = _conversation_filename(procedure, mode, persona_type, repeat_idx)
                filepath = os.path.join(output_dir, filename)

                if os.path.exists(filepath):
                    if filepath not in indexed_files:
                        _upsert_index_row(all_conversations, {
                            "procedure": procedure,
                            "persona": persona_type,
                            "repeat_index": repeat_idx,
                            "mode": mode,
                            "filepath": filepath,
                            "turns": None,
                            "patient_temperature": pat_temperature,
                            "dataset_tag": dataset_tag,
                            "status": "existing",
                        })
                        indexed_files.add(filepath)
                        _save_index(index_path, all_conversations)
                    print(f"Skipping existing: {filename}")
                    continue

                try:
                    filepath, conv_log = generate_single_conversation(
                        document_store=document_store,
                        procedure_name=procedure,
                        persona_type=persona_type,
                        doc_model=doc_model,
                        pat_model=pat_model,
                        max_turns=max_turns,
                        pat_temperature=pat_temperature,
                        min_turns=min_turns,
                        mandatory_questions_path=mandatory_questions_path,
                        mode=mode,
                        output_dir=output_dir,
                        repeat_idx=repeat_idx,
                        dataset_tag=dataset_tag,
                    )

                    _upsert_index_row(all_conversations, {
                        "procedure": procedure,
                        "persona": persona_type,
                        "repeat_index": repeat_idx,
                        "mode": mode,
                        "filepath": filepath,
                        "turns": conv_log["metadata"].get("total_turns"),
                        "patient_temperature": pat_temperature,
                        "dataset_tag": dataset_tag,
                        "status": "new",
                    })
                    indexed_files.add(filepath)

                    # Checkpoint after each successful conversation.
                    _save_index(index_path, all_conversations)

                    print("\nWaiting 10 seconds before next conversation...")
                    time.sleep(10)

                except Exception as e:
                    # Continue safely without losing already saved conversations.
                    print(f"\nError generating {filename}: {e}")
                    _save_index(index_path, all_conversations)
                    if "HF_EMBEDDING_CREDITS_DEPLETED" in str(e):
                        raise RuntimeError(
                            "Stopping generation: Hugging Face embedding credits are depleted. "
                            "Please top up HF credits or switch retriever embedding backend before continuing."
                        ) from e
                    continue

    # Print summary
    print(f"\n{'='*60}")
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"\nTotal conversations indexed: {len(all_conversations)}")
    print(f"Saved in: {output_dir}")
    
    return all_conversations
    
    
if __name__ == "__main__":
    print("Loading document store...")
    document_store = load_document_store()
    
    # Define all procedures and personas
    procedures = [
        "Narkose",
        "Kaiserschnitt", 
        "Geburtseinleitung",
        "Geburtshilfliche Maßnahmen",
        "Äußere Wendung",
        "DRK Geburtshilfe Infos" 
    ]
    
    persona_types = [
        "baseline",               # Control
        "induction_risk",         # Tests C-section history trap
        "anesthesia_risk",        # Tests fasting/teeth trap
        "version_contraindication", # Tests bleeding trap
        "allergy_risk",           # Tests allergy trap
        "anticoagulation_risk",   # Tests blood thinner disclosure
        "trauma_history_risk",    # Tests anxiety/previous bad experience
        "hypertension_risk",      # Tests blood pressure risk disclosure
        "language_barrier_risk"   # Tests comprehension/clarification needs
    ]
    
    dataset_tag = os.getenv("DATASET_TAG", "patient_agent_probabilistic_v3")
    master_index = []

    # Generate full dataset
    for mode in ["passive", "active"]:
        print(f"\n{'='*70}")
        print(f"GENERATING DATASET WITH MODE: {mode.upper()}")
        print(f"{'='*70}\n")
        
        mode_conversations = generate_conversation_dataset(
            document_store=document_store,
            procedures=procedures,
            persona_types=persona_types,
            dataset_tag=dataset_tag,
            max_turns=12,
            pat_temperature=0.8,
            min_turns=8,
            mandatory_questions_path="data/mandatory_questions.json",
            mode=mode,
            repeats_per_scenario=2,
        )

        master_index.extend(mode_conversations)

    # Save run-level master index in the dataset folder.
    dataset_dir = os.path.join("data", "conversations", dataset_tag)
    run_index_path = os.path.join(dataset_dir, "conversation_index_run.json")
    _save_index(run_index_path, master_index)

    print(f"\n✓ Dataset tag: {dataset_tag}")
    print(f"✓ Run index: {run_index_path}")
    print(f"✓ Indexed entries: {len(master_index)}")