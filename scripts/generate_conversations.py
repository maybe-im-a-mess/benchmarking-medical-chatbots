from chatbot.embeddings import load_document_store
from chatbot.doctor_agent import DoctorAgent
from chatbot.patient_agent import PatientAgent, PatientPersona, create_patient
from chatbot.dialogue_manager import DialogueManager
from typing import Optional, List
from datetime import datetime
import time
import os



def generate_single_conversation(document_store, 
                                 procedure_name: str,
                                 persona_type: str,
                                 doc_model: str,
                                 pat_model: str,
                                 max_turns: int,
                                 conversation_id: int = 1,
                                 min_turns: int = 4,
                                 mandatory_questions_path: str = "data/mandatory_questions.json",
                                 mode: str = "active"
                                 ):
    chatbot = DoctorAgent(document_store, doc_model)
    patient = create_patient(procedure_name=procedure_name,
                                 persona_type=persona_type,
                                 model=pat_model,
                                 max_questions=max_turns)
        
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
    
    # Generate filename
    filename = f"{procedure_name}_{mode}_{persona_type}_{conversation_id:03d}.json"
    filepath = os.path.join("data", "conversations", filename)
    
    # Save conversation
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    manager.save_conversation(filepath)
    
    return filepath, conversation_log


def generate_conversation_dataset(document_store,
                                  persona_types: List[str],
                                  procedures: List[str],
                                  doc_model: str = "gpt-5-mini",
                                  pat_model: str = "gpt-5-mini",
                                  max_turns: int = 8,
                                  min_turns: int = 4,
                                  mandatory_questions_path: str = "data/mandatory_questions.json",
                                  mode: str = "active",
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
    
    # Track generated conversations
    all_conversations = []
    conversation_id = 1

    # Generate conversations
    for procedure in procedures:
        for persona_type in persona_types:
            try:
                filepath, conv_log = generate_single_conversation(
                    document_store=document_store,
                    procedure_name=procedure,
                    persona_type=persona_type,
                    doc_model=doc_model,
                    pat_model=pat_model,
                    max_turns=max_turns,
                    conversation_id=conversation_id,
                    min_turns=min_turns,
                    mandatory_questions_path=mandatory_questions_path,
                    mode=mode
                )
                
                all_conversations.append({
                    "id": conversation_id,
                    "procedure": procedure,
                    "persona": persona_type,
                    "filepath": filepath,
                    "turns": conv_log["metadata"]["total_turns"]
                })
                
                conversation_id += 1
                
                # Delay between conversations to avoid rate limits
                print("\nWaiting 10 seconds before next conversation...")
                time.sleep(10)
                
            except Exception as e:
                print(f"\nError generating conversation {conversation_id}: {e}")
                continue

    # Print summary
    print(f"\n{'='*60}")
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"\nTotal conversations generated: {len(all_conversations)}")
    print(f"Saved in: data/conversations/")
    
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
        "allergy_risk"            # Tests allergy trap
    ]
    
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
            max_turns=8,
            min_turns=4,
            mandatory_questions_path="data/mandatory_questions.json",
            mode=mode
        )
        
        for conv in mode_conversations:
            conv["mode"] = mode
            master_index.append(conv)
    
    # Save master index
    import json
    with open("data/conversations/conversation_index_all.json", 'w') as f:
        json.dump(master_index, f, indent=2)
    
    print(f"\n✓ Master index: {len(master_index)} conversations")