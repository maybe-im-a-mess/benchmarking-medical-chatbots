import re
import json
import requests
from typing import Dict, List

from utils.llm_config import make_api_call

def get_completion(prompt: str, model_name: str = None, system_message: str = None) -> str:
    """Helper function for multiple LLM calls."""
    return make_api_call(prompt, model_name, temperature=0.3, timeout=600, system_message=system_message)

def extract_statements_atomic(text: str, model_name: str = None) -> List[Dict]:
    """
    Two steps extraction:
    1. Atomic Fact Extraction (break text into independent atomic statements).
    2. Synthesis (group atomic statements into patient discussion points).
    """
    
    system_msg_step1 = (
        "Extract key medical facts from text. Each fact should be a single, independent statement. "
        "Focus on: risks, contraindications, procedures, patient requirements, and informed consent items. "
        "List as bullet points (max 30 facts)."
    )
    
    atomic_raw = get_completion(f"TEXT:\n{text}", model_name, system_message=system_msg_step1)
    
    system_msg_step2 = (
        "Group medical facts into statements that should be communicated with the patient. "
        "These statements represent information for patient consultation. "
        "Filter out irrelevant details. "
        "Return JSON array with format: [{\"statement\": \"...\", \"rationale\": \"...\"}]. "
        "Keep it concise. "
        "LANGUAGE: Output all statements and rationales in German, maintaining the original medical terminology."
    )
    
    final_content = get_completion(f"FACTS:\n{atomic_raw}", model_name, system_message=system_msg_step2)

    # Parse json in case the model returns additional text
    m = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", final_content)
    if not m:
        raise RuntimeError("Failed to parse JSON from Synthesis stage.")
        
    try:
        data = json.loads(m.group(1))
    except Exception:
        raise RuntimeError("JSON decoding error in Synthesis stage.")

    return {
        "extracted_data": data,
        "metadata": {
            "system_message_stage1": system_msg_step1,
            "system_message_stage2": system_msg_step2,
            "temperature": 0.3,
            "two_stage": True
        }
    }

if __name__ == "__main__":
    with open("data/raw_md_files/DRK Geburtshilfe Infos.md", "r") as f:
        sample = f.read()
    results = extract_statements_atomic(sample)
    print(json.dumps(results["extracted_data"], indent=2))