import re
import json
import requests
from typing import Dict, List

from utils.llm_config import make_api_call

def extract_statements_cot(text: str, model_name: str = None) -> List[Dict]:
    """
    Chain-of-Thought prompting.
    Forces the model to generate a 'Thinking Process' block BEFORE the JSON output.
    """

    system_message = (
        "You are a medical assistant. Your task is to extract statements that should be communicated with the patient. "
        "These statements represent information for patient consultation.\n\n"
        "STEP 1: THINKING PROCESS\n"
        "First, analyze the document section by section. Identify complex instructions, "
        "contraindications, and risks. Write down your reasoning for what is critical to mention.\n\n"
        "STEP 2: JSON GENERATION\n"
        "Output the list as a JSON array where each item has:\n"
        "- 'category': Select one: 'RISK', 'PROCEDURE', 'INSTRUCTION', 'GENERAL'\n"
        "- 'statement': The fact content\n"
        "- 'importance': Assign 'Critical', 'High', 'Medium', or 'Low'.\n"
        "  (Note: 'Critical' = Essential safety warnings or absolute contraindications)\n"
        "LANGUAGE: Output all statements and rationales in German, maintaining the original medical terminology from the source document.\n\n"
        "Output format:\n"
        "Thinking: <your analysis here>\n"
        "JSON: <the json array>"
    )
    
    user_message = f"DOCUMENT:\n{text}"

    try:
        content = make_api_call(user_message, model_name, temperature=0.3, timeout=600,
                               system_message=system_message)
    except Exception as exc:
        raise RuntimeError(f"API call failed: {exc}")

    # Parse the thinking and JSON parts
    m = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", content)
    
    # Extract thinking process
    thinking_part = content.split("JSON:")[0].replace("Thinking:", "").strip() if "JSON:" in content else ""
    
    if not m:
        raise RuntimeError("CoT extraction failed: No JSON block found after reasoning.")

    try:
        data = json.loads(m.group(1))
    except Exception:
        raise RuntimeError("JSON decoding failed.")

    return {
        "extracted_data": data,
        "metadata": {
            "thinking_process": thinking_part,
            "system_message": system_message,
            "temperature": 0.3,
            "has_reasoning": True
        }
    }

if __name__ == "__main__":
    with open("data/raw_md_files/DRK Geburtshilfe Infos.md", "r") as f:
        sample = f.read()
    results = extract_statements_cot(sample)
    
    if "metadata" in results and "thinking_process" in results["metadata"]:
        print("=== THINKING PROCESS ===")
        print(results["metadata"]["thinking_process"][:500])  # First 500 chars
        print("\n=== EXTRACTED DATA ===")
    
    print(json.dumps(results["extracted_data"], indent=2))