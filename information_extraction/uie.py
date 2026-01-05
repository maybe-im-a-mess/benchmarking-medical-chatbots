import re
import json
import requests
from typing import Dict, List

from utils.llm_config import make_api_call

def extract_statements_schema(text: str, model_name: str = None) -> List[Dict]:
    """
    Extracts statements using a strict Schema-Guided approach.
    Forces the model to classify statements into specific categories (Risk, Instruction, etc.)
    """

    schema_definition = json.dumps({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["RISK", "INSTRUCTION", "PREREQUISITE", "GENERAL_INFO"]},
                "topic": {"type": "string", "description": "The medical topic, e.g., 'Anesthesia'"},
                "statement": {"type": "string", "description": "The specific statement to communicate with the patient"},
                "rationale": {"type": "string", "description": "Brief explanation of why this should be discussed"},
                "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}
            },
            "required": ["category", "topic", "statement", "rationale", "priority"]
        }
    }, indent=2)

    system_message = (
        "You are a strict data extraction system. Your task is to extract medical statements "
        "that should be communicated with the patient. These statements represent information for patient consultation. "
        "Structure them exactly according to the provided JSON Schema.\n\n"
        f"SCHEMA DEFINITION:\n{schema_definition}\n\n"
        "RULES:\n"
        "1. Classify each statement into one of the allowed categories: RISK, INSTRUCTION, PREREQUISITE, GENERAL_INFO.\n"
        "2. Assign a PRIORITY based on patient safety importance.\n"
        "3. Provide a rationale explaining why this statement should be discussed with the patient.\n"
        "4. Do not include conversational filler. Output ONLY the JSON array.\n"
        "\n"
        "LANGUAGE: Output all statements and rationales in German, maintaining the original medical terminology from the source document."
    )
    
    user_message = f"DOCUMENT CONTENT:\n{text}"

    try:
        content = make_api_call(user_message, model_name, temperature=0.3, timeout=600,
                               system_message=system_message)
    except Exception as exc:
        raise RuntimeError(f"API call failed: {exc}")

    # Parse json in response
    m = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", content)
    if not m:
        raise RuntimeError("Model failed to adhere to JSON Schema format.")

    try:
        data = json.loads(m.group(1))
    except Exception:
        raise RuntimeError("JSON parsing failed.")

    # Flatten to consistent output format: statement, rationale
    out: List[Dict] = []
    for item in data:
        out.append({
            "statement": item.get('statement', ''),
            "rationale": item.get('rationale', '')
        })

    return {
        "extracted_data": out,
        "metadata": {
            "system_message": system_message,
            "temperature": 0.3,
            "schema_guided": True,
            "raw_structured_data": data
        }
    }

if __name__ == "__main__":
    # Test run
    with open("data/raw_md_files/DRK Geburtshilfe Infos.md", "r") as f:
        sample = f.read()
    results = extract_statements_schema(sample)
    print(json.dumps(results["extracted_data"], indent=2))