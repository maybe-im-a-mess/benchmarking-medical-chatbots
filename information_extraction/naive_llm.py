import os
import re
import json
import requests
from typing import Any, Dict, List, Optional

from utils.llm_config import LOCAL_MODEL_URL, MODEL_NAME, make_api_call


def extract_statements_naive(
	text: str,
	model_name: str = None,
) -> List[Dict]:
	"""
	Naive approach for medical statement extraction.
	"""

	system_message = (
		"You are a medical information extraction specialist. Extract key statements "
		"that a healthcare provider should communicate with the patient based on medical documents. "
		"These statements represent information that needs to be discussed during patient consultation.\n\n"
		"CRITICAL: Return ONLY a valid JSON array. Each object MUST have exactly these two fields:\n"
		"- 'statement': The specific statement to communicate with the patient\n"
		"- 'rationale': Brief explanation of why this should be discussed\n"
		"\n"
		"DO NOT use alternative field names like 'point', 'topic', 'title', 'summary', or 'explanation'.\n"
		"DO NOT add extra fields or wrap the JSON in markdown code blocks.\n"
		"\n"
		"Example format:\n"
		'[{"statement": "...", "rationale": "..."}]\n'
		"\n"
		"LANGUAGE: Output all statements and rationales in German, maintaining the original medical terminology from the source document.\n"
		"\n"
		"Include all relevant statements without arbitrary limits. "
		"Focus on clinical relevance, patient safety, and informed consent. "
		"Each statement should be concrete and directly supported by the document."
	)
	
	user_message = f"Document:\n{text}"

	try:
		content = make_api_call(user_message, model_name, temperature=0.3, timeout=600, 
		                        system_message=system_message)
	except Exception as exc:
		raise RuntimeError(f"API call failed: {exc}")

	# Parse JSON in response
	m = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", content)
	if not m:
		raise RuntimeError(
			"Could not parse JSON list from local model response. Response preview: "
			+ (content[:1000] + "..." if len(content) > 1000 else content)
		)

	try:
		data = json.loads(m.group(1))
	except Exception as exc:
		raise RuntimeError(f"Failed to decode JSON from local model response: {exc}")

	return {
		"extracted_data": data,
		"metadata": {
			"system_message": system_message,
			"temperature": 0.3
		}
	}


if __name__ == "__main__":
	sample = open("data/raw_md_files/DRK Geburtshilfe Infos.md").read()
	print(sample)
	try:
		results = extract_statements_naive(sample)
	except RuntimeError as e:
		print(f"Extraction failed: {e}")
	else:
		for i, t in enumerate(results["extracted_data"], 1):
			print(f"{i}. {t['statement']}")
			print(f"   Why: {t['rationale']}\n")
