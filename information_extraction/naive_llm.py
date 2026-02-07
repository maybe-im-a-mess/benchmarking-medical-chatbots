import os
import re
import json
import requests
from typing import Any, Dict, List, Optional

from utils.llm_config import make_api_call


def extract_statements_naive(
	text: str,
	model_name: str = None,
) -> List[Dict]:
	"""
	Naive approach for medical statement extraction.
	"""

	system_message = (
		"You are a medical information extraction specialist. Extract the key facts from the document.\n"
		"Return a JSON array of objects. Each object must have exactly these fields:\n"
		"- 'category': Classify strictly into one of: ['RISK', 'PROCEDURE', 'INSTRUCTION', 'GENERAL']\n"
		"- 'statement': The specific medical fact\n"
		"- 'importance': One of ['Critical', 'High', 'Medium', 'Low']\n"
		"  (Use 'Critical' for life-threatening risks or mandatory legal warnings)\n\n"
		"Example:\n"
		'[{"category": "Risks", "statement": "Infection is possible.", "importance": "Medium"}]\n'
		"LANGUAGE: German."
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
	try:
		results = extract_statements_naive(sample)
	except RuntimeError as e:
		print(f"Extraction failed: {e}")
	else:
		for i, t in enumerate(results["extracted_data"], 1):
			print(f"{i}. [{t['importance']}] {t['category']}: {t['statement']}")
