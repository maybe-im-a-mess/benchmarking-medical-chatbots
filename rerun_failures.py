#!/usr/bin/env python3
"""
Re-run only the failed extractions from the full pipeline run.
- GPT-5 CoT: all 6 documents
- Qwen atomic: Kaiserschnitt and Narkose (with increased timeout)
"""

import json
import time
from pathlib import Path

from information_extraction.naive_llm import extract_statements_naive
from information_extraction.atomic_fact_extraction import extract_statements_atomic
from information_extraction.cot_extraction import extract_statements_cot
from information_extraction.uie import extract_statements_schema
from utils.llm_config import MODELS

# Increase timeout for atomic method
import information_extraction.atomic_fact_extraction as atomic_module

# Patch the get_completion function with higher timeout for qwen atomic
original_get_completion = atomic_module.get_completion

def get_completion_long_timeout(prompt: str, model_name: str = None, system_message: str = None) -> str:
    """Helper function with extended timeout for atomic extractions."""
    from utils.llm_config import make_api_call
    return make_api_call(prompt, model_name, temperature=0.3, timeout=900, system_message=system_message)

# Apply the patch
atomic_module.get_completion = get_completion_long_timeout

def load_document(doc_name: str) -> str:
    """Load a document from the raw_md_files directory."""
    doc_path = Path("data/raw_md_files") / f"{doc_name}.md"
    with open(doc_path, "r", encoding="utf-8") as f:
        return f.read()

def save_result(output_dir: Path, document: str, method: str, result: dict, execution_time: float, model_key: str, model_name: str):
    """Save extraction result to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle both dict format (with extracted_data) and list format
    if isinstance(result, dict) and "extracted_data" in result:
        extracted_data = result["extracted_data"]
        metadata = result.get("metadata", {})
    elif isinstance(result, list):
        extracted_data = result
        metadata = {}
    else:
        extracted_data = result
        metadata = {}
    
    output = {
        "document": document,
        "method": method,
        "model": {
            "key": model_key,
            "name": model_name
        },
        "execution_time_seconds": round(execution_time, 1),
        "item_count": len(extracted_data) if isinstance(extracted_data, list) else 1,
        "extracted_data": extracted_data,
        "metadata": metadata
    }
    
    output_file = output_dir / f"{document}_{method}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"  -> Saved: {output_file.name} ({len(extracted_data)} items, {execution_time:.1f}s)")

def run_extraction(document: str, method: str, model_key: str):
    """Run a single extraction task."""
    model_config = MODELS[model_key]
    model_name = model_config["model_id"]
    
    print(f"Running: {document:30s} | {method:10s} | {model_key}")
    
    # Load document
    doc_text = load_document(document)
    
    # Select extraction method
    extract_methods = {
        "naive": extract_statements_naive,
        "atomic": extract_statements_atomic,
        "cot": extract_statements_cot,
        "uie": extract_statements_schema
    }
    extract_fn = extract_methods[method]
    
    # Run extraction
    start_time = time.time()
    try:
        result = extract_fn(doc_text, model_name=model_name)
        execution_time = time.time() - start_time
        
        # Save result
        output_dir = Path("data/processed") / model_key
        save_result(output_dir, document, method, result, execution_time, model_key, model_config["display_name"])
        
        item_count = len(result["extracted_data"]) if isinstance(result, dict) and "extracted_data" in result else len(result)
        print(f"✓ {document:30s} | {method:10s} | {item_count:3d} items | {execution_time:6.1f}s\n")
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)[:100]
        print(f"✗ {document:30s} | {method:10s} | FAILED: {error_msg}\n")
        
        # Save error
        output_dir = Path("data/processed") / model_key
        save_result(output_dir, document, method, 
                   {"error": str(e), "status": "failed"}, 
                   execution_time, model_key, model_config["display_name"])

def main():
    print("=" * 70)
    print("RE-RUNNING FAILED EXTRACTIONS")
    print("=" * 70)
    
    # Failed GPT-5 CoT extractions (all 6 documents)
    gpt5_cot_docs = [
        "DRK Geburtshilfe Infos",
        "Geburtseinleitung",
        "Geburtshilfliche Maßnahmen",
        "Kaiserschnitt",
        "Narkose",
        "Äußere Wendung"
    ]
    
    # Failed Qwen atomic extractions (2 documents with increased timeout)
    qwen_atomic_docs = [
        "Kaiserschnitt",
        "Narkose"
    ]
    
    print("\n" + "=" * 70)
    print("GPT-5 CoT Extractions (6 documents)")
    print("=" * 70 + "\n")
    
    for doc in gpt5_cot_docs:
        run_extraction(doc, "cot", "gpt-5-mini")
    
    print("\n" + "=" * 70)
    print("Qwen Atomic Extractions (2 documents, timeout=900s)")
    print("=" * 70 + "\n")
    
    for doc in qwen_atomic_docs:
        run_extraction(doc, "atomic", "qwen3-4b")
    
    print("\n" + "=" * 70)
    print("RE-RUN COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
