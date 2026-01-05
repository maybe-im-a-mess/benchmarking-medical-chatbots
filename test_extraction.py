"""
Quick test script to validate extraction pipeline on a single document.
Useful for debugging without running the full extraction on all documents.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from information_extraction.naive_llm import extract_statements_naive
from information_extraction.uie import extract_statements_schema
from information_extraction.atomic_fact_extraction import extract_statements_atomic
from information_extraction.cot_extraction import extract_statements_cot
from utils.llm_config import MODELS


def test_single_extraction(doc_path: str, method: str = "naive", model_key: str = "qwen3-4b"):
    """Test a single extraction method on one document.
    
    Args:
        doc_path: Path to the markdown document
        method: Extraction method (naive, atomic, cot, uie)
        model_key: Model key from MODELS config
    """
    
    # Validate inputs
    methods = {
        "naive": extract_statements_naive,
        "atomic": extract_statements_atomic,
        "cot": extract_statements_cot,
        "uie": extract_statements_schema
    }
    
    if method not in methods:
        print(f"Error: Unknown method '{method}'. Choose from: {list(methods.keys())}")
        return
    
    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'. Choose from: {list(MODELS.keys())}")
        return
    
    doc_path = Path(doc_path)
    if not doc_path.exists():
        print(f"Error: Document not found: {doc_path}")
        return
    
    # Load document
    print(f"\n{'='*70}")
    print(f"TEST EXTRACTION")
    print(f"{'='*70}")
    print(f"Document: {doc_path.name}")
    print(f"Method: {method}")
    print(f"Model: {MODELS[model_key]['display_name']}")
    print(f"{'='*70}\n")
    
    with open(doc_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Document length: {len(text)} characters\n")
    print("Starting extraction...")
    
    # Run extraction
    import time
    start = time.time()
    
    try:
        extraction_func = methods[method]
        model_id = MODELS[model_key]["model_id"]
        
        results = extraction_func(text, model_name=model_id)
        
        duration = time.time() - start
        
        # Handle new format: {"extracted_data": [...], "metadata": {...}}
        if isinstance(results, dict) and "extracted_data" in results:
            extracted_items = results["extracted_data"]
            metadata = results.get("metadata", {})
        else:
            extracted_items = results
            metadata = {}
        
        print(f"\n{'='*70}")
        print(f"SUCCESS!")
        print(f"{'='*70}")
        print(f"Extracted {len(extracted_items)} items in {duration:.1f} seconds\n")
        
        # Show first 3 results
        for i, item in enumerate(extracted_items[:3], 1):
            print(f"{i}. {item.get('statement', 'N/A')}")
            print(f"   Rationale: {item.get('rationale', 'N/A')[:100]}...")
            print()
        
        if len(extracted_items) > 3:
            print(f"... and {len(extracted_items) - 3} more items\n")
        
        # Show metadata if present
        if metadata:
            print(f"{'='*70}")
            print("METADATA:")
            print(f"{'='*70}")
            print(f"Temperature: {metadata.get('temperature', 'N/A')}")
            if 'thinking_process' in metadata:
                print(f"Thinking Process: {metadata['thinking_process'][:200]}...")
            if 'two_stage' in metadata:
                print(f"Two-stage extraction: {metadata['two_stage']}")
            if 'schema_guided' in metadata:
                print(f"Schema-guided: {metadata['schema_guided']}")
            print()
        
        # Show full results structure
        print(f"{'='*70}")
        print("SAMPLE OUTPUT STRUCTURE:")
        print(f"{'='*70}")
        import json
        print(json.dumps(extracted_items[0] if extracted_items else {}, indent=2, ensure_ascii=False))
        print()
        
    except Exception as e:
        duration = time.time() - start
        print(f"\n{'='*70}")
        print(f"FAILED after {duration:.1f} seconds")
        print(f"{'='*70}")
        print(f"Error: {str(e)}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Default test: smallest document with fastest method
    default_doc = "data/raw_md_files/Äußere Wendung.md"
    
    if len(sys.argv) > 1:
        doc = sys.argv[1]
    else:
        doc = default_doc
        print(f"No document specified. Using default: {doc}")
        print(f"Usage: python test_extraction.py <path_to_doc> [method] [model_key]\n")
    
    method = sys.argv[2] if len(sys.argv) > 2 else "naive"
    model_key = sys.argv[3] if len(sys.argv) > 3 else "qwen3-14b"
    
    test_single_extraction(doc, method, model_key)
