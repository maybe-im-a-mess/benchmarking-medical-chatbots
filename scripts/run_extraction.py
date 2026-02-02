import os
import json
import time
from pathlib import Path
from typing import Dict, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.llm_config import MODELS
from information_extraction.naive_llm import extract_statements_naive
from information_extraction.uie import extract_statements_schema
from information_extraction.atomic_fact_extraction import extract_statements_atomic
from information_extraction.cot_extraction import extract_statements_cot


# CONFIGURATION
INPUT_DIR = Path("data/raw_md_files")
OUTPUT_BASE_DIR = Path("data/processed")
PARALLEL = True  # Set to False for sequential processing
MAX_WORKERS = 1  # Local models (LM Studio) can only handle 1 request at a time

# Map method names to their functions for easy iteration
METHODS = {
    "naive": extract_statements_naive,
    "uie": extract_statements_schema,
    "atomic": extract_statements_atomic,
    "cot": extract_statements_cot
}

def check_directories(model_key: str):
    """Create output directory for this model if it doesn't exist."""
    output_dir = OUTPUT_BASE_DIR / model_key
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"Created output directory: {output_dir}")
    return output_dir

def save_result(doc_name: str, method_name: str, model_key: str, model_display: str, 
                data, execution_time: float, output_dir: Path):
    """Save the extraction result to a JSON file.
    
    Args:
        data: Either a list of dicts (standard) or dict with 'extracted_data' and 'metadata' (CoT)
    """
    output_filename = f"{doc_name}_{method_name}.json"
    output_path = output_dir / output_filename
    
    # Handle both standard format (list) and CoT format (dict with metadata)
    if isinstance(data, dict) and "extracted_data" in data:
        # CoT format: includes metadata
        extracted_items = data["extracted_data"]
        metadata = data.get("metadata", {})
    else:
        # Standard format: just the list
        extracted_items = data
        metadata = {}
    
    final_output = {
        "document": doc_name,
        "method": method_name,
        "model": {
            "key": model_key,
            "name": model_display
        },
        "execution_time_seconds": round(execution_time, 2),
        "item_count": len(extracted_items),
        "extracted_data": extracted_items
    }
    
    # Add metadata if present (e.g., thinking process from CoT)
    if metadata:
        final_output["metadata"] = metadata
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"  -> Saved: {output_filename} ({len(extracted_items)} items, {execution_time:.1f}s)")

def process_single_extraction(file_path: Path, method_name: str, extraction_func: Callable,
                              model_key: str, model_id: str, model_display: str, 
                              output_dir: Path) -> Dict:
    """Process a single extraction task. Designed for parallel execution.
    
    Returns:
        dict with status, timing, and result info
    """
    doc_name = file_path.stem
    start_time = time.time()
    
    try:
        # Read document
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        
        # Execute extraction
        result_data = extraction_func(text_content, model_name=model_id)
        duration = time.time() - start_time
        
        # Save results
        save_result(doc_name, method_name, model_key, model_display, result_data, duration, output_dir)
        
        # Get item count for reporting
        if isinstance(result_data, dict) and "extracted_data" in result_data:
            item_count = len(result_data["extracted_data"])
        else:
            item_count = len(result_data) if isinstance(result_data, list) else 0
        
        return {
            "status": "success",
            "doc": doc_name,
            "method": method_name,
            "items": item_count,
            "time": duration
        }
    
    except Exception as e:
        duration = time.time() - start_time
        error_log = [{"error": str(e), "status": "failed"}]
        save_result(doc_name, method_name, model_key, model_display, error_log, duration, output_dir)
        
        return {
            "status": "failed",
            "doc": doc_name,
            "method": method_name,
            "error": str(e),
            "time": duration
        }

def process_document(file_path: Path, model_key: str, model_id: str, model_display: str, output_dir: Path):
    """Run all 4 methods on a single document for one model."""
    doc_name = file_path.stem # e.g., "DRK_Info" without .md
    
    print(f"\n[{doc_name}] Reading file...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return

    # Run each method
    for method_name, extraction_func in METHODS.items():
        print(f"  Running: {method_name.upper()}...", end=" ", flush=True)
        start_time = time.time()
        
        try:
            # Execute the extraction function with model_name parameter
            result_data = extraction_func(text_content, model_name=model_id)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Save results
            save_result(doc_name, method_name, model_key, model_display, result_data, duration, output_dir)
            
        except Exception as e:
            print(f"FAILED!")
            print(f"      Reason: {str(e)}")
            # We save a failure log so you know what happened
            error_log = [{"error": str(e), "status": "failed"}]
            save_result(doc_name, method_name, model_key, model_display, error_log, 
                       time.time() - start_time, output_dir)

def main():
    """Process all documents with all models and all extraction methods."""
    
    # Get all markdown files from input directory
    md_files = sorted(INPUT_DIR.glob("*.md"))
    
    if not md_files:
        print(f"No .md files found in {INPUT_DIR}")
        return
    
    print(f"=" * 70)
    print(f"INFORMATION EXTRACTION PIPELINE")
    print(f"=" * 70)
    print(f"Found {len(md_files)} documents:")
    for f in md_files:
        print(f"  - {f.name}")
    print(f"\nModels to test: {len(MODELS)}")
    for key, config in MODELS.items():
        print(f"  - {key}: {config['display_name']}")
    print(f"\nMethods: {', '.join(METHODS.keys())}")
    print(f"Parallel processing: {'ENABLED' if PARALLEL else 'DISABLED'}")
    if PARALLEL:
        print(f"Max workers: {MAX_WORKERS}")
    print(f"=" * 70)
    
    total_start = time.time()
    
    # Iterate over each model
    for model_key, model_config in MODELS.items():
        model_id = model_config["model_id"]
        model_display = model_config["display_name"]
        
        print(f"\n{'='*70}")
        print(f"MODEL: {model_display} ({model_key})")
        print(f"{'='*70}")
        
        # Create output directory for this model
        output_dir = check_directories(model_key)
        
        if PARALLEL:
            # Parallel processing: submit all tasks and collect results
            print(f"\nSubmitting {len(md_files) * len(METHODS)} extraction tasks...")
            
            tasks = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all extraction tasks
                for file_path in md_files:
                    for method_name, extraction_func in METHODS.items():
                        future = executor.submit(
                            process_single_extraction,
                            file_path, method_name, extraction_func,
                            model_key, model_id, model_display, output_dir
                        )
                        tasks.append((future, file_path.stem, method_name))
                
                # Collect results as they complete
                for future, doc_name, method_name in tasks:
                    try:
                        result = future.result()
                        if result["status"] == "success":
                            print(f"✓ {doc_name:30s} | {method_name:10s} | "
                                  f"{result['items']:3d} items | {result['time']:6.1f}s")
                        else:
                            print(f"✗ {doc_name:30s} | {method_name:10s} | FAILED: {result.get('error', 'Unknown')[:40]}")
                    except Exception as e:
                        print(f"✗ {doc_name:30s} | {method_name:10s} | ERROR: {str(e)[:40]}")
        
        else:
            # Sequential processing (original behavior)
            for file_path in md_files:
                process_document(file_path, model_key, model_id, model_display, output_dir)
        
    total_duration = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"ALL DONE!")
    print(f"Total time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Results saved in: {OUTPUT_BASE_DIR.absolute()}/")
    print(f"  - {len(MODELS)} models x {len(md_files)} documents x {len(METHODS)} methods")
    print(f"  - Total extractions: {len(MODELS) * len(md_files) * len(METHODS)}")
    print(f"={'='*70}")

if __name__ == "__main__":
    main()