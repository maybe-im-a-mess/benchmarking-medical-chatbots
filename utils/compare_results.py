"""
Utility script to compare and analyze extraction results across models and methods.
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_all_results(base_dir: Path = Path("data/processed")) -> Dict:
    """Load all extraction results from the processed directory.
    
    Returns:
        Dict with structure: {model_key: {document: {method: result_data}}}
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist")
        return results
    
    # Iterate through model directories
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_key = model_dir.name
        
        # Load all JSON files in this model directory
        for json_file in model_dir.glob("*.json"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            doc_name = data["document"]
            method = data["method"]
            results[model_key][doc_name][method] = data
    
    return dict(results)


def get_summary_statistics(results: Dict) -> Dict:
    """Generate summary statistics across all extractions.
    
    Returns:
        Dict with statistics: item counts, execution times, failures, etc.
    """
    stats = {
        "by_model": defaultdict(lambda: {
            "total_extractions": 0,
            "successful": 0,
            "failed": 0,
            "total_items_extracted": 0,
            "total_time_seconds": 0.0,
            "by_method": defaultdict(lambda: {
                "count": 0,
                "items": [],
                "times": [],
                "failures": 0
            })
        }),
        "by_method": defaultdict(lambda: {
            "total_extractions": 0,
            "successful": 0,
            "failed": 0,
            "total_items": 0,
            "avg_items": 0.0,
            "avg_time": 0.0
        }),
        "by_document": defaultdict(lambda: {
            "extractions": 0,
            "avg_items_per_method": {}
        })
    }
    
    for model_key, model_data in results.items():
        for doc_name, doc_data in model_data.items():
            for method, extraction in doc_data.items():
                # Check if extraction failed
                extracted_data = extraction.get("extracted_data", [])
                is_failed = (
                    # Case 1: extracted_data is a dict with error info (API failure)
                    isinstance(extracted_data, dict) and "error" in extracted_data
                ) or (
                    # Case 2: extracted_data is a list but first item contains error
                    isinstance(extracted_data, list) and 
                    len(extracted_data) > 0 and
                    isinstance(extracted_data[0], dict) and
                    "error" in extracted_data[0]
                )
                
                item_count = 0 if is_failed else extraction.get("item_count", 0)
                exec_time = extraction.get("execution_time_seconds", 0)
                
                # By model stats
                stats["by_model"][model_key]["total_extractions"] += 1
                stats["by_model"][model_key]["by_method"][method]["count"] += 1
                stats["by_model"][model_key]["by_method"][method]["items"].append(item_count)
                stats["by_model"][model_key]["by_method"][method]["times"].append(exec_time)
                
                if is_failed:
                    stats["by_model"][model_key]["failed"] += 1
                    stats["by_model"][model_key]["by_method"][method]["failures"] += 1
                else:
                    stats["by_model"][model_key]["successful"] += 1
                    stats["by_model"][model_key]["total_items_extracted"] += item_count
                
                stats["by_model"][model_key]["total_time_seconds"] += exec_time
                
                # By method stats (global)
                stats["by_method"][method]["total_extractions"] += 1
                if is_failed:
                    stats["by_method"][method]["failed"] += 1
                else:
                    stats["by_method"][method]["successful"] += 1
                    stats["by_method"][method]["total_items"] += item_count
                
                # By document stats
                stats["by_document"][doc_name]["extractions"] += 1
                if method not in stats["by_document"][doc_name]["avg_items_per_method"]:
                    stats["by_document"][doc_name]["avg_items_per_method"][method] = []
                stats["by_document"][doc_name]["avg_items_per_method"][method].append(item_count)
    
    # Calculate averages
    for method_name, method_stats in stats["by_method"].items():
        if method_stats["successful"] > 0:
            method_stats["avg_items"] = method_stats["total_items"] / method_stats["successful"]
    
    return dict(stats)


def print_summary_report(results: Dict):
    """Print a comprehensive summary report of all extractions."""
    stats = get_summary_statistics(results)
    
    print("\n" + "="*80)
    print("EXTRACTION RESULTS SUMMARY")
    print("="*80)
    
    # Overview
    total_models = len(results)
    total_docs = len(set(doc for model_data in results.values() for doc in model_data.keys()))
    total_methods = len(set(method for model_data in results.values() 
                           for doc_data in model_data.values() 
                           for method in doc_data.keys()))
    
    print(f"\nOVERVIEW:")
    print(f"  Models tested: {total_models}")
    print(f"  Documents processed: {total_docs}")
    print(f"  Extraction methods: {total_methods}")
    
    # By Model
    print(f"\n{'-'*80}")
    print("RESULTS BY MODEL:")
    print(f"{'-'*80}")
    for model_key, model_stats in stats["by_model"].items():
        print(f"\n  {model_key.upper()}:")
        print(f"    Total extractions: {model_stats['total_extractions']}")
        print(f"    Successful: {model_stats['successful']} | Failed: {model_stats['failed']}")
        print(f"    Total items extracted: {model_stats['total_items_extracted']}")
        print(f"    Total execution time: {model_stats['total_time_seconds']:.1f}s ({model_stats['total_time_seconds']/60:.1f} min)")
        
        print(f"\n    By Method:")
        for method_name, method_data in model_stats["by_method"].items():
            avg_items = sum(method_data["items"]) / len(method_data["items"]) if method_data["items"] else 0
            avg_time = sum(method_data["times"]) / len(method_data["times"]) if method_data["times"] else 0
            print(f"      {method_name:10s}: {method_data['count']} runs, "
                  f"avg {avg_items:.1f} items, avg {avg_time:.1f}s, "
                  f"{method_data['failures']} failures")
    
    # By Method (across all models)
    print(f"\n{'-'*80}")
    print("RESULTS BY METHOD (all models):")
    print(f"{'-'*80}")
    for method_name, method_stats in stats["by_method"].items():
        print(f"\n  {method_name.upper()}:")
        print(f"    Total runs: {method_stats['total_extractions']}")
        print(f"    Successful: {method_stats['successful']} | Failed: {method_stats['failed']}")
        print(f"    Avg items per successful run: {method_stats['avg_items']:.1f}")
    
    # By Document
    print(f"\n{'-'*80}")
    print("RESULTS BY DOCUMENT:")
    print(f"{'-'*80}")
    for doc_name, doc_stats in stats["by_document"].items():
        print(f"\n  {doc_name}:")
        print(f"    Total extractions: {doc_stats['extractions']}")
        for method, items_list in doc_stats["avg_items_per_method"].items():
            avg = sum(items_list) / len(items_list) if items_list else 0
            print(f"      {method:10s}: avg {avg:.1f} items across {len(items_list)} model(s)")
    
    print(f"\n{'='*80}\n")


def compare_methods_for_document(results: Dict, document_name: str, model_key: str = None):
    """Compare extraction methods for a specific document.
    
    Args:
        results: Results dictionary from load_all_results()
        document_name: Name of document to compare
        model_key: Specific model to compare (or None for all models)
    """
    print(f"\n{'='*80}")
    print(f"METHOD COMPARISON FOR: {document_name}")
    if model_key:
        print(f"Model: {model_key}")
    print(f"{'='*80}\n")
    
    models_to_check = [model_key] if model_key else list(results.keys())
    
    for mk in models_to_check:
        if mk not in results or document_name not in results[mk]:
            print(f"No data found for model '{mk}', document '{document_name}'")
            continue
        
        print(f"Model: {mk}")
        print(f"{'-'*80}")
        
        doc_data = results[mk][document_name]
        
        for method_name in ["naive", "uie", "atomic", "cot"]:
            if method_name not in doc_data:
                print(f"  {method_name:10s}: NO DATA")
                continue
                
            extraction = doc_data[method_name]
            extracted_data = extraction.get("extracted_data", [])
            is_failed = (
                # Case 1: extracted_data is a dict with error info (API failure)
                isinstance(extracted_data, dict) and "error" in extracted_data
            ) or (
                # Case 2: extracted_data is a list but first item contains error
                isinstance(extracted_data, list) and 
                len(extracted_data) > 0 and
                isinstance(extracted_data[0], dict) and
                "error" in extracted_data[0]
            )
            
            if is_failed:
                error_msg = extraction["extracted_data"][0].get("error", "Unknown error")
                print(f"  {method_name:10s}: FAILED - {error_msg[:60]}")
            else:
                item_count = extraction.get("item_count", 0)
                exec_time = extraction.get("execution_time_seconds", 0)
                print(f"  {method_name:10s}: {item_count:3d} items in {exec_time:6.1f}s")
        
        print()


def export_comparison_csv(results: Dict, output_path: Path = Path("data/evaluation_results/comparison.csv")):
    """Export comparison data to CSV for further analysis."""
    import csv
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "document", "method", "item_count", "execution_time_seconds", "status"])
        
        for model_key, model_data in results.items():
            for doc_name, doc_data in model_data.items():
                for method, extraction in doc_data.items():
                    extracted_data = extraction.get("extracted_data", [])
                    is_failed = (
                        # Case 1: extracted_data is a dict with error info (API failure)
                        isinstance(extracted_data, dict) and "error" in extracted_data
                    ) or (
                        # Case 2: extracted_data is a list but first item contains error
                        isinstance(extracted_data, list) and 
                        len(extracted_data) > 0 and
                        isinstance(extracted_data[0], dict) and
                        "error" in extracted_data[0]
                    )
                    
                    status = "failed" if is_failed else "success"
                    item_count = 0 if is_failed else extraction.get("item_count", 0)
                    exec_time = extraction.get("execution_time_seconds", 0)
                    
                    writer.writerow([model_key, doc_name, method, item_count, exec_time, status])
    
    print(f"Exported comparison data to: {output_path}")


if __name__ == "__main__":
    # Load all results
    results = load_all_results()
    
    if not results:
        print("No results found. Run extraction first with run_extraction.py")
    else:
        # Print summary report
        print_summary_report(results)
        
        # Export CSV for further analysis
        export_comparison_csv(results)
        
        # Example: Compare methods for a specific document
        # Uncomment and modify as needed:
        # compare_methods_for_document(results, "DRK Geburtshilfe Infos", model_key="qwen3-4b")
