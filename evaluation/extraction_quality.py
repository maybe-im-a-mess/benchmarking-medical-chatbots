"""
Evaluate information extraction methods against ground truth mandatory topics.

This script measures:
- Recall: What percentage of mandatory topics were identified?
- Precision: What percentage of extracted items are actually mandatory?
- F1-score: Harmonic mean of precision and recall

Uses LLM to match extracted statements to ground truth topics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.llm_config import make_api_call


def load_ground_truth(gt_path: Path = Path("data/ground_truth.json")) -> Dict:
    """Load ground truth mandatory topics."""
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_extraction_result(model_key: str, document: str, method: str) -> Dict:
    """Load a single extraction result."""
    result_path = Path(f"data/processed/{model_key}/{document}_{method}.json")
    
    if not result_path.exists():
        return None
    
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_match_with_llm(extracted_item: str, gt_facts: List[Dict], model: str = "gpt-5-mini") -> Tuple[bool, str]:
    """
    Use LLM to check if an extracted item matches any ground truth fact.
    
    Returns:
        (is_match, matched_fact_id) - True if match found, with the fact_id
    """
    facts_text = "\n".join([
        f"[{fact['fact_id']}] {fact['content']}"
        for fact in gt_facts
    ])
    
    prompt = f"""Du bist ein Experte für medizinische Informationsextraktion.

AUFGABE: Bestimme, ob die extrahierte Aussage mit einem der Ground-Truth-Fakten übereinstimmt.

EXTRAHIERTE AUSSAGE:
{extracted_item}

GROUND TRUTH FAKTEN:
{facts_text}

ANLEITUNG:
- Die extrahierte Aussage muss inhaltlich das GLEICHE aussagen wie ein Ground-Truth-Fakt
- Umformulierungen sind OK, solange die Bedeutung identisch ist
- Wenn die Aussage nur TEILWEISE passt oder ein anderes Detail erwähnt, ist es KEIN Match
- Wenn die Aussage mehrere Fakten kombiniert, wähle den besten Match

Antworte NUR mit einem JSON-Objekt in diesem Format:
{{
  "match": true/false,
  "fact_id": "X.Y" oder null,
  "explanation": "Kurze Begründung"
}}"""

    system_message = "Du bist ein präziser medizinischer Informations-Evaluator. Antworte nur mit gültigem JSON."
    
    try:
        result = make_api_call(
            prompt=prompt,
            system_message=system_message,
            model_key=model,
            temperature=0.1,
            max_tokens=300
        )
        
        # Parse JSON response
        response_text = result.strip()
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        match_result = json.loads(response_text)
        
        return match_result["match"], match_result.get("fact_id")
        
    except Exception as e:
        print(f"    ⚠️  LLM matching error: {e}")
        return False, None


def evaluate_extraction(
    extraction_result: Dict,
    ground_truth_topics: List[Dict],
    model: str = "gpt-5-mini",
    verbose: bool = True
) -> Dict:
    """
    Evaluate a single extraction result against ground truth.
    
    Returns:
        Dict with precision, recall, F1, and detailed matching info
    """
    # Collect all ground truth facts
    gt_facts = []
    for topic in ground_truth_topics:
        for sub_topic in topic["sub_topics"]:
            gt_facts.append(sub_topic)
    
    total_gt_facts = len(gt_facts)
    
    # Get extracted items
    extracted_data = extraction_result.get("extracted_data", [])
    
    # Handle failed extractions
    if isinstance(extracted_data, dict) and "error" in extracted_data:
        return {
            "status": "failed",
            "error": extracted_data["error"],
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    if isinstance(extracted_data, list) and len(extracted_data) > 0 and isinstance(extracted_data[0], dict) and "error" in extracted_data[0]:
        return {
            "status": "failed",
            "error": extracted_data[0]["error"],
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    total_extracted = len(extracted_data)
    
    if total_extracted == 0:
        return {
            "status": "empty",
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "total_extracted": 0,
            "total_gt_facts": total_gt_facts,
            "true_positives": 0,
            "matched_facts": []
        }
    
    # Match each extracted item to ground truth
    matched_fact_ids = set()
    true_positives = 0
    matching_details = []
    
    if verbose:
        print(f"    Matching {total_extracted} extracted items against {total_gt_facts} ground truth facts...")
    
    for idx, item in enumerate(extracted_data):
        # Handle different extraction formats
        if isinstance(item, dict):
            item_text = item.get("statement") or item.get("fact") or item.get("information") or str(item)
        else:
            item_text = str(item)
        
        if verbose and (idx + 1) % 5 == 0:
            print(f"      Progress: {idx + 1}/{total_extracted} items checked...")
        
        is_match, fact_id = check_match_with_llm(item_text, gt_facts, model)
        
        if is_match and fact_id:
            matched_fact_ids.add(fact_id)
            true_positives += 1
            matching_details.append({
                "extracted_item": item_text[:100],
                "matched_fact_id": fact_id,
                "matched": True
            })
        else:
            matching_details.append({
                "extracted_item": item_text[:100],
                "matched_fact_id": None,
                "matched": False
            })
    
    # Calculate metrics
    precision = true_positives / total_extracted if total_extracted > 0 else 0.0
    recall = len(matched_fact_ids) / total_gt_facts if total_gt_facts > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "status": "success",
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_extracted": total_extracted,
        "total_gt_facts": total_gt_facts,
        "true_positives": true_positives,
        "matched_fact_count": len(matched_fact_ids),
        "matched_fact_ids": sorted(list(matched_fact_ids)),
        "matching_details": matching_details
    }


def evaluate_all_methods(
    document: str,
    model_keys: List[str] = ["qwen3-4b", "gpt-5-mini"],
    methods: List[str] = ["naive", "uie", "atomic", "cot"],
    eval_model: str = "gpt-5-mini",
    verbose: bool = True
) -> Dict:
    """
    Evaluate all extraction methods for a given document.
    
    Returns:
        Dict with results for each model/method combination
    """
    # Load ground truth
    ground_truth = load_ground_truth()
    
    # Map document filename to ground truth key
    doc_key = f"{document}.md"
    if doc_key not in ground_truth:
        print(f"❌ No ground truth found for document: {doc_key}")
        return {}
    
    gt_topics = ground_truth[doc_key]
    
    results = {}
    
    for model_key in model_keys:
        results[model_key] = {}
        
        for method in methods:
            if verbose:
                print(f"\n  📊 Evaluating: {model_key} / {method}")
            
            extraction_result = load_extraction_result(model_key, document, method)
            
            if extraction_result is None:
                if verbose:
                    print(f"    ⚠️  No extraction file found")
                results[model_key][method] = {"status": "not_found"}
                continue
            
            evaluation = evaluate_extraction(
                extraction_result,
                gt_topics,
                model=eval_model,
                verbose=verbose
            )
            
            results[model_key][method] = evaluation
            
            if verbose and evaluation["status"] == "success":
                print(f"    ✅ Precision: {evaluation['precision']:.2%} | "
                      f"Recall: {evaluation['recall']:.2%} | "
                      f"F1: {evaluation['f1_score']:.3f}")
    
    return results


def print_comparison_table(results: Dict):
    """Print a formatted comparison table of all methods."""
    print("\n" + "="*100)
    print("EXTRACTION QUALITY COMPARISON")
    print("="*100)
    
    # Collect all model/method combinations
    all_combos = []
    for model_key, methods_data in results.items():
        for method, evaluation in methods_data.items():
            if evaluation.get("status") == "success":
                all_combos.append({
                    "model": model_key,
                    "method": method,
                    "precision": evaluation["precision"],
                    "recall": evaluation["recall"],
                    "f1_score": evaluation["f1_score"],
                    "extracted": evaluation["total_extracted"],
                    "matched": evaluation["matched_fact_count"],
                    "gt_total": evaluation["total_gt_facts"]
                })
    
    # Sort by F1 score (descending)
    all_combos.sort(key=lambda x: x["f1_score"], reverse=True)
    
    # Print table header
    print(f"\n{'Rank':<6}{'Model':<15}{'Method':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Extracted':<12}{'Matched':<10}")
    print("-"*100)
    
    # Print rows
    for rank, combo in enumerate(all_combos, 1):
        print(f"{rank:<6}"
              f"{combo['model']:<15}"
              f"{combo['method']:<12}"
              f"{combo['precision']:<12.2%}"
              f"{combo['recall']:<12.2%}"
              f"{combo['f1_score']:<12.3f}"
              f"{combo['extracted']:<12}"
              f"{combo['matched']}/{combo['gt_total']:<8}")
    
    print("="*100 + "\n")
    
    # Winner
    if all_combos:
        winner = all_combos[0]
        print(f"🏆 BEST METHOD: {winner['model']} / {winner['method']} "
              f"(F1: {winner['f1_score']:.3f}, Precision: {winner['precision']:.2%}, Recall: {winner['recall']:.2%})")
    

def save_evaluation_results(document: str, results: Dict, output_dir: Path = Path("data/evaluation_results")):
    """Save detailed evaluation results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"extraction_quality_{document}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "document": document,
            "evaluation_model": "gpt-5-mini",
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate extraction methods against ground truth")
    parser.add_argument("--document", type=str, required=True, help="Document name (without .md)")
    parser.add_argument("--models", nargs="+", default=["qwen3-4b", "gpt-5-mini"], help="Model keys to evaluate")
    parser.add_argument("--methods", nargs="+", default=["naive", "uie", "atomic", "cot"], help="Methods to evaluate")
    parser.add_argument("--eval-model", default="gpt-5-mini", help="Model to use for evaluation matching")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    print(f"\n{'='*100}")
    print(f"EVALUATING EXTRACTION QUALITY: {args.document}")
    print(f"{'='*100}")
    
    results = evaluate_all_methods(
        document=args.document,
        model_keys=args.models,
        methods=args.methods,
        eval_model=args.eval_model,
        verbose=not args.quiet
    )
    
    print_comparison_table(results)
    
    save_evaluation_results(args.document, results)
